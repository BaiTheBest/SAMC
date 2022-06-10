import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import cv2
import quadprog

from .common import MLP, ResNet18

# Import GradCAM class
from pytorch_grad_cam import GradCAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
            self.target_layer = self.net.layer4[-1]
            self.cam = GradCAM(model=self.net, target_layer=self.target_layer, use_cuda=True)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = {}
        self.memory_labs = {}
        self.pxl_needed = {}
        # self.memory_data = torch.FloatTensor(
        #     n_tasks, self.n_memories, n_inputs)
        # self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        # if args.cuda:
        #     self.memory_data = self.memory_data.cuda()
        #     self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # set the threshold for GradCAM mast
        self.theta = args.theta

        # set parameters for dynamic memory
        self.max_pxl = self.n_memories * self.n_inputs
        self.pxl_stored = np.zeros(n_tasks)
        self.img_stored = np.zeros(n_tasks)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):

        # mini-batch size
        bsz = y.data.size(0)

        # new task comes
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            # initialize episodic memory for the new task
            self.memory_data[t] = torch.FloatTensor(bsz, self.n_inputs)
            self.memory_labs[t] = torch.LongTensor(bsz)
            self.pxl_needed[t] = np.zeros(bsz)
            if self.gpu:
                self.memory_data[t].cuda()
                self.memory_labs[t].cuda()

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task], past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
        self.opt.step()
        
        # Update ring buffer storing examples from current task with memory efficiency by GradCAM
        tmp_x_data = x.data # tensor shape: bsz by 3*32*32
        tmp_x_data = tmp_x_data.view(tmp_x_data.size(0),3,32,32) # convert the shape to be 4D
        original_x = tmp_x_data.clone()
        original_x = np.float32(original_x.detach().cpu())

        target_category = None # y.detach().cpu().tolist()
        grayscale_cam = self.cam(input_tensor=tmp_x_data, target_category=target_category, task_index=t)
        masked_x = torch.empty_like(tmp_x_data)
        pxl_needed = np.zeros(bsz) # number of non-zero pixels for each image within this mini-batch

        tmp_x_data = tmp_x_data * 255.0 # convert image back to 0 - 255 value range
        for i in range(bsz):
            tmp_x = np.uint8(tmp_x_data[i].detach().cpu())
            a = original_x[i,:]
            a = np.rollaxis(a, 0, 3)
            tmp_x = np.rollaxis(tmp_x, 0, 3)
            tmp_gc = grayscale_cam[i,:]
            # get GradCAM mask by threshold theta
            mask = np.where(tmp_gc < self.theta, 1, 0)
            # calculate number of non-zero pixels of this image after applying the mask
            pxl_needed[i] = 3*32*32 - 3*np.count_nonzero(mask)
            # mask the image
            mask = np.uint8(mask)
            tmp_inpainted = cv2.inpaint(tmp_x, mask, 3, cv2.INPAINT_TELEA)
            tmp_inpainted = tmp_inpainted / 255.0
            tmp_inpainted = np.rollaxis(tmp_inpainted, 2, 0)
            masked_x[i] = torch.from_numpy(tmp_inpainted).to(masked_x)

        # get the mini-batch data after GradCAM
        masked_x = masked_x.view(masked_x.size(0),-1)
        masked_x.cuda()

        total_pxl_needed = np.sum(pxl_needed)

        # now we begin to store the mini-batch into episodic memory with dynamic size
        if self.img_stored[t] == 0:
            self.memory_data[t].copy_(masked_x)
            self.img_stored[t] += bsz
            self.pxl_stored[t] += total_pxl_needed
            self.memory_labs[t].copy_(y)
            self.pxl_needed[t] = pxl_needed

        elif self.pxl_stored[t] + total_pxl_needed <= self.max_pxl:
            self.memory_data[t] = torch.cat((self.memory_data[t].cuda(),masked_x),0)
            self.img_stored[t] += bsz
            self.pxl_stored[t] += total_pxl_needed
            self.memory_labs[t] = torch.cat((self.memory_labs[t].cuda(),y))
            self.pxl_needed[t] = np.concatenate((self.pxl_needed[t], pxl_needed), axis=None)

        else:
            pxl_released = 0
            for k in range(int(self.img_stored[t])):
                pxl_released += self.pxl_needed[t][k]
                if self.pxl_stored[t] + total_pxl_needed - pxl_released <= self.max_pxl:
                    # remove images up to the current one from memory
                    self.memory_data[t] = self.memory_data[t][k+1:,]
                    self.memory_labs[t] = self.memory_labs[t][k+1:]
                    self.pxl_needed[t] = self.pxl_needed[t][k+1:]
                    self.img_stored[t] -= k+1
                    self.pxl_stored[t] -= pxl_released
                    # now store the current mini-batch into memory
                    self.memory_data[t] = torch.cat((self.memory_data[t].cuda(),masked_x),0)
                    self.img_stored[t] += bsz
                    self.pxl_stored[t] += total_pxl_needed
                    self.memory_labs[t] = torch.cat((self.memory_labs[t].cuda(),y))
                    self.pxl_needed[t] = np.concatenate((self.pxl_needed[t], pxl_needed), axis=None)
                    break
                else:
                    continue

