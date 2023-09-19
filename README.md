# SAMC
This is the GitHub repo for the paper *"[Saliency-Augmented Memory Completion for Continual Learning](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977653.ch28)"* published at SIAM SDM 2023.

Our code is built upon the following repo:

https://github.com/facebookresearch/GradientEpisodicMemory/tree/master/model

which is for the paper "Gradient Episodic Memory for Continual Learning"


We also leverage the Grad-CAM implementation in PyTorch from the following repo:

https://github.com/jacobgil/pytorch-grad-cam


## Instructions

Here, we include our code of the proposed method SAMC on Split CIFAR-100. Other datasets follow directly. To replicate the experiment, please:

1. Create an empty folder called "data". Generate the Split CIFAR-100 dataset "cifar100.pt" at "data" folder. The detailed procedure has been shown in the repo of GEM and please refer to its repo. 

2. Run the following command: 

python main.py --n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt --cuda yes --seed 0 --model samc --n_epochs 1 --lr 0.1 --n_memories 10 --memory_strength 0.5 --theta 0.6

Remark: Our code has been tested in Anaconda environment with conda 4.10.3, Python 3.8.3, and PyTorch 1.6.0.

## Reference and Citation

If you find our paper or code useful, please consider citing our work :)

        @inproceedings{bai2023saliency,
          title={Saliency-Augmented Memory Completion for Continual Learning},
          author={Bai, Guangji and Ling, Chen and Gao, Yuyang and Zhao, Liang},
          booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
          pages={244--252},
          year={2023},
          organization={SIAM}
        }
