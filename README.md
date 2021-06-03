# A continual learning survey: Defying forgetting in classification tasks

This is the original source code for the Continual Learning survey paper _"A continual learning survey: Defying forgetting in classification tasks"_ published at TPAMI [[TPAMI paper]](https://ieeexplore.ieee.org/abstract/document/9349197) [[Open-Access paper]](https://arxiv.org/abs/1909.08383).

This work allows comparing the state-of-the-art in a fair fashion using the **Continual Hyperparameter Framework**, which sets the hyperparameters dynamically based on the stability-plasticity dilemma.
This addresses the longstanding problem in literature to set hyperparameters for different methods in a fair fashion, using ONLY the current task data (hence without using iid validation data, which is not available in continual learning).

The code contains a generalizing framework for 11 SOTA methods and 4 baselines in Pytorch. </br>
Implemented task-incremental methods are
<div align="center">
<p align="center"><b>
  SI | EWC | MAS | mean/mode-IMM | LWF | EBLL | PackNet |  HAT | GEM | iCaRL 
</b></p>
</div>

These are compared with 4 baselines:
<div align="center">
<p align="center"><b>
  Joint | Finetuning | Finetuning-FM | Finetuning-PM
</b></p>
</div>

- **Joint**: Learn from all task data at once with a single head (multi-task learning baseline).
- **Finetuning**: standard SGD
- **Finetuning with Full Memory replay**: Allocate memory dynamically to incoming tasks.
- **Finetuning with Partial Memory replay**: Divide memory a priori over all tasks.


This source code is released under a Attribution-NonCommercial 4.0 International
license, find out more about it in the [LICENSE file](LICENSE).




## Pipeline
**Reproducibility**: Results from the paper can be obtained from [src/main_'dataset'.sh](src/main_tinyimagenet.sh). 
Full pipeline example in [src/main_tinyimagenet.sh](src/main_tinyimagenet.sh) .

**Pipeline**: Constructing a custom pipeline typically requires the following steps.
1. Project Setup
    1. For all requirements see [requirements.txt](requirements.txt).
    Main packages can be installed as in
        ```
        conda create --name <ENV-NAME> python=3.7
        conda activate <ENV-NAME>
       
        # Main packages
        conda install -c conda-forge matplotlib tqdm
        conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

       # For GEM QP
        conda install -c omnia quadprog
       
       # For PackNet: torchnet 
       pip install git+https://github.com/pytorch/tnt.git@master
        ```
    1. Set paths in '[config.init](src/config.init)' (or leave default)
        1. '{tr,test}_results_root_path': where to save training/testing results.
        1. 'models_root_path': where to store initial models (to ensure same initial model)
        1. 'ds_root_path': root path of your datasets
    1. Prepare dataset: see [src/data](src/data)/"dataset"_dataprep.py (e.g. [src/data/tinyimgnet_dataprep.py](src/data/tinyimgnet_dataprep.py))
1. **Train** any out of the 11 SOTA methods or 4 baselines
    1. **Regularization-based/replay methods:** We run a *first task model dump*, for Synaptic Intelligence (SI) as it acquires importance weights during training. 
    Other methods start from this same initial model. 
    1. **Baselines/parameter isolation methods**: Start training sequence from scratch
1. **Evaluate** performance, sequence for testing on a task is saved in dictionary format under *test_results_root_path* defined in [config.init](src/config.init).
1. **Plot** the evaluation results, using one of the configuration files in [utilities/plot_configs](src/utilities/plot_configs)

## Implement Your Method
1. Find class "YourMethod" in [methods/method.py](src/methods/method.py). Implement the framework phases (documented in code).
1. Implement your task-based training script in [methods](src/methods): methods/"YourMethodDir". 
The class "YourMethod" will call this code for training/eval/processing of a single task. 

        
## Project structure
- [src/data](src/data): datasets and automated preparation scripts for Tiny Imagenet and iNaturalist.
- [src/framework](src/framework): the novel task incremental continual learning framework. 
**main.py** starts training pipeline, specify *--test* argument to perform evaluation with **eval.py**. 
- [src/methods](src/methods): all methods source code and **method.py** wrapper.
- [src/models](src/models): **net.py** all model preprocessing.
- [src/utilities](src/utilities): utils used across all modules and plotting.
- Config:
    - [src/data](src/data)/{datasets/models}: default datasets and models directory (see [config.init](src/config.init))
    - [src/results](src/results)/{train/test}: default training and testing results directory (see [config.init](src/config.init))


## Credits
- Consider citing our work upon using this repo.
  ```
  @ARTICLE{delange2021clsurvey,
    author={M. {Delange} and R. {Aljundi} and M. {Masana} and S. {Parisot} and X. {Jia} and A. {Leonardis} and G. {Slabaugh} and T. {Tuytelaars}},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={A continual learning survey: Defying forgetting in classification tasks}, 
    year={2021},volume={},number={},pages={1-1},
    doi={10.1109/TPAMI.2021.3057446}}
  ```
- Thanks to Huawei for funding this project.
- Checkout the CL [Avalanche project](https://github.com/ContinualAI/avalanche) for benchmark setups of TinyImagenet and iNaturalist2018.
- Thanks to the following repositories:
    - https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    - https://github.com/facebookresearch/GradientEpisodicMemory
    - https://github.com/arunmallya/packnet
    - https://github.com/joansj/hat
* If you want to join the Continual Learning community, checkout https://www.continualai.org

## Support
* If you have troubles, please open a Git issue.
* Have you defined your method in the framework and want to share it with the community? Send a pull request!
