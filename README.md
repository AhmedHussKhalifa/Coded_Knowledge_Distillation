# A Coded Knowledge Distillation framework for Image Classification Based on Adaptive JPEG Encoding


Official implementation of paper A Coded Knowledge Distillation (CKD) framework for Image Classification Based on Adaptive JPEG Encoding , Submitted to [Pattern Recognition Journal](https://www.sciencedirect.com/journal/pattern-recognition).  
By Ahmed H. Salamah, Shayan Mohajer Hamidi, En-Hui Yang.



:fire: **DIST: a simple and effective KD method.**

## Updates  

* **July 24, 2023**: Sumbitted to Pattern Recognition Journal.

* **Ocotober 25, 2023**: Release code.

## Getting started  
### Clone training code  
```shell
git clone https://github.com/AhmedHussKhalifa/Coded_Knowledge_Distillation 
cd Coded_Knowledge_Distillation 
```

This repo:


**(1) benchmarks 12 state-of-the-art knowledge distillation methods in PyTorch, including:**

(KD) - Distilling the Knowledge in a Neural Network  
(FitNet) - Fitnets: hints for thin deep nets  
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer  
(SP) - Similarity-Preserving Knowledge Distillation  
(CC) - Correlation Congruence for Knowledge Distillation  
(VID) - Variational Information Distillation for Knowledge Transfer  
(RKD) - Relational Knowledge Distillation  
(PKT) - Probabilistic Knowledge Transfer for deep representation learning  
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons  
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer  
(FSP) - A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning  
(NST) - Like what you like: knowledge distill via neuron selectivity transfer 

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    - `--ckd`: to enable the coded teacher.
    
    Therefore, the command for running CKD is something like:
    ```
    python train_student_ckd.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --ckd ckd --distill kd --model_s resnet8x4 -a 0.1 -a 0.9 --trial 1
    ```

4. Use pre-trained backbone and train all auxiliary classifiers. 

The pre-trained backbone weights follow .pth files downloaded from repositories of [CRD](https://github.com/HobbitLong/RepDistiller) and [SSKD](https://github.com/xuguodong03/SSKD).

This repo was based on [CRD](https://github.com/HobbitLong/RepDistiller) to ensure reproducibility. 

## Corresponding Author

For any questions, please contact Ahmed H. Salamah (ahamsala@uwaterloo.ca).
