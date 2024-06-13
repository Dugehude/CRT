# CRT

Source-free domain adaptation with class relationships transfer

## Abstract

​		**In this work, we tackle the challenging task of source-free domain adaptation (SFDA), which utilizes only a pre-trained model to transfer knowledge to an unlabeled target domain without access to source data. Current methods mainly overlook the inherent inter-class relationships or unilaterally presume them domain-invariant and directly utilize source knowledge, leading to negative transfer. To address these issues, we propose a novel source-free domain adaptation method with class relationship transfer (CRT). Specifically, our approach explores and integrates dynamic class relationships of label space and feature space to constrain constrastive learning within target domain, thereby comprehensively achieving the class relationship adaptation. Furthermore, we introduce a coordination optimization algorithm by virtual samples generation to mitigate the contradictions between different loss functions, which treats constrastive learning as the privilege in SFDA instead of merely an isolated performance enhancement trick. We have conducted comprehensive experiments on multiple datasets, and compared our findings with those from various existing algorithms. The results indicate that our proposed algorithm achieves state-of-the-art (SOTA) performance.**

## Requisites

* python==3.6.8
* pytorch==1.1.0
* torchvision==0.3.0
* numpy, scipy, sklearn, argparse, tqdm

## Datasets

* [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)
* [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)
* [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

## Train

1. source model training

   * office-31 dataset source model training in domain A (s=0)

   ~~~shell
   cd CRT/
   python source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset office --max_epoch 100 --s 0
   ~~~

   * office-home dataset source model training in domain A (s=0)

   ~~~shell
   cd CRT/
   python source.py --trte val --da uda --output ckps/source/ --gpu_id 0 --dset office-home --max_epoch 100 --s 0
   ~~~

   * VisDa dataset  source model training 

   ~~~shell
   python source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 20 --s 0
   ~~~

2. source-free domain adaptation

   * office-31 dataset source-free domain adaptation in AD task (s=0, t=1)

   ~~~shell
   python target.py --s 0 --t 1 --sset office
   ~~~

   * office-home dataset source-free domain adaptation in AC task (s=0, t=1)

   ~~~shell
   python target.py --s 0 --t 1 --sset office-home
   ~~~

   * VisDa  dataset source-free domain adaptation in TV task (s=0, t=1)

   ~~~shell
   python target.py --s 0 --t 1 --sset VISDA --lr 5e-5 --net resnet101 --batch_size 64 --max_e 30 --vat_e 30 --ema_alpha 0.999
   ~~~

## Acknowledgement

This code is based on [SHOT](https://github.com/tim-learn/SHOT)

