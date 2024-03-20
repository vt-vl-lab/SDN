# SDN: Scene Debiasing Network for Action Recognition in PyTorch
We release the code of the "Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition". The code is built upon the [3D-ResNets-PyTorch codebase](https://github.com/kenshohara/3D-ResNets-PyTorch).

For the details, visit our [project website](http://chengao.vision/SDN/) or see our [full paper](https://papers.nips.cc/paper/8372-why-cant-i-dance-in-the-mall-learning-to-mitigate-scene-bias-in-action-recognition.pdf).

## Reference
[Jinwoo Choi](https://sites.google.com/site/jchoivision/), [Chen Gao](https://gaochen315.github.io/), [Joseph C. E. Messou](https://josephcmessou.weebly.com/about.html), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/index.html). Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition. Neural Information Processing Systems (NeurIPS) 2019.

```
@inproceedings{choi2019sdn,
    title = {Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition},
    author = {Choi, Jinwoo and Gao, Chen and Messou, C. E. Joseph and Huang, Jia-Bin},
    booktitle={NeurIPS},
    year={2019}
}
```

## Requirements
This codebase was developed and tested with:
- Python 3.6
- PyTorch 0.4.1
- torchvision 0.2.1
- CUDA 9.0
- CUDNN 7.1
- GPU: 2xP100 

You can find dependencies from `sdn_packages.txt`

You can install dependencies by
```
pip install -r sdn_packages.txt 
```

## Datasets
### Prepare your dataset
**1. Download and pre-process data**
- Follow the [3D-ResNets-PyTorch instruction](https://github.com/kenshohara/3D-ResNets-PyTorch#preparation).

**2. Download scene and human detection data numpy files**
- [Download the Mini-Kinetics scene pseudo labels](https://filebox.ece.vt.edu/~jinchoi/files/sdn/places_data.zip)
- [Download the Mini-Kinetics human detections](https://filebox.ece.vt.edu/~jinchoi/files/sdn/detections.zip)

## Train
### Training on a source dataset (mini-Kinetics)
**- Baseline model without any debiasing**
```
 python train.py 
 --video_path <your dataset dir path> \
 --annotation_path <your dataset dir path>/kinetics.json \
 --result_path <path to save your model> \
 --root_path <your dataset dir path> \
 --dataset kinetics \
 --n_classes 200 \
 --n_finetune_classes 200 \
 --model resnet \
 --model_depth 18 \
 --resnet_shortcut A \
 --batch_size 32 \
 --val_batch_size 16 \
 --n_threads 16 \
 --checkpoint 1 \
 --ft_begin_index 0 \
 --is_mask_adv \
 --learning_rate 0.0001 \
 --weight_decay 1e-5 \
 --n_epochs 100 \
 --pretrain_path <pre-trained model file path>
 ```
 
**- SDN model with scene adversarial loss only**
```
python train.py \
--video_path <your dataset dir path> \
--annotation_path <your dataset dir path>/kinetics.json \
--result_path <path to save your model> \
--root_path <your dataset dir path> \
--dataset kinetics_adv \
--n_classes 200 \
--n_finetune_classes 200 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 32 \
--val_batch_size 16 \
--n_threads 16 \
--checkpoint 1 \
--ft_begin_index 0 \
--num_place_hidden_layers 3 \
--new_layer_lr 1e-2 \
--learning_rate 1e-4 \
--warm_up_epochs 5 \
--weight_decay 1e-5 \
--n_epochs 100 \
--place_pred_path <full path of your kinetics pseudo scene labels> \
--is_place_adv \
--is_place_soft \
--alpha 1.0 \
--is_mask_adv \
--num_places_classes 365 \
--pretrain_path <pre-trained model file path>
```

**- Full SDN model with 1) scene adversarial loss and 2) human mask confussion loss**
```
python train.py \
--video_path <your dataset dir path> \
--annotation_path <your dataset dir path>/kinetics.json \
--result_path <path to save your model> \
--root_path <your dataset dir path> \
--dataset kinetics_adv_msk \
--n_classes 200 \
--n_finetune_classes 200 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 32 \
--val_batch_size 16 \
--n_threads 16 \
--checkpoint 1 \
--ft_begin_index 0 \
--num_place_hidden_layers 3 \
--num_human_mask_adv_hidden_layers 1 \
--new_layer_lr 1e-4 \
--learning_rate 1e-4 \
--warm_up_epochs 0 \
--weight_decay 1e-5 \
--n_epochs 100 \
--place_pred_path <full path of your kinetics pseudo scene labels> \
--is_place_adv \
--is_place_soft \
--is_mask_entropy \
--alpha 0.5 \
--mask_ratio 1.0 \
--slower_place_mlp \
--not_replace_last_fc \
--num_places_classes 365 \
--human_dets_path <full path of your kinetics human detections> \
--pretrain_path <pre-trained model file path: e.g., your SDN model with scene adversarial loss only>
```

### Finetuning on target datasets
#### [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) as an example
```
python train.py \
--dataset diving48 \
--root_path <your dataset path> \
--video_path <your dataset path> \
--n_classes 200 \
--n_finetune_classes 48 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--ft_begin_index 0 \
--batch_size 32 \
--val_batch_size 16 \
--n_threads 4 \
--checkpoint 1 \
--learning_rate 0.005 \
--weight_decay 1e-5 \
--n_epochs $epoch_ft \
--is_mask_adv \
--annotation_path $anno_path \
--result_path <path to save your fine-tuned model> \
--pretrain_path <pre-trained model file path: e.g., your full SDN model path>
```

## Test
```
python train.py \
--dataset diving48 \
--root_path <your dataset path> \
--video_path <your dataset path> \
--n_finetune_classes 48 \
--n_classes 48 \
--model resnet \
--model_depth 18 \
--resnet_shortcut A \
--batch_size 32 \
--val_batch_size 16 \
--n_threads 4 \
--test \
--test_subset val \
--no_train \
--no_val \
--is_mask_adv \
--annotation_path $anno_path \
--result_path <path (dir) to save your fine-tuned model> \
--resume_path <path (the model checkpoint file) to save your fine-tuned model>
```
This step will generate `val.json` file under `$result_path`.

## Evaluation
```
python utils/eval_diving48.py \
--annotation_path $anno_path \
--prediction_path <path to your test result file (val.json) generated from the test step>
```

## Pre-trained model weights provided
[Download the pre-trained weights](https://www.dropbox.com/scl/fi/j2pgucu8gvpz3jp5ygl91/pre-trained_weights.tar?rlkey=gicecxrpj2o7ipjmhmx0hlcrl&dl=0)

## Acknowledgments
This code is built upon [3D-ResNets-PyTorch codebase](https://github.com/kenshohara/3D-ResNets-PyTorch). We thank to Kensho Hara. 
