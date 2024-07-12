## Introduction
This is the codebase for ELSA forked from TinyViT.

## Pending List
- [ ] Environmental requirements
- [ ] Quantization results
- [ ] ConvNext results
- [ ] DominoSearch/ERK results  

## Training supernet with distillation
```
python -m torch.distributed.launch --nproc_per_node 8 \
main.py --cfg configs/sparse/deit_small_vanilla.yaml \
--data-path /imagenet/ \
--batch-size 128 \
--resume {PATH_DEIT_CHECKPOINT} \
--opts DISTILL.TEACHER_LOGITS_PATH {PATH_TO_TEACHER_LOGITS}
```

## Experimental Results
| Model            | Sparsity Pattern | FLOPs          | Accuracy |
|------------------|------------------|----------------|----------|
| DeiT-S           | Dense            | 4.7G           | 79.824   |
| ELSA-DeiT-S-2:4  | Uniform 2:4      | 2.5G (1.00×)   | 79.144(no_eps)/78.654(eps) |
| ELSA-DeiT-S-N:4  | Layer-wise N:4   | 2.0G (1.25×)   | 78.340   |
| DeiT-B           | Dense            | 17.6G          | 81.806   |
| ELSA-DeiT-B-2:4  | Uniform 2:4      | 9.2G (1.00×)   | 81.668   |
| ELSA-DeiT-B-N:4  | Layer-wise N:4   | 6.9G (1.30×)   | 81.538   |
|                  |                  | 6.0G (1.53×)   | 81.358   |
| Swin-S           | Dense            | 8.7G           | 83.170   |
| ELSA-Swin-S-2:4  | Uniform 2:4      | 4.6G (1.00×)   | 82.814   |
| ELSA-Swin-S-N:4  | Layer-wise N:4   | 4.0G (1.15×)   | 82.760   |
|                  |                  | 3.5G (1.31×)   | 82.536   |
| Swin-B           | Dense            | 15.4G          | 83.416   |
| ELSA-Swin-B-2:4  | Uniform 2:4      | 8.0G (1.00×)   | 83.124   |
| ELSA-Swin-B-N:4  | Layer-wise N:4   | 5.9G (1.33×)   | 82.982   |
|                  |                  | 5.3G (1.51×)   | 82.794   |

- You can reproduce the results using the config and weights provided (all included in `scripts.sh`). Below is an example:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/deit_small_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_small_no_eps2.pth \
--eval
```

- You can download trained supernet weight from links below:
| Model           | Download | 
|-----------------|----------|
| ELSA-DeiT-S-N:4 | [no_eps](https://drive.google.com/file/d/1-76rgS2xA2dHTCWHascRaVoV2LEzRami/view)/[eps](https://drive.google.com/file/d/1XTjraDO7U-j80ZwnfashO3bZe3gsYw5x/view) |
| ELSA-DeiT-B-N:4 | [eps](https://drive.google.com/file/d/1WOv-8IdDH3T11YXdcMZG2zq8IhMuX1aV/view) |
| ELSA-Swin-S-N:4 | [eps](https://drive.google.com/file/d/1BEEeAG86cOlw3hjN2_ivRuSdxlIAtxSx/view) |
| ELSA-Swin-B-N:4 | [eps](https://drive.google.com/file/d/1mkiqoSchNfkisiYTp0ylYl8tvtZoNlyP/view) |