## Introduction
This is the new codebase for SparsifyViT forked from TinyViT.

## Training supernet with distillation
```
python -m torch.distributed.launch --nproc_per_node 8 \
main.py --cfg configs/sparse/deit_small_vanilla.yaml \
--data-path /imagenet/ \
--batch-size 128 \
--resume {PATH_DEIT_CHECKPOINT} \
--opts DISTILL.TEACHER_LOGITS_PATH {PATH_TO_TEACHER_LOGITS}
```