
# region deit-small
# Eval deit-small u24 (79.144%)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/deit_small_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_small_no_eps2.pth \
--eval

# Eval deit-small u24 (78.654%)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/deit_small_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_small_eps2_50.pth \
--eval

# Eval deit-small 2G
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29502 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_deit_small_1.96G_78.346.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_small_eps2_50.pth \
--eval
# endregion

# region deit-base
# Eval deit-base u24
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/deit_base_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_base_eps2_50.pth \
--eval

# Eval deit-base 7G
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29502 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_deit_base_6.88G_81.538.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_base_eps2_50.pth \
--eval

# Eval deit-base 6G
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29503 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_deit_base_6.01G_81.358.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/deit_base_eps2_50.pth \
--eval

# endregion

# region swin-small
# Eval swin-small u24
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/swin_small_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_small_affine_eps2_50.pth \
--eval

# Eval swin-small 4G
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29502 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_swin_small_3.99G_82.760.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_small_affine_eps2_50.pth \
--eval

# Eval swin-small 3.5G
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29503 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_swin_small_3.50G_82.548.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_small_affine_eps2_50.pth \
--eval

# endregion

# region swin-base
# Eval swin-base u24
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29501 main.py \
--cfg configs/sparse_subnet/uniform/swin_base_baseline_24.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_base_no_eps2.pth \
--eval

# Eval swin-base 5.9G
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29502 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_swin_base_5.92G_82.978.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_base_no_eps2.pth \
--eval

# Eval swin-base 5.3G
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 29503 main.py \
--cfg configs/sparse_subnet/ELSA/sparse_swin_base_5.33G_82.768.yaml \
--data-path /dataset/imagenet \
--batch-size 128 \
--resume supernet_weights/swin_base_no_eps2.pth \
--eval

# endregion


