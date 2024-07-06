# --------------------------------------------------------
# TinyViT Model Builder
# Copyright (c) 2022 Microsoft
# --------------------------------------------------------

from .tiny_vit import TinyViT
from .sparse_deit import SparseVisionTransformer
from .deit import VisionTransformer
from .sparse_swin_transformer import SparseSwinTransformer
from .swin import SwinTransformer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'tiny_vit':
        M = config.MODEL.TINY_VIT
        model = TinyViT(img_size=config.DATA.IMG_SIZE,
                        in_chans=M.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dims=M.EMBED_DIMS,
                        depths=M.DEPTHS,
                        num_heads=M.NUM_HEADS,
                        window_sizes=M.WINDOW_SIZES,
                        mlp_ratio=M.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        mbconv_expand_ratio=M.MBCONV_EXPAND_RATIO,
                        local_conv_size=M.LOCAL_CONV_SIZE,
                        layer_lr_decay=config.TRAIN.LAYER_LR_DECAY,
                        )
    elif model_type == 'clip_vit_large14_224':
        from .clip import CLIP
        kwargs = {
            'embed_dim': 768, 'image_resolution': 224,
            'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14,
            "num_classes": config.MODEL.NUM_CLASSES,
        }
        model = CLIP(**kwargs)
    elif model_type == 'sparse_deit':
        model = SparseVisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTH,
            num_heads = config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio = config.MODEL.DEIT.MLP_RATIO,
            qkv_bias = config.MODEL.DEIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == 'deit':
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTH,
            num_heads = config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio = config.MODEL.DEIT.MLP_RATIO,
            qkv_bias = config.MODEL.DEIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'sparse_swin':
        model = SparseSwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
