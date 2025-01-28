_base_ = [
    "../_base_/models/upernet_vim.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="MM_FastVim",
        img_size=512,
        patch_size=16,
        scanpath_type="rowwise",
        stride=16,
        in_chans=3,
        embed_dim=384,
        depth=24,
        out_indices=[5, 11, 17, 23],
        pretrained=None,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        if_abs_pos_embed=True,
        final_pool_type="all",
        use_norm_after_ssm=True,
        rotate_every_block=True,
        load_ema=True,
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384], num_classes=150
    ),  # channels=384),
    auxiliary_head=dict(in_channels=384, num_classes=150),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "A_log": dict(decay_mult=0.0),
            "D": dict(decay_mult=0.0),
            "A_b_log": dict(decay_mult=0.0),
            "D_b": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
