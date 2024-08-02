dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/vietnamese_digits_symbols_space.txt",
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True,
)

model = dict(
    type="ASTER",
    preprocessor=dict(
        type="STN",
        in_channels=3,
        resized_image_size=(32, 64),
        output_image_size=(32, 100),
        num_control_points=20,
    ),
    backbone=dict(
        type="ResNet",
        in_channels=3,
        stem_channels=[32],
        block_cfgs=dict(type="BasicBlock", use_conv1x1="True"),
        arch_layers=[3, 4, 6, 6, 3],
        arch_channels=[32, 64, 128, 256, 512],
        strides=[(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)],
        init_cfg=[
            dict(type="Kaiming", layer="Conv2d"),
            dict(type="Constant", val=1, layer="BatchNorm2d"),
        ],
    ),
    encoder=dict(type="ASTEREncoder", in_channels=512),
    decoder=dict(
        type="ASTERDecoder",
        max_seq_len=25,
        in_channels=512,
        emb_dims=512,
        attn_dims=512,
        hidden_size=512,
        postprocessor=dict(type="AttentionPostprocessor"),
        module_loss=dict(type="CEModuleLoss", flatten=True, ignore_first_char=True),
        dictionary=dictionary,
    ),
    data_preprocessor=dict(
        type="TextRecogDataPreprocessor",
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", ignore_empty=True, min_size=5),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(type="Resize", scale=(256, 64)),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(256, 64)),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio", "instances"),
    ),
]

val_pipeline = test_pipeline

tta_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=0, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=1, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
                dict(
                    type="ConditionApply",
                    true_transforms=[
                        dict(
                            type="ImgAugWrapper",
                            args=[dict(cls="Rot90", k=3, keep_size=False)],
                        )
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                ),
            ],
            [dict(type="Resize", scale=(256, 64))],
            [dict(type="LoadOCRAnnotations", with_text=True)],
            [
                dict(
                    type="PackTextRecogInputs",
                    meta_keys=(
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "valid_ratio",
                        "instances",
                    ),
                )
            ],
        ],
    ),
]

default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffer=dict(type="SyncBuffersHook"),
    visualization=dict(
        type="VisualizationHook",
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False,
    ),
)
# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=True)

load_from = "epoch_0.pth"
resume = True

# Evaluation
val_evaluator = [
    dict(type="WordMetric", mode=["exact", "ignore_case", "ignore_case_symbol"]),
    dict(type="CharMetric"),
]
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="TextRecogLocalVisualizer", name="visualizer", vis_backends=vis_backends
)

tta_model = dict(type="EncoderDecoderRecognizerTTAModel")

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW", lr=4e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05
    ),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=6, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning policy
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=6, eta_min=4e-6, convert_to_iter_based=True)
]

_base_ = ["../_base_/datasets/vintext.py"]

# dataset settings
train_list = [_base_.vintext_textrecog_train]
val_list = [_base_.vintext_textrecog_val]
test_list = [_base_.vintext_textrecog_test]

train_dataset = dict(type="ConcatDataset", datasets=train_list, pipeline=train_pipeline)
val_dataset = dict(type="ConcatDataset", datasets=val_list, pipeline=val_pipeline)
test_dataset = dict(type="ConcatDataset", datasets=test_list, pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=1024,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

auto_scale_lr = dict(base_batch_size=1024)

val_dataloader = dict(
    batch_size=1024,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)
