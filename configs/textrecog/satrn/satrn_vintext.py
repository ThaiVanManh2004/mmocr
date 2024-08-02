default_scope = "mmocr"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
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
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=3e-4))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="MultiStepLR", milestones=[3, 4], end=5),
]

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
    type="SATRN",
    backbone=dict(type="ShallowCNN", input_channels=3, hidden_dim=512),
    encoder=dict(
        type="SATRNEncoder",
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1,
    ),
    decoder=dict(
        type="NRTRDecoder",
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8,
        module_loss=dict(type="CEModuleLoss", flatten=True, ignore_first_char=True),
        dictionary=dictionary,
        max_seq_len=25,
        postprocessor=dict(type="AttentionPostprocessor"),
    ),
    data_preprocessor=dict(
        type="TextRecogDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", ignore_empty=True, min_size=2),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(type="Resize", scale=(100, 32), keep_ratio=False),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(100, 32), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
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
            [dict(type="Resize", scale=(100, 32), keep_ratio=False)],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type="LoadOCRAnnotations", with_text=True)],
            [
                dict(
                    type="PackTextRecogInputs",
                    meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
                )
            ],
        ],
    ),
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
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)

auto_scale_lr = dict(base_batch_size=64 * 8)
