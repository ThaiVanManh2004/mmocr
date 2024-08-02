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
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=1e-3))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=5, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="MultiStepLR", milestones=[3, 4], end=5),
]

dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/vietnamese_digits_symbols_space.txt",
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True,
)

model = dict(
    type="SARNet",
    data_preprocessor=dict(
        type="TextRecogDataPreprocessor", mean=[127, 127, 127], std=[127, 127, 127]
    ),
    backbone=dict(type="ResNet31OCR"),
    encoder=dict(
        type="SAREncoder",
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type="ParallelSARDecoder",
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True,
        postprocessor=dict(type="AttentionPostprocessor"),
        module_loss=dict(type="CEModuleLoss", ignore_first_char=True, reduction="mean"),
        dictionary=dictionary,
        max_seq_len=30,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", ignore_empty=True, min_size=2),
    dict(type="LoadOCRAnnotations", with_text=True),
    dict(
        type="RescaleToHeight", height=48, min_width=48, max_width=160, width_divisor=4
    ),
    dict(type="PadToWidth", width=160),
    dict(
        type="PackTextRecogInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "valid_ratio"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RescaleToHeight", height=48, min_width=48, max_width=160, width_divisor=4
    ),
    dict(type="PadToWidth", width=160),
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
            [
                dict(
                    type="RescaleToHeight",
                    height=48,
                    min_width=48,
                    max_width=160,
                    width_divisor=4,
                )
            ],
            [dict(type="PadToWidth", width=160)],
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

auto_scale_lr = dict(base_batch_size=64 * 48)
