vintext_textrecog_data_root = "data/vintext"

vintext_textrecog_train = dict(
    type="OCRDataset",
    data_root=vintext_textrecog_data_root,
    ann_file="textrecog_train.json",
    pipeline=None,
)

vintext_textrecog_val = dict(
    type="OCRDataset",
    data_root=vintext_textrecog_data_root,
    ann_file="textrecog_val.json",
    test_mode=True,
    pipeline=None,
)

vintext_textrecog_test = dict(
    type="OCRDataset",
    data_root=vintext_textrecog_data_root,
    ann_file="textrecog_test.json",
    test_mode=True,
    pipeline=None,
)
