import gdown
import zipfile
import os
import cv2
import json

gdown.download(id="1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml", quiet=True)
zipfile.ZipFile("vietnamese_original.zip", "r").extractall()
os.remove("vietnamese_original.zip")

train_imagesDirPath = "vietnamese/train_images/"
test_imageDirPath = "vietnamese/test_image/"
unseen_test_imagesDirPath = "vietnamese/unseen_test_images/"
oldLabelsDirPath = "vietnamese/labels/"

datasetDirPath = "data/vintext/"
os.makedirs(datasetDirPath + "textrecog_imgs/train")
os.makedirs(datasetDirPath + "textrecog_imgs/val")
os.makedirs(datasetDirPath + "textrecog_imgs/test")

trainData = {
    "metainfo": {"dataset_type": "TextRecogDataset", "task_name": "textrecog"},
    "data_list": [],
}
valData = {
    "metainfo": {"dataset_type": "TextRecogDataset", "task_name": "textrecog"},
    "data_list": [],
}
testData = {
    "metainfo": {"dataset_type": "TextRecogDataset", "task_name": "textrecog"},
    "data_list": [],
}

for i in range(1, 2001):

    if i <= 1200:
        oldImageFilePath = train_imagesDirPath
        prefix = "train/"
        data = trainData
    elif i <= 1500:
        oldImageFilePath = test_imageDirPath
        prefix = "val/"
        data = valData
    else:
        oldImageFilePath = unseen_test_imagesDirPath
        prefix = "test/"
        data = testData

    oldImageFilePath += f"im{i:04d}.jpg"
    oldImage = cv2.imread(oldImageFilePath)
    oldImageHeight, oldImageWidth = oldImage.shape[:2]

    oldLabelFilePath = oldLabelsDirPath + f"gt_{i}.txt"
    oldLabelFile = open(oldLabelFilePath, "r", encoding="utf-8")
    id = 1
    for line in oldLabelFile.readlines():
        line = line.split(",")
        x_min = min(int(line[0]), int(line[2]), int(line[4]), int(line[6]))
        y_min = min(int(line[1]), int(line[3]), int(line[5]), int(line[7]))
        x_max = max(int(line[0]), int(line[2]), int(line[4]), int(line[6]))
        y_max = max(int(line[1]), int(line[3]), int(line[5]), int(line[7]))
        text = ",".join(line[8:]).removesuffix("\n")
        if (
            0 <= x_min <= oldImageWidth
            and 0 <= y_min <= oldImageHeight
            and 0 <= x_max <= oldImageWidth
            and 0 <= y_max <= oldImageHeight
            and text != "###"
        ):
            if x_max - x_min < 5 or y_max - y_min < 5:
                continue
            croppedImage = oldImage[y_min:y_max, x_min:x_max]
            croppedImageFilePath = "textrecog_imgs/" + prefix + f"im{i:04d}_{id}.jpg"
            cv2.imwrite(datasetDirPath + croppedImageFilePath, croppedImage)
            data["data_list"].append(
                {
                    "instances": [{"text": text}],
                    "img_path": croppedImageFilePath,
                }
            )
            id += 1

    oldLabelFile.close()
    os.remove(oldImageFilePath)
    os.remove(oldLabelFilePath)

json.dump(
    trainData,
    open(datasetDirPath + "textrecog_train.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    valData,
    open(datasetDirPath + "textrecog_val.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    testData,
    open(datasetDirPath + "textrecog_test.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=4,
)

os.rmdir(train_imagesDirPath)
os.rmdir(test_imageDirPath)
os.rmdir(unseen_test_imagesDirPath)
os.rmdir(oldLabelsDirPath)
os.remove("vietnamese/general_dict.txt")
os.remove("vietnamese/vn_dictionary.txt")
os.rmdir("vietnamese")
