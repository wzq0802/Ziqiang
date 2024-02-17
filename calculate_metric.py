
import os
import numpy as np
from PIL import Image

def calculate_dice_coefficient(pred_img, true_img):
    intersection = np.logical_and(pred_img, true_img).sum()
    dice = (2.0 * intersection) / (pred_img.sum() + true_img.sum())
    return dice

def calculate_iou(pred_img, true_img):
    intersection = np.logical_and(pred_img, true_img)
    union = np.logical_or(pred_img, true_img)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_precision(pred_img, true_img):
    true_positives = np.logical_and(pred_img, true_img).sum()
    false_positives = np.logical_and(pred_img, np.logical_not(true_img)).sum()
    precision = true_positives / (true_positives + false_positives + 1e-7)
    return precision
def calculate_recall(pred_img, true_img):
    true_positives = np.logical_and(pred_img, true_img).sum()
    false_negatives = np.logical_and(np.logical_not(pred_img), true_img).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    return recall


pred_folder = 'DU/pred'
true_folder = 'true'


pred_files = os.listdir(pred_folder)
true_files = os.listdir(true_folder)

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []


for i in range(len(pred_files)):
    pred_path = os.path.join(pred_folder, pred_files[i])
    true_path = os.path.join(true_folder, true_files[i])

    pred_img = np.array(Image.open(pred_path).convert('L')) > 0
    true_img = np.array(Image.open(true_path).convert('L')) > 0

    pred_img = np.logical_not(pred_img)
    true_img = np.logical_not(true_img)


    if pred_img.shape != true_img.shape:
        raise ValueError(f"图像 {pred_files[i]} 和 {true_files[i]} 的尺寸不匹配！")


    dice_scores.append(calculate_dice_coefficient(pred_img, true_img))
    iou_scores.append(calculate_iou(pred_img, true_img))
    precision_scores.append(calculate_precision(pred_img, true_img))
    recall_scores.append(calculate_recall(pred_img, true_img))

average_dice = np.mean(dice_scores)
average_iou = np.mean(iou_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)

print(f"平均Dice系数: {average_dice}")
print(f"平均IoU: {average_iou}")
print(f"平均精确率: {average_precision}")
print(f"平均Recall: {average_recall}")


