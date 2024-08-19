# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt
from typing import Iterable
from typing import Optional
import time
import os
from datetime import datetime
import json


# 0:unlabeled, 1:fire_extinhuisher, 2:backpack, 3:hand_drill, 4:rescue_randy
def get_palette_PV1():
    unlabelled          = [0,0,0]
    penguin             = [0,0,255]
    palette    = np.array([unlabelled, penguin]).astype(np.uint8)
    return palette


def visualize_pred(palette, pred_):
    mapped_label = palette[pred_]
    return mapped_label


def make_save_dir(path_root, pred_name):
    dir_root = os.path.join(path_root)

    if not os.path.exists(dir_root):
        os.mkdir(dir_root)

    dir_pred = os.path.join(path_root, pred_name)
    if not os.path.exists(dir_pred):
        os.mkdir(dir_pred)

    dir_rgb = os.path.join(path_root, 'rgb')
    if not os.path.exists(dir_rgb):
        os.mkdir(dir_rgb)

    dir_thr = os.path.join(path_root, 'thr')
    if not os.path.exists(dir_thr):
        os.mkdir(dir_thr)

    dir_gt  = os.path.join(path_root, 'gt')
    if not os.path.exists(dir_gt):
        os.mkdir(dir_gt)


def aggregate_by_node(node_name, json_objects):
    return [o[node_name] for o in json_objects]


def load_logs(output_dir):
    logs = [o for o in os.listdir(output_dir) if o.startswith("log_")]
    logs.sort(key=lambda f: datetime.strptime(f, 'log_%Y%m%d_%H%M%S.txt'), reverse=True)
    json_objects = []
    with open(os.path.join(output_dir, logs[0]), 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())
                json_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line.strip()}")
    return json_objects


def confusion_matrix(c):
    """
    render confusion matrix
    :param c: 2 x 2 Numpy array with the values of the confusion matrix [TP, TN],[FP, FN]
    :return: None
    """
    fig, axs = plt.subplots(1,1)
    c = np.array(c)
   
    c = c / c.sum()
    c = np.round(c, 3)
    axs.matshow(c, cmap="Wistia")
    axs.grid(False)
    plt.xticks([])
    plt.yticks([])
    axs.text(x=0, y=0,s=c[0][0], va='center', ha='center', size='xx-large')
    axs.text(x=1, y=0,s=c[0][1], va='center', ha='center', size='xx-large')
    axs.text(x=0, y=1,s=c[1][0], va='center', ha='center', size='xx-large')
    axs.text(x=1, y=1,s=c[1][1], va='center', ha='center', size='xx-large')
    plt.xlabel('Positive     Negative', fontsize=18)
    plt.ylabel('False          True', fontsize=18)
    plt.show()


def training_perf(loss, precision, recall, iou, ca):
    epochs: int = len(loss)
    passes = np.linspace(1,epochs,epochs)
    fig, axs = plt.subplots(2,2, figsize=(16, 10))
    xticks = range(0, epochs+1, 10)
    plt.setp(axs, xticks=xticks, xticklabels=[f"{int(p):d}" for p in xticks])
    axs[0][0].plot(passes, loss, label="Loss")
    axs[0][1].plot(passes, ca, label="Counting Accuracy")
    axs[1][0].plot(passes, precision, label="Precision")
    axs[1][0].plot(passes, recall, label="Recall")
    axs[1][1].plot(passes, iou, label="IoU")
    axs[0][0].set_ylim(0.0, 1.0)
    axs[0][1].set_ylim(-50.0, 50.0)
    axs[1][0].set_ylim(0.0, 1.0)
    axs[1][1].set_ylim(0.0, 1.0)
    axs[0][0].legend()
    axs[0][1].legend()
    axs[1][0].legend()
    axs[1][1].legend()
    axs[0][0].set_xlabel("Training Epochs")
    axs[0][1].set_xlabel("Training Epochs")
    axs[1][0].set_xlabel("Training Epochs")
    axs[1][1].set_xlabel("Training Epochs")
    plt.show()


def evaluation_perf(loss, precision, recall, iou, ca):
    fig, axs = plt.subplots(1,1, figsize=(14, 8))
    axs.bar(height = loss, x="Loss")
    plt.text(x="Loss", y=loss/2.0, s=loss, ha='center', color="white")
    axs.bar(height = precision, x="Precision")
    plt.text(x="Precision", y=precision/2.0, s=precision, ha='center', color="white")
    axs.bar(height = recall, x="Recall")
    plt.text(x="Recall", y=recall/2.0, s=recall, ha='center', color="white")
    axs.bar(height = ca, x="Counting Accuracy")
    plt.text(x="Counting Accuracy", y=ca/2.0, s=loss, ha='center', color="white")
    axs.bar(height = iou, x="IoU")
    plt.text(x="IoU", y=iou/2.0, s=iou, ha='center', color="white")
    plt.show()
        