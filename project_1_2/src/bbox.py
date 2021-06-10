import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import torch
import wandb
from torch import nn
from tqdm import tqdm

from project_1_2.src.data import get_data_raw


def plt_bboxes(img, bboxes, labels, filename=None):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(img)
    for bbox, label in zip(bboxes, labels):
        # TODO: add colors
        l, t, w, h = bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2]
        plt.text(l,t, str(label), fontsize=30)
        rect = patches.Rectangle((l, t), w, h, linewidth=1, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
    if filename is not None:
        plt.savefig(filename)


def bbox_intersection_over_union(boxA, boxB):
    max_l = max(boxA[0], boxB[0])
    min_r = min(boxA[1], boxB[1])
    max_t = max(boxA[2], boxB[2])
    min_b = min(boxA[3], boxB[3])

    interArea = max(0, min_r - max_l + 1) * max(0, min_b - max_t + 1)
    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def non_maximum_suppression(bboxes, p_classes, p_threshold=0.6, iou_threshold=0.5):
    class_idx, probs = np.argmax(p_classes, axis=1), np.max(p_classes, axis=1)
    idx = np.argwhere(p_threshold < probs).flatten()
    bboxes, probs, class_idx = bboxes[idx], probs[idx], class_idx[idx]
    final_bboxes, final_probs, final_class_idx = list(), list(), list()
    while probs.shape[0] > 0:
        max_idx = np.argmax(probs)
        final_bboxes.append(bboxes[max_idx]), np.delete(bboxes, max_idx)
        final_probs.append(probs[max_idx]), np.delete(probs, max_idx)
        final_class_idx.append(class_idx[max_idx]), np.delete(class_idx, max_idx)
        ious = list()
        for bbox in bboxes:
            ious.append(bbox_intersection_over_union(final_bboxes[-1], bbox))
        idx = np.argwhere(np.array(ious) < iou_threshold).flatten()
        bboxes, probs, class_idx = bboxes[idx], probs[idx], class_idx[idx]
    return final_bboxes, final_probs, final_class_idx


def filter_bboxes(result, img, filename=None, p_threshold=0.1, iou_threshold=0.5):
    bboxes, p_classes = map(np.array, zip(*result))
    bboxes, probs, class_idx = non_maximum_suppression(bboxes, p_classes, p_threshold=p_threshold, iou_threshold=iou_threshold)
    plt_bboxes(img, bboxes, class_idx, filename=filename)
    return bboxes, probs, class_idx

def coord_to_boxes(coord, ratio):  # TODO: this is probably wrong...
    center = (coord * 32 + 112)
    coord = ( # who knows where is up or left
        int((center[0] - 112) / ratio),
        int((center[0] + 112) / ratio),
        int((center[1] - 112) / ratio),
        int((center[1] + 112) / ratio),
    )
    return coord

def run_detection(engine):
    engine.model.resnet[-1] = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
    n_images_to_process = 10

    thresholds = [(0.1, 0.5), (0.2, 0.5), (0.1, 0.3), (0.25, 0.25)]

    resize_ratios = (0.55, 0.7, 0.8) # todo: config
    # dataset = get_data_raw(split='test', resize_ratios=resize_ratios)
    dataset = get_data_raw(split='train', resize_ratios=resize_ratios)
    for i in tqdm(range(n_images_to_process)):
        im_dir = os.path.join(wandb.run.dir, f'img_{i}')
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        result2 = []
        resized, original_image, ratios, meta = dataset[i]
        # this is ugly but should work
        for img, ratio in zip(resized, ratios):
            result = []

            y = engine(img.unsqueeze(dim=0))
            probs = torch.softmax(y, 1).detach().cpu().numpy()
            for i in range(probs.shape[-1]):  # x-axis
                for j in range(probs.shape[-2]):  # y-axis
                    p = probs[0, :, j, i]
                    coord = coord_to_boxes(np.array([i, j]), ratio)

                    result.append((
                        coord,
                        p[:10]
                    ))
                    result2.append((
                        coord,
                        p[:10]
                    ))
            for th in thresholds:
                th_dir = os.path.join(im_dir, f'thresh_{th[0]:.2f}_{th[1]:.2f}')
                if not os.path.isdir(th_dir):
                    os.mkdir(th_dir)
                p_threshold, iou_threshold = th
                filename = os.path.join(th_dir, f'bbox_{ratio:.2f}.png')
                filter_bboxes(result, original_image, filename=filename, p_threshold=p_threshold, iou_threshold=iou_threshold)
        for  th in thresholds:
            th_dir = os.path.join(im_dir, f'thresh_{th[0]:.2f}_{th[1]:.2f}')
            if not os.path.isdir(th_dir):
                os.mkdir(th_dir)
            p_threshold, iou_threshold = th
            filename = os.path.join(th_dir, f'bbox_.png')
            filter_bboxes(result2, original_image, filename=filename, p_threshold=p_threshold, iou_threshold=iou_threshold)

