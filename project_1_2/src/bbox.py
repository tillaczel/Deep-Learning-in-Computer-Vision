import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plt_bboxes(img, bboxes, filename=None):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(img)
    for bbox in bboxes:
        l, t, w, h = bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2]
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


def filter_bboxes(result, img, filename=None):
    bboxes, p_classes = map(np.array, zip(*result))
    bboxes, probs, class_idx = non_maximum_suppression(bboxes, p_classes, p_threshold=0.1)
    plt_bboxes(img, bboxes, filename=filename)
    return bboxes, probs, class_idx
