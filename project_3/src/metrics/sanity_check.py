import torch

from project_2.src.metrics import calc_all_metrics


def sanity_check_metrics():
    print('Sanity checks')
    a = torch.zeros((1, 1, 100, 100), dtype=torch.int32)
    b = torch.zeros((1, 1, 100, 100), dtype=torch.int32)
    a[:, :, :, :50] = 1
    b[:, :, :50, :] = 1
    print('half half', calc_all_metrics(a, b))

    a[:, :, :, :] = 0
    b[:, :, :, :] = 0
    a[:, :, :50, :50] = 1
    b[:, :, :50, :] = 1
    print('quarter in half', calc_all_metrics(a, b))

    a[:, :, :, :] = 0
    b[:, :, :, :] = 0
    a[:, :, :50, :] = 1
    b[:, :, :50, :50] = 1
    print('half around quarter', calc_all_metrics(a, b))

    a[:, :, :, :] = 0
    b[:, :, :, :] = 0
    a[:, :, :50, :50] = 1
    print('quarter in black', calc_all_metrics(a, b))

    a[:, :, :, :] = 0
    b[:, :, :, :] = 1
    a[:, :, :50, :50] = 1
    print('quarter in white', calc_all_metrics(a, b))

    a[:, :, :, :] = 0
    b[:, :, :, :] = 0
    b[:, :, :50, :50] = 1
    print('black over quarter', calc_all_metrics(a, b))

    a[:, :, :, :] = 1
    b[:, :, :, :] = 0
    b[:, :, :50, :50] = 1
    print('white over quarter', calc_all_metrics(a, b))
