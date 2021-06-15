from project_2.src.metrics.iou import calculate_iou
import numpy as np

def calculate_energy(preds, segs):
    E0 = []
    E1 = []
    E2 = []
    n_experts = segs.shape[1]
    n_samples = preds.shape[1]

    for n in range(len(preds)):
        x = preds[n]
        y = segs[n]

        i = np.random.randint(0, n_experts)
        j = np.random.randint(0, n_samples)
        d0 = 1 - calculate_iou(x[j], y[i][0], spatial_dim=(0, 1))
        E0.append(d0.item())

        i = np.random.randint(0, n_experts)
        j = np.random.randint(0, n_experts)
        d1 = 1 - calculate_iou(y[j][0], y[i][0], spatial_dim=(0, 1))
        E1.append(d1.item())

        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, n_samples)
        d2 = 1 - calculate_iou(x[i], x[j], spatial_dim=(0, 1))
        E2.append(d2.item())
    E0 = np.mean(E0)
    E1 = np.mean(E1)
    E2 = np.mean(E2)

    D = 2 * E0 - E1 - E2
    print('distance expectations', E0, E1, E2)
    print('GED:', D)

    return D