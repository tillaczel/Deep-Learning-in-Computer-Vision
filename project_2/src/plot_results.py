import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch

def plot_predictions(input_data, predictions): 
    plt.rcParams['figure.figsize'] = [18, 6]
    for i in range(6):
        plt.subplot(2, 6, i+1)
        plt.imshow(torch.squeeze(input_data[i],0))
        plt.subplot(2, 6, i+7)
        plt.imshow(torch.squeeze(predictions[i],0))
    plt.show();
