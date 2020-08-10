import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import sys

args = sys.argv
data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
model_name = args[2]    # 'resnet34', 'resnet18', 'vgg16', 'vgg13', 'vgg11'

runs = 'runs'
batch_per_epoch = 391 if 'cifar' in data_name else 573
file_path = os.path.join(runs, data_name, model_name, 'log', 'train energy', 'train_energy.csv')

test_path = os.path.join(runs, data_name, model_name, 'log', 'test_acc.csv')
test_results = pd.read_csv(test_path)
test_steps = test_results['steps']
test_epoch = (test_steps-1)/batch_per_epoch
test_epoch = test_epoch.astype(np.int)
test_error = 1 - test_results['values']

noise_path = os.path.join(runs, data_name, model_name, 'log', 'noise_acc.csv')
noise_results = pd.read_csv(noise_path)
noise_steps = noise_results['steps']
noise_epoch = (noise_steps-1)/batch_per_epoch
noise_epoch = noise_epoch.astype(np.int)
noise_error = 1 - noise_results['values']


data = pd.read_csv(file_path).values

itr_index = data[0:, 0].astype(np.int)
fft_epoch = (itr_index-1)/batch_per_epoch
fft_epoch = fft_epoch.astype(np.int)

energy = data[0:, 1:].transpose(1, 0)
delta_energy = np.log(energy / energy[0:1, :]+1e-10)

fig = plt.figure(figsize=(5, 3))

ax = fig.add_subplot(111)

cmap = sb.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True)
heatmap = ax.contourf(fft_epoch[:], range(delta_energy.shape[0]), delta_energy, 30, cmap=cmap)
ax.set_ylabel('frequency', fontsize=14)
ax.set_xlabel('epoch', fontsize=14)
ax.set_xlim([min(fft_epoch), max(fft_epoch)])

ax2 = ax.twinx()

ax2.plot(test_epoch, test_error, color='darkorange', label='test')
ax2.plot(noise_epoch, noise_error, color='red', label='perturbed')
ax2.set_ylabel('error', fontsize=14)

fig.colorbar(heatmap, pad=0.15)

plt.grid(False)
plt.xscale('log')

plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()
