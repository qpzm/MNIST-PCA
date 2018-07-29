import matplotlib
matplotlib.use('Agg')


import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os


class Plotter(object):
    def __init__(self, plot_dir, data_dir):
        self.plot_dir = plot_dir
        self.data_dir = data_dir

    def plot_3d(self, data, i):
        mpl.rcParams['legend.fontsize'] = 20

        fig = plt.figure(figsize=(12,9))
        ax = fig.gca(projection='3d')
        ax.plot(data[:,2*i], data[:,2*i+1], -np.log10(data[:,-1]), '-*', label='negative log train loss')
        plt.xlabel('PC' + str(2*i+1))
        plt.ylabel('PC' + str(2*i+2))
        plt.tight_layout()
        ax.legend()
        fig.savefig(f'{self.plot_dir}/3d-{i}.png', bbox_inches='tight')


    def plot_2d(self, data, i):
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure(figsize=(6,6))
        ax = fig.gca()
        ax.plot(data[:,2*i], data[:,2*i+1], '-*', label='train trajectory')
        plt.xlabel('PC' + str(2 * i + 1))
        plt.ylabel('PC' + str(2 * i + 2))
        ax.legend()
        fig.savefig(f'{self.plot_dir}/2d-{i}.png', bbox_inches='tight')


def plot(plot_dir, data_dir):
    plotter = Plotter(plot_dir, data_dir)

    with open(data_dir, 'rb') as f:
        coordinates, history, init_loss = pickle.load(f)

    train_losses = history['loss']
    train_losses.insert(0, init_loss[0])
    np_train_losses = np.array(train_losses).reshape((-1,1))
    data = np.concatenate((coordinates, np_train_losses), axis=1)

    # plot pc1 & pc2, pc3 & pc4 ... pc9 & pc10
    if not os.path.exists(plotter.plot_dir):
        os.mkdir(plotter.plot_dir)

    for i in range(0, 5):
        plotter.plot_3d(data, i)
        plotter.plot_2d(data, i)


if __name__ == '__main__':
    main()
