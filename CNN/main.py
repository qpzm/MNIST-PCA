from threading import Thread
import sys
sys.path.append('../')
from plotter import plot
from cnn import calc
import os
from os.path import dirname


epochs = 50
norm='maxmin'
configs = {
    'data': ['mnist', 'fashion_mnist'],
    'optimizer': ['SGD', 'Adam'],
    'lr': [1e-2, 1e-3],
}

def recursive_mkdir(path):
    parent = dirname(path)
    if not os.path.exists(parent):
        recursive_mkdir(parent)
    if not os.path.exists(path):
        os.mkdir(path)


def calc_and_plot(data, optimizer, norm, lr, epochs, iteration):
    calc(data, optimizer, norm, lr, epochs, iteration)
    data_dir = f'{data}-weights-{optimizer}-{lr}-{norm}'
    plot_parent_dir = './plots'
    for i in range(0, 4):
        data_path = f'{data_dir}/store-{i}.pkl'
        plot_path = f'{plot_parent_dir}/{data_dir}/layer{i}'
        recursive_mkdir(plot_path)
        plot(plot_path, data_path)

def main():
    iteration = 0
    threads = []
    for data in configs['data']:
        for optimizer in configs['optimizer']:
                for lr in configs['lr']:
                    iteration += 1
                    # t = Thread(target=calc_and_plot, args=(data, optimizer, norm, lr, epochs, iteration))
                    # t.start()
                    calc_and_plot(data, optimizer, norm, lr, epochs, iteration)
                    #threads.append(t)

if __name__ == '__main__':
    main()
