from plotter import plot
from logistic_pca_evaluate import calc
from utils import recursive_mkdir
import os

n_classes = 10
input_dim = 784  # 28*28
output_dim = n_classes
epochs = 100
configs = {
    'data': ['mnist', 'fashion_mnist'],
    'optimizer': ['SGD', 'Adam'],
    'norm': ['maxmin', 'no-norm'],
    'lr': [1e-2, 1e-3, 1e-4],
}

def main():
    iteration = 0
    for data in configs['data']:
        for norm in configs['norm']:
            for optimizer in configs['optimizer']:
                for lr in configs['lr']:
                    iteration += 1
                    dir = f'{data}-weights-{optimizer}-{lr}-{norm}'
                    data_dir = f'{dir}/store.pkl'
                    plot_dir = f'./plots/{dir}'
                    calc(data, optimizer, norm, lr, epochs, iteration)
                    recursive_mkdir(plot_dir)
                    plot(plot_dir, data_dir)

if __name__ == '__main__':
    main()
