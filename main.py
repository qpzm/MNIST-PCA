from plotter import plot
from logistic_pca_evaluate import calc

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
                    calc(data, optimizer, norm, lr, epochs, iteration)
                    dir = f'{self.data}-weights-{self.optimizer}-{self.lr}-{self.norm}'
                    plot(dir)

if __name__ == '__main__':
    main()
