import csv
import os
import torch.utils.data.sampler as torch_sampler
import numpy as np

def LOG2CSV(data, csv_file, flag = 'a'):
    '''
    data: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()


def print_model_arch(model):

    print('\n-----------\n')
    print('MODEL ARCH\n')
    print('-----------\n')
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')

def random_train_valid_samplers(data_set, split_ratio = 0.2, seed = 619):

    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(split_ratio * num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch_sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch_sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler