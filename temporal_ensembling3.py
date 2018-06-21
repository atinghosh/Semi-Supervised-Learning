import random
from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sys import path
path.insert(0,'/home/atin/New_deployed_projects/SSL')

from SL1 import Net, test
import math

from torchvision.transforms import ToTensor, Compose, Lambda, ToPILImage, Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from timeit import default_timer as timer
from torch.autograd import Variable
to_img = ToPILImage()
CUDA = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

mean = (0.4376821 , 0.4437697 , 0.47280442)
std = (0.19803012, 0.20101562, 0.19703614)

k = 1000 #no of labelled sample



def addGaussian(I, ismulti=True):
    """Add Gaussian with noise
    input is numpy array H*W*C
    """
    ax = np.asarray(I)
    ax = ax.copy()
    shape = (32, 32)  # ax.shape[:2]
    intensity_noise = np.random.uniform(low=0, high=0.05)
    if ismulti:
        ax[:, :, 0] = ax[:, :, 0] * (
                    1 + intensity_noise * np.random.normal(loc=0, scale=1, size=shape[0] * shape[1]).reshape(shape[0],
                                                                                                             shape[1]))
    else:
        ax[:, :, 0] = ax[:, :, 0] + intensity_noise * np.random.normal(loc=0, scale=1,
                                                                       size=shape[0] * shape[1]).reshape(shape[0],
                                                                                                         shape[1])

    return ax


def create_dataloader(train_dataset, test_dataset, batch_size, k, seed):

    '''Unlabelled images get label = -1'''
    n_sample = len(train_dataset)
    random.seed(seed)
    labelled_data_index = np.random.choice(range(n_sample), k, replace= False)
    un_labelled_data_index = list(set(range(n_sample)) - set(labelled_data_index))
    train_dataset.labels[un_labelled_data_index] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=False)

    return train_loader, test_loader



def temporal_loss(out1, out2, w, labels):
    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()

    def masked_crossentropy(out, labels):
        nbsup = len(torch.nonzero(labels >= 0))
        loss = F.cross_entropy(out, labels, size_average=False, ignore_index=-1)
        if nbsup != 0:
            loss = loss / nbsup
        return loss, nbsup

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


def ramp_up(epoch, max_epochs, max_val=30., mult=-5):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, n_labeled, n_samples, max_val=30, mult=-5):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)

def temporal_train(train_loader, test_dataloader, model, optimizer, num_epoch,n_sample, n_class =10, alpha = .6, max_epoch = 80):

    n_labeled = len(train_loader.dataset)
    temp_z = torch.zeros(n_sample, n_class).float().cuda()
    Z = torch.zeros(n_sample, n_class).float().cuda()
    output = torch.zeros(n_sample, n_class).float().cuda()

    for epoch in range(num_epoch):
        w = weight_schedule(epoch,max_epoch,n_labeled,n_sample)
        w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

        start = 0
        for i, (images, label) in enumerate(train_loader):
            images, label = images.cuda(), label.cuda()
            out = model(images)
            end = start + images.size(0)
            output[start:end] = out.data.clone()
            final_loss, sup_loss, unsup_loss, nbsup = temporal_loss(out, temp_z[start:end], w, label)
            start = end



            final_loss.backward()
            optimizer.step()

        Z = alpha * Z + (1. - alpha) * output
        temp_z = Z * (1. / (1. - alpha ** (epoch + 1)))

        test(model, test_dataloader)


if __name__ == "__main__":
    svhn_dataset_train = SVHN(root='/data02/Atin/DeployedProjects/SVHN', split='train',
                              transform=Compose([Lambda(addGaussian), ToTensor(), Normalize(mean, std)]))
    svhn_dataset_test = SVHN(root='/data02/Atin/DeployedProjects/SVHN', split='test', download=True,
                             transform=Compose([ToTensor(), Normalize(mean, std)]))

    train_loader, test_loader = create_dataloader(svhn_dataset_train,svhn_dataset_test, 256,1000, 13)
    model = Net()
    if CUDA: model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=.002, weight_decay=.00001, betas=(0.9, 0.99))

    temporal_train(train_loader, test_loader, model, optimizer, num_epoch=150, n_sample=len(svhn_dataset_train))













