from torch.nn.utils import weight_norm
from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Lambda, ToPILImage, Normalize
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import os
to_img = ToPILImage()
CUDA = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# mean = (0.2632, 0.2522, 0.2302)
# std = (0.1123, 0.1081, .006)
mean = (0.4376821 , 0.4437697 , 0.47280442)
std = (0.19803012, 0.20101562, 0.19703614)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device = torch.device("cuda:0")


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

def get_padding(padding_type, kernel_size):
    assert padding_type in ['SAME', 'VALID']
    if padding_type == 'SAME':
        return tuple((k - 1) // 2 for k in kernel_size)
    return tuple(0 for _ in kernel_size)


class Conv_block(nn.Module):
    def __init__(self, nb_filter,in_channel):
        '''nb_filter must be an array of length as nb_conv
        '''
        super(Conv_block, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channels=in_channel, out_channels=nb_filter, kernel_size= (3,3), padding=(1,1)))
        self.conv2 = weight_norm(nn.Conv2d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=(3,3), padding=(1,1)))
        self.conv3 = weight_norm(nn.Conv2d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=(3,3), padding=(1,1)))
        self.conv_drop = nn.Dropout(p=.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=.1)
        x = F.leaky_relu(self.conv3(x), negative_slope=.1)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv_drop(x)
        return x

class Net(nn.Module):
    def __init__(self):
        '''nb_filter must be an array of length as nb_conv
        '''
        super(Net, self).__init__()
        self.conv_block1 = Conv_block(128,3)
        self.conv_block2 = Conv_block(256, 128)
        self.conv1 = weight_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)))
        self.conv2 = weight_norm(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)))
        self.conv3 = weight_norm(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1)))
        self.conv_drop = nn.Dropout(p=.5)
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=.1)
        x = F.leaky_relu(self.conv3(x), negative_slope=.1)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(-1, 128)
        x = self.fc(x)

        return x


def train(epoch, model, dataloader, optimizer):
    model.train()

    train_loss = 0
    count = 0
    for batch_id, (data, target) in enumerate(dataloader):
        count += data.size(0)
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(input=output, target=target)
        train_loss = train_loss + loss.data[0] * data.size(0)

        loss.backward()
        optimizer.step()

        if batch_id % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_id * len(data), len(dataloader.dataset), 100. * batch_id / len(dataloader), loss.data[0]))

    train_loss /= count
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))


def test(model, testloader):
    model.eval()

    test_loss = 0
    correct = 0
    for _, (data, target) in enumerate(testloader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


if __name__ == "__main__":

    svhn_dataset_train = SVHN(root='/data02/Atin/DeployedProjects/SVHN', split='train',
                              transform=Compose([Lambda(addGaussian),ToTensor(),Normalize(mean, std)]))
    svhn_dataset_test = SVHN(root='/data02/Atin/DeployedProjects/SVHN', split='test', download=True,
                             transform=Compose([ToTensor(),Normalize(mean, std)]))


    train_dataloader = DataLoader(svhn_dataset_train, batch_size=64, num_workers=10, shuffle=True)
    test_dataloader = DataLoader(svhn_dataset_test, batch_size=64, num_workers=10, shuffle=True)

    model = Net()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay= .00001)
    # optimizer = optim.SGD(model.parameters(), lr =.003, momentum=.9)

    nb_epoch = 300
    for epoch in range(1, nb_epoch + 1):
        train(epoch, model, train_dataloader, optimizer)
        test(model, test_dataloader)


