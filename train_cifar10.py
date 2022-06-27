from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from vit import ViT

# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--dimhead', default="512", type=int)


args = parser.parse_args()


bs = int(args.bs)
imsize = int(args.size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

size = imsize
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ViT(image_size = size,
          patch_size = args.patch,
          num_classes = 10,
          dim = int(args.dimhead),
          depth = 6,
          heads = 8,
          mlp_dim = 512,
          dropout = 0.1,
          emb_dropout = 0.1
)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)

    
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

scaler = torch.cuda.amp.GradScaler(enabled=True)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step(test_loss)
    return test_loss, acc

list_loss = []
list_acc = []

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    test_loss, acc = test(epoch)
    
    if args.cos:
        scheduler.step(epoch-1)
    
    list_loss.append(test_loss)
    list_acc.append(acc)

    print(list_loss)