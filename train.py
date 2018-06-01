from __future__ import print_function, division

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import datasets, models, transforms

import numpy as np
from scipy import linalg
import os
import time
import config


# OLE loss:
class OLELoss(Function):
    def __init__(self, lambda_=config.lambda_):
        super(OLELoss, self).__init__()
        self.lambda_ = lambda_
        self.dX = 0

    def forward(self, x, y):
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        classes = np.unique(y)

        n, d = x.shape

        lambda_ = 1.
        delta = 1.

        # gradients initialization
        obj_c = 0
        dx_c = np.zeros((n, d))

        eigthd = 1e-6  # threshold small eigenvalues for a better subgradient

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            a = x[y == c, :]

            # SVD
            u, s, v = linalg.svd(a, full_matrices=False)

            v = v.T
            nuclear = np.sum(s)

            # L_c = max(DELTA, ||TY_c||_*)-DELTA

            if nuclear > delta:
                obj_c += nuclear

                # discard small singular values
                r = np.sum(s < eigthd)
                uprod = u[:, 0:u.shape[1] - r].dot(v[:, 0:v.shape[1] - r].T)

                dx_c[y == c, :] += uprod
            else:
                obj_c += delta

        # compute objective and gradient for secon term ||TX||*

        u, s, v = linalg.svd(x, full_matrices=False)  # all classes

        v = v.T

        obj_all = np.sum(s)

        r = np.sum(s < eigthd)

        uprod = u[:, 0:u.shape[1] - r].dot(v[:, 0:v.shape[1] - r].T)

        dx_all = uprod

        obj = (obj_c - lambda_ * obj_all) / n * np.float(self.lambda_)

        dx = (dx_c - lambda_ * dx_all) / n * np.float(self.lambda_)

        self.dX = torch.FloatTensor(dx).cuda()
        return torch.FloatTensor([float(obj)]).cuda()

    def backward(self, grad_output):
        dx = self.dX
        return dx, None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = config.data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if config.use_OLE:
        print('using CrossEntropyloss with OLEloss')
        criterion = [nn.CrossEntropyLoss()] + [OLELoss()]
    else:
        print('using CrossEntropyloss')
        criterion = [nn.CrossEntropyLoss()]

    if config.use_pretrained:
        model = models.resnet152(pretrained=True)
    else:
        model = models.resnet152(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.class_num)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    torch.backends.cudnn.benchmark = True
    model = model.to(device)
    start_epoch = 0
    best_acc = 0
    lr = config.initial_lr

    if config.is_resume:
        print('==> Resuming from checkpoint..')
        weight_dir = os.path.join(config.checkpoint, 'final.pth.tar')
        checkpoint = torch.load(weight_dir)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr = pow(config.decay_scalar, int(start_epoch // config.decay_step)) * lr

    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.decay_step, gamma=config.decay_scalar)

    best_acc = train_model(dataloaders, dataset_sizes, start_epoch, best_acc, model, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=config.max_epoch)
    filename = os.path.join(config.checkpoint, 'final.pth.tar')
    torch.save({
        'epoch': config.max_epoch + start_epoch,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
    }, filename)
    print('Best val Acc: {:4f}'.format(best_acc), 'save at %s' % filename)


def train_model(dataloaders, dataset_sizes, start_epoch, best_acc, model, criterion, optimizer, scheduler,
                num_epochs=25):

    best_acc = best_acc

    for epoch in range(start_epoch, start_epoch + num_epochs):
        since = time.time()

        print('Epoch {}/{}'.format(epoch + 1, start_epoch + num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # criterion is a list composed of crossentropy loss and OLE loss.
                    losses_list = [-1, -1]
                    # output_Var contains scores in the first element and features in the second element
                    loss = 0
                    for cix, crit in enumerate(criterion):
                        if cix == 1:
                            losses_list[cix] = crit(outputs, labels)[0]
                        else:
                            losses_list[cix] = crit(outputs, labels)
                        loss += losses_list[cix]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                time_elapsed = time.time() - since
                print('use {:.0f}h {:.0f}m {:.0f}s'.format(
                    time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if epoch + 1 >= 10:
                    filename = os.path.join(config.checkpoint, 'Acc:%4f.pth.tar' % best_acc)
                    print('new best acc, save at %s' % filename)
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, filename)

        torch.cuda.empty_cache()
        print()

    return best_acc


if __name__ == '__main__':
    main()
