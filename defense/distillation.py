import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsm = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):

        y_d = self.logsm(outputs)
        L = torch.sum(-(targets*y_d), dim=1)

        return torch.mean(L)


def Train(data, model, epoch=200, train_temp=1, student=False):
    x_train = torch.from_numpy(data['x_train']).float()
    if student:
        y_train = torch.from_numpy(data['y_train']).float()
    else:
        y_train = torch.LongTensor(data['y_train'])
    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
    
    model.weight_init()

    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 1e-3}], betas=(0.5, 0.999))

    for e in range(epoch):
        cost = 0.

        # train
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            # print(y)
            # print(list(x.size()))
            logit = model(x)
            # print(list(logit.size()))
            if student:
                criterion = My_loss()
                cost = criterion(logit/train_temp, y)
            else:
                prediction = logit.max(1)[1]
                cost = F.cross_entropy(logit/train_temp, y)  # 交叉熵
                
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        valid_best = 0
        # test
        if (e+1) % 10 == 0:
            print(e)
            model.eval()
            logit = model(torch.from_numpy(data['x_valid']).float())
            prediction = logit.max(1)[1]
            valid_correct = torch.eq(prediction, torch.LongTensor(data['y_valid'])).float().mean().numpy()
            print(valid_correct)
            logit = model(torch.from_numpy(data['x_test']).float())
            prediction = logit.max(1)[1]
            test_correct = torch.eq(prediction, torch.LongTensor(data['y_test'])).float().mean().numpy()
            print(test_correct)
            if valid_correct > valid_best:
                valid_best = valid_correct
                torch.save(model,'models/best_temp.pkl')
    best_model = torch.load('models/best_temp.pkl')
    print(" [*] Training Finished!")
    return best_model


def Train_distillation(data,net, epoch=1000, train_temp=100):
    from defense.plain_dnn import train as plain_train
    
    print("#######start train teacher net#######")
    # teacher = Train(path + "_teacher", dataloader, epoch, train_temp)
    # teacher = Train(data, net, epoch, train_temp)
    teacher = torch.load( 'models/TEP/distillation/model_t.pkl')
    # teacher = torch.load( 'models/TEP/plain_dnn/model.pkl')
    # evalutate labels
    eva_data = get_dataloader_evaluate(data, teacher, train_temp)
    print("#######start train student net#######")
    student = Train(eva_data, net, epoch, train_temp, True)

    return teacher,student


def get_dataloader_evaluate(nparray, model, train_temp):

    x_train = nparray['x_train']
    x_train = torch.from_numpy(x_train).float()
    y_logit = model(x_train)
    y_train = F.softmax(y_logit / train_temp, dim=1).detach().numpy()
    nparray['y_train'] = y_train

    return nparray
