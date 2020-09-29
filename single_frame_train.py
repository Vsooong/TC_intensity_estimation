import torch
import numpy as np
import os
import torch.nn as nn
from utils.Utils import args
from TC_data import TC_Data
from TC_estimate.single_frame import get_pretrained_model


def train_one_epoch(model, dataset, optimizer, criterion):
    model.train()
    loss_epoch = 0
    for minibatch in dataset.get_batches():
        X_im, Y_int = minibatch
        images = X_im[:, 0, ...]
        targets = Y_int[:, 0, ...]
        pred = model(images)
        optimizer.zero_grad()
        loss = criterion(targets, pred)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(loss.item())
    return loss_epoch


evaluateL1 = nn.L1Loss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')


def evaluate(model, dataset):
    model.eval()
    n_samples = 0
    total_loss1 = 0
    total_loss2 = 0
    labels = []
    predicts = []

    for minibatch in dataset.get_batches():
        images, targets = minibatch
        images = images[:, 0, ...]
        targets = targets[:, 0, ...]
        pred = model(images)
        total_loss1 += evaluateL1(targets, pred).data.item()
        total_loss2 += evaluateL2(targets, pred).data.item()
        n_samples += len(targets)
        for index in range(len(targets)):
            labels.append(targets[index].data.item())
            predicts.append(pred[index].data.item())
    r = np.corrcoef(labels, predicts)[0][1]
    return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples), r


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y)) + self.mae(yhat, y)


def main(train_process=False):
    device = args.device
    dataset = TC_Data()
    dataset_test = TC_Data(years=args.test_years)

    model = get_pretrained_model()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.8)
    criterion = Criterion().to(device)
    best_loss = 9999999
    if train_process == True:
        for epoch in range(args.epochs):
            loss_epoch = train_one_epoch(model, dataset, optimizer, criterion)
            lr_scheduler.step()
            print('Epoch:', epoch, loss_epoch)
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                torch.save(model.state_dict(), os.path.join(args.save_model, 'resnet_50.pth'))
                print('performance improved, save model to:', args.model_save1)
            if epoch % 3 == 0:
                loss1, loss2, r = evaluate(model, dataset_test)
                print("test performance:", loss1, loss2, r)

        print('training finish ')
    else:
        loss1, loss2, r = evaluate(model, dataset_test)
        print(loss1, loss2, r)

    # for epoch in range(args.epochs):
    #     # train for one epoch, printing every 10 iterations
    #     loss_epoch = train_one_epoch(model, optimizer, train_loader, criterion)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset
    #     # evaluate(model, data_loader_test, device=device)
    #     # torch.save(model.state_dict(), save_path)
    #     print(loss_epoch)


if __name__ == '__main__':
    main(train_process=True)
