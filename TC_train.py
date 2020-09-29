import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
from Utils import args
from TC_data import TC_Data
from TC_estimate.single_frame import get_pretrained_model


def train_one_epoch(model, optimizer, data_loader, criterion):
    model.train()
    loss_epoch = 0
    for images, targets in data_loader:
        optimizer.zero_grad()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        pred = model(images)[0]
        # print(pred)
        # print(targets)
        loss = criterion(targets, pred)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        print(loss.item())
    return loss_epoch


def train():
    device = args.device
    dataset = TC_Data()
    # dataset_test = VideoDataset(get_transform(train=False))

    model = get_pretrained_model()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = nn.MSELoss().to(device)

    for epoch in range(args.epochs):
        # train for one epoch, printing every 10 iterations
        loss_epoch = train_one_epoch(model, optimizer, train_loader, criterion)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        # torch.save(model.state_dict(), save_path)
        print(loss_epoch)


if __name__ == '__main__':
    train()
