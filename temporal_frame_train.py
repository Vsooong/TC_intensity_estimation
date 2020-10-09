import torch
from single_frame_train import Criterion, evaluateL1, evaluateL2
from utils.Utils import args
from TC_data import TC_Data
import os
import numpy as np
from TC_estimate.temporal_frame import get_basline_model
from TC_estimate.MSFN_DC import get_MSFN_DC
from TC_estimate.MSFN_GF import get_MSFN_GF
from TC_estimate.MSFN import get_MSFN


def train_one_epoch(model, dataset, optimizer, criterion):
    global which_model
    model.train()
    loss_epoch = 0
    for one_ty in dataset.get_one_ty():
        images, efactors, envsst, targets = one_ty
        if which_model == 1:
            pred = model(images).squeeze(-1)
        else:
            pred = model(images, efactors, envsst).squeeze(-1)
        targets=targets[0,:,:]
        optimizer.zero_grad()
        loss = criterion(targets, pred)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(loss.item())
    return loss_epoch


def evaluate(model, dataset):
    global which_model
    model.eval()
    n_samples = 0
    total_loss1 = 0
    total_loss2 = 0
    labels = []
    predicts = []

    for one_ty in dataset.get_one_ty():
        images, efactors, envsst, targets = one_ty
        if which_model == 1:
            pred = model(images).squeeze(-1)
        else:
            pred = model(images, efactors, envsst).squeeze(-1)
        targets = targets[0, :, :]
        if np.isnan(pred.data.cpu()).sum() != 0:
            print(efactors)
            print(targets)
            print(pred)
        total_loss1 += evaluateL1(targets, pred).data.item()
        total_loss2 += evaluateL2(targets, pred).data.item()
        n_samples += len(targets)
        for index in range(len(targets)):
            labels.append(targets[index].data.item())
            predicts.append(pred[index].data.item())
    r = np.corrcoef(labels, predicts)[0][1]
    return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples), r


def main(train_process=False, load_states=False):
    global which_model
    device = args.device

    if which_model == 1:
        model = get_basline_model(load_states=load_states)
        model_name = 'convlstm.pth'
    elif which_model == 2:
        model = get_MSFN_DC(load_states=load_states)
        model_name = 'MSFN_DC.pth'
    elif which_model == 3:
        model = get_MSFN_GF(load_states=load_states)
        model_name = 'MSFN_GF.pth'
    else:
        model = get_MSFN(load_states=load_states)
        model_name = 'MSFN.pth'
    print('use model:', model_name)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    # dataset = TC_Data(years=[2006])
    test_years = [2000, 2006, 2011, 2017]
    dataset_test = TC_Data(years=test_years)
    dataset = TC_Data()
    # dataset_test = TC_Data(years=args.test_years)
    print('Training samples:', len(dataset.targets))
    print('Test samples:', len(dataset_test.targets))
    print('----------------------------------------------')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)
    criterion = Criterion().to(device)
    best_loss = 9999999
    if train_process == True:
        for epoch in range(args.epochs):
            if epoch % 3 == 0:
                loss1, loss2, r = evaluate(model, dataset_test)
                print("test performance:", loss1, loss2, r)
                if loss2 < best_loss:
                    best_loss = loss2
                    path = os.path.join(args.save_model, model_name)
                    torch.save(model.state_dict(), path)
                    print('test performance improved, save model to:', path)

            loss_epoch = train_one_epoch(model, dataset, optimizer, criterion)
            lr_scheduler.step()

            print('Epoch:', epoch, loss_epoch)
            print('----------------------------------------------')
        print('training finish ')
    else:
        loss1, loss2, r = evaluate(model, dataset_test)
        print(loss1, loss2, r)


if __name__ == '__main__':
    which_model = 2
    main(train_process=True, load_states=False)
