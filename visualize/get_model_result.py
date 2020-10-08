from single_frame_train import  evaluateL1, evaluateL2
from utils.Utils import args
from TC_data import TC_Data
import os
import numpy as np
from TC_estimate.MSFN import get_MSFN


def evaluate(model, dataset):
    model.eval()
    n_samples = 0
    total_loss1 = 0
    total_loss2 = 0
    labels = []
    predicts = []

    for minibatch in dataset.get_batches():
        images, efactors, envsst, targets = minibatch
        pred = model(images, efactors, envsst)
        if np.isnan(pred.data.cpu()).sum() != 0:
            print(efactors)
            print(targets)
            print(pred)
        targets = targets[:, -1, :]

        total_loss1 += evaluateL1(targets, pred).data.item()
        total_loss2 += evaluateL2(targets, pred).data.item()
        n_samples += len(targets)
        for index in range(len(targets)):
            labels.append(targets[index].data.item())
            predicts.append(pred[index].data.item())
    r = np.corrcoef(labels, predicts)[0][1]
    return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples), r


def main():
    model = get_MSFN(load_states=True)
    model_name = 'MSFN.pth'
    print('use model:', model_name)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)
    dataset_test = TC_Data(years=[2016])
    # dataset_test = TC_Data(years=args.test_years)
    print('Test samples:', len(dataset_test.targets))
    print('------------------------------------------\n')
    loss1, loss2, r = evaluate(model, dataset_test)
    print(loss1, loss2, r)


if __name__ == '__main__':
    main()
