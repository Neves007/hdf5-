import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
def _unpacktest_result_curepochResult(test_result_curepoch):
    T = len(test_result_curepoch)
    loss_all = 1.*torch.zeros(T)
    acc_all = 1.*torch.zeros(T)
    for t,data in enumerate(test_result_curepoch):
        time_index = data['time_index']
        loss = data['loss']
        acc = data['acc']
        x = data['x']
        y_pred = data['y_pred']
        y_true = data['y_true']
        y_ob = data['y_ob']
        loss_all[t] = loss
        acc_all[t] = acc
    return loss_all,acc_all

def unpackBatchData(data):
    # 加weight
    epoch_index = data['epoch_index']
    time_index = data['time_index']
    loss = data['loss']
    acc = data['acc']
    x = data['x']
    y_pred = data['y_pred']
    y_true = data['y_true']
    y_ob = data['y_ob']
    w = data['w']
    return epoch_index, time_index, loss, acc, x, y_pred, y_true, y_ob, w

def compute_test_result_curepoch_loss_acc(test_result_curepoch):
    loss_all,acc_all = _unpacktest_result_curepochResult(test_result_curepoch)
    return loss_all.mean(),acc_all.mean()

def _get_metrics( performance_data):
    true_pred = torch.cat(performance_data).detach().numpy()
    corrcoef = np.corrcoef(true_pred.T)[0, 1]
    r2 = r2_score(true_pred[:, 0], true_pred[:, 1])
    return corrcoef, r2