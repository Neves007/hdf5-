import torch
import itertools

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

def get_all_transition_types(dynamics_STATES_MAP):
    '''

    :param dynamics_STATES_MAP:
    :return:
    '''
    states = dynamics_STATES_MAP.keys()
    all_transition_types = itertools.product(states,states)
    return list(all_transition_types)


