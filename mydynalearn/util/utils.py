import torch



def accuracy(output, labels,x):
    pre_labels = output.max(1)[1].type_as(labels)
    ob_labels = labels.max(1)[1].type_as(labels)

    right_prediction = pre_labels == ob_labels

    N = x.shape[0]
    S_S = torch.where((x==0) & (ob_labels == 0))[0].shape[0]
    S_I= torch.where((x == 0) & (ob_labels == 1))[0].shape[0]
    I_S = torch.where((x==1) & (ob_labels == 0))[0].shape[0]
    I_I = torch.where((x==1) & (ob_labels == 1))[0].shape[0]

    S_S_Right = torch.where((x==0) & (ob_labels == 0) & right_prediction)[0].shape[0]
    S_I_Right = torch.where((x==0) & (ob_labels == 1) & right_prediction)[0].shape[0]
    I_S_Right = torch.where((x==1) & (ob_labels == 0) & right_prediction)[0].shape[0]
    I_I_Right = torch.where((x==1) & (ob_labels == 1) & right_prediction)[0].shape[0]
    # 混淆矩阵
    acc_value = torch.where(right_prediction)[0].shape[0]/N

    acc = {
        "S_S_Right":int(S_S_Right),
        "S_I_Right":int(S_I_Right),
        "I_S_Right":int(I_S_Right),
        "I_I_Right":int(I_I_Right),
        "S_S":S_S,
        "S_I":S_I,
        "I_S":I_S,
        "I_I":I_I
    }
    
    return acc_value,acc

