from torchmetrics.functional import accuracy
def acc(y_true,y_pred):
    num_classes = y_pred.shape()
    acc_value = accuracy(y_pred,y_true,num_classes=3,task="multiclass")
    return acc_value