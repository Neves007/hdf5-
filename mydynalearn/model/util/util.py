from mydynalearn.transformer import *
from mydynalearn.metrics import *
def get_acc(y_y_ob,y_pred_TP):
    y_pred_class = TP_to_class(y_pred_TP)
    y_ob_class = TP_to_class(y_y_ob)
    num_classes = y_y_ob.shape[1]
    acc_value = accuracy(y_pred_class,y_ob_class,num_classes=num_classes,task="multiclass")
    return acc_value