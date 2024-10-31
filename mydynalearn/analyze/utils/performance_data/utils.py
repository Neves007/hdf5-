import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
def _get_metrics( performance_data):
    true_pred = torch.cat(performance_data).detach().numpy()
    corrcoef = np.corrcoef(true_pred.T)[0, 1]
    r2 = r2_score(true_pred[:, 0], true_pred[:, 1])
    return corrcoef, r2