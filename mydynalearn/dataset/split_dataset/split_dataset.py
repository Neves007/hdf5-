from .train_set import TrainSet
from .val_set import ValSet
from .test_set import TestSet
from mydynalearn.dataset import PeriodicDateset
class SplitDataset():
    def __init__(self, config):
        self.periodic_dateset = PeriodicDateset(config)
        self.train_set = TrainSet(self.periodic_dateset.config, self.periodic_dateset.dao.group_path)
        self.val_set = ValSet(self.periodic_dateset.config, self.periodic_dateset.dao.group_path)
        self.test_set = TestSet(self.periodic_dateset.config, self.periodic_dateset.dao.group_path)
        pass

    def run(self):
        dataset = self.periodic_dateset.get_dataset()
        # 创建 TensorDataset
        # 按照 45, 45, 10 的比例划分数据集
        # 将原始数据拆分为 train, val, test
        train_size = self.train_set.metadata['NUM_TRAIN']
        val_size = self.val_set.metadata['NUM_VAL']
        test_size = self.test_set.metadata['NUM_TEST']

        # 使用 slicing 来对每个 tensor 进行分割
        train_set = {
            "x0_T": dataset["x0_T"][:train_size],
            "y_ob_T": dataset["y_ob_T"][:train_size],
            "y_true_T": dataset["y_true_T"][:train_size],
            "weight_T": dataset["weight_T"][:train_size],
        }

        val_set = {
            "x0_T": dataset["x0_T"][train_size:train_size + val_size],
            "y_ob_T": dataset["y_ob_T"][train_size:train_size + val_size],
            "y_true_T": dataset["y_true_T"][train_size:train_size + val_size],
            "weight_T": dataset["weight_T"][train_size:train_size + val_size],
        }

        test_set = {
            "x0_T": dataset["x0_T"][-test_size:],
            "y_ob_T": dataset["y_ob_T"][-test_size:],
            "y_true_T": dataset["y_true_T"][-test_size:],
            "weight_T": dataset["weight_T"][-test_size:],
        }

        self.train_set.run(train_set)
        self.val_set.run(val_set)
        self.test_set.run(test_set)

