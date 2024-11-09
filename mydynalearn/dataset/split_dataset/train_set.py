import torch

from Dao import DataHandler
from mydynalearn.logger import Log
from torch.utils.data import Dataset, DataLoader

class TrainSet(Dataset, DataHandler):
    def __init__(self, config, parent_group, cur_group="train_set"):
        self.config = config
        self.DEVICE = config.DEVICE
        self.parent_group = parent_group
        self.cur_group = cur_group
        self.log = Log("TrainSet")
        DataHandler.__init__(self, self.parent_group, self.cur_group)

    def init_metadata(self):
        dataset_config = self.config.dataset
        metadata = {}
        metadata['NUM_TRAIN'] = int((dataset_config['NUM_SAMPLES'] - dataset_config['NUM_TEST'])/2)
        metadata['IS_WEIGHT'] = dataset_config['IS_WEIGHT']
        self.set_metadata(metadata)

    def __len__(self):
        if not hasattr(self,"dataset"):
            self.load()
        T,N,S=self.dataset['x0_T'].size()
        # 数据集中样本的总数为时间步数 * 节点数
        return int(T)

    def __getitem__(self, idx):
        if not hasattr(self, "dataset"):
            self.load()
        x0 = self.dataset["x0_T"][idx,:,:]
        y_ob = self.dataset["y_ob_T"][idx,:,:]
        y_true = self.dataset["y_true_T"][idx,:,:]
        weight = self.dataset["weight_T"][idx,:]
        return x0, y_ob, y_true, weight

    def run(self, dataset):
        self.log.increase_indent()
        if self._get_build_necessity():
            self.log.log("Building dataset...")
            self.set_dataset(dataset)
            self._save()
            self.log.log(f"Dataset '{self.dao.group_path}' created and stored.")
        else:
            self.log.log(f"Dataset '{self.dao.group_path}'  already exists.")
        self.log.decrease_indent()

    def convert_from_storage_format(self, dataset):
        for key, value in dataset.items():
            dataset[key] = torch.tensor(value, device=self.DEVICE)
        return dataset