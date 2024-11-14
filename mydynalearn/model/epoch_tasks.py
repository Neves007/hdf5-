import os
import torch
import numpy as np
from mydynalearn.networks.getter import get as net_getter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mydynalearn.model.optimizer import get as get_optimizer
from mydynalearn.model.getter import get as get_model
from mydynalearn.model.batch_task import BatchTask
from mydynalearn.logger import Log
from Dao import DataHandler
from mydynalearn.dataset import SplitDataset


class EpochTasks(DataHandler):
    def __init__(self, config, epoch_index, *args, **kwargs):
        self.config = config
        self.epoch_index = epoch_index
        self.init_metadata()
        self.logger = Log("EpochTasks")
        self.split_dataset = SplitDataset(config)
        self.batch_task = BatchTask(config)
        #
        self.model = get_model(config)
        self.get_optimizer = get_optimizer(config)
        self.optimizer = self.get_optimizer(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4, eps = 1e-8, threshold =0.1)
        #
        parent_group = "model"
        cur_group = (
                     f"NETWORK_NAME_{self.metadata['NETWORK_NAME']}_"
                     f"NUM_NODES_{self.metadata['NUM_NODES']}/"
                     f"DYNAMIC_NAME_{self.metadata['DYNAMIC_NAME']}_"
                     f"T_INIT_{self.metadata['T_INIT']}_"
                     f"SEED_FREC_{self.metadata['SEED_FREC']}/"
                     f"MODEL_NAME_{self.metadata['MODEL_NAME']}/"
                     f"epoch_{self.epoch_index}"
                     )
        DataHandler.__init__(self, parent_group, cur_group)
        self.params_file = self.metadata['params_file']



    def init_metadata(self):
        metadata = {}
        metadata['DEVICE'] = self.config.DEVICE
        # network
        metadata['NETWORK_NAME'] = self.config.network['NAME']
        metadata["NUM_NODES"] = self.config.network['NUM_NODES']
        # dynamics
        metadata['DYNAMIC_NAME'] = self.config.dynamics['NAME']
        metadata["SEED_FREC"] = self.config.dynamics['SEED_FREC']
        metadata["STATES"] = list(self.config.dynamics.STATES_MAP.keys())
        # dataset
        metadata['NUM_SAMPLES'] = self.config.dataset['NUM_SAMPLES']
        metadata['T_INIT'] = self.config.dataset['T_INIT']
        metadata['IS_WEIGHT'] = self.config.dataset['IS_WEIGHT']
        # model
        metadata['EPOCHS'] = self.config.model.EPOCHS
        metadata['MODEL_NAME'] = self.config.model.NAME
        metadata['epoch_index'] = self.epoch_index
        params_file_name = (
            f"NETWORK_NAME_{metadata['NETWORK_NAME']}_"
            f"NUM_NODES_{metadata['NUM_NODES']}_"
            f"DYNAMIC_NAME_{metadata['DYNAMIC_NAME']}_"
            f"T_INIT_{metadata['T_INIT']}_"
            f"SEED_FREC_{metadata['SEED_FREC']}_"
            f"MODEL_NAME_{metadata['MODEL_NAME']}_"
            f"epoch_{metadata['epoch_index']}.pth"
        )
        metadata['params_file'] = os.path.join(self.config.model.model_dir, params_file_name)
        self.metadata = metadata

    def save_params(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.params_file)

    def load_params(self):
        '''
        输入：当前epoch_index
        规则：加载模型参数
        输出：
        '''
        checkpoint = torch.load(self.params_file, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def low_the_lr(self):
        if (self.epoch_index>0) and (self.epoch_index % 5 == 0):
            self.optimizer.param_groups[0]['lr'] *= 0.5

    def build_dataset(self):
        self.logger.increase_indent()
        self.logger.log(f"train epoch: {self.epoch_index}")
        train_set = self.split_dataset.train_set
        val_set = self.split_dataset.val_set
        # 使用 DataLoader 加载训练集和验证集，指定 batch_size=3

        for train_dataset_batch, val_dataset_batch in zip(train_set, val_set):
            self.model.train()
            self.optimizer.zero_grad()
            train_loss, train_x, train_y_pred, train_y_true, train_y_ob, train_w = self.batch_task._do_batch_(self.model, train_dataset_batch)
            train_loss.backward()
            self.optimizer.step()
            self.model.eval()
            val_loss, val_x, val_y_pred, val_y_true, val_y_ob, val_w = self.batch_task._do_batch_(self.model, val_dataset_batch)
            y_true_flat = val_y_true.flatten()
            y_pred_flat = val_y_pred.flatten()
            R = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))[0, 1]
        dataset = {
            "epoch_index": self.epoch_index,
            "train_loss": train_loss,
            "train_x": train_x,
            "train_y_pred": train_y_pred,
            "train_y_true": train_y_true,
            "train_y_ob": train_y_ob,
            "train_w": train_w,
            "val_loss": val_loss,
            "val_x": val_x,
            "val_y_pred": val_y_pred,
            "val_y_true": val_y_true,
            "val_y_ob": val_y_ob,
            "val_w": val_w,
            "R": R,
        }
        self.set_dataset(dataset)
        self.logger.decrease_indent()

    def load(self):
        """
        加载数据集和元数据，并执行元反射操作
        """
        self.dataset = self.get_dataset()  # 从Dao对象加载数据集
        self.load_params()

    def end(self):
        self.save_params()


    def get_test_result(self):
        self.logger.increase_indent()
        self.logger.log("run test")
        self.model.eval()
        test_set = self.split_dataset.test_set
        batch_results = []
        network = net_getter(self.config)
        network.load()
        for test_dataset_batch in test_set:
            test_loss, test_x, test_y_pred, test_y_true, test_y_ob, test_w = self.batch_task._do_batch_(self.model, test_dataset_batch)
            batch_result  = {
                "x": test_x.detach().cpu().numpy(),
                "y_pred": test_y_pred.detach().cpu().numpy(),
                "y_true": test_y_true.detach().cpu().numpy(),
                "y_ob": test_y_ob.detach().cpu().numpy(),
                "w": test_w.detach().cpu().numpy(),
            }
            # 检查并添加 network 的附加属性
            if hasattr(network, "NUM_NEIGHBOR_NODES"):
                batch_result["NUM_NEIGHBOR_NODES"] = network.NUM_NEIGHBOR_NODES
            if hasattr(network, "NUM_NEIGHBOR_EDGES"):
                batch_result["NUM_NEIGHBOR_EDGES"] = network.NUM_NEIGHBOR_EDGES
            if hasattr(network, "NUM_NEIGHBOR_TRIANGLES"):
                batch_result["NUM_NEIGHBOR_TRIANGLES"] = network.NUM_NEIGHBOR_TRIANGLES
            batch_results.append(batch_result)

        total_dataset = self._merge_batch_results(batch_results)

        y_true_flat = total_dataset['y_true'].flatten()
        y_pred_flat = total_dataset['y_pred'].flatten()
        # 计算相关系数
        R = np.corrcoef(np.stack([y_true_flat, y_pred_flat]))[0, 1]
        node_loss = (-total_dataset['y_true'] * np.log(total_dataset['y_pred'])).sum(1)
        # 计算损失
        mean_loss = (-total_dataset['y_true'] * np.log(total_dataset['y_pred'])).sum(axis=-1).mean()
        dataset = {
            "epoch_index": self.epoch_index,
            "mean_loss": mean_loss,
            "R": R,
            "node_loss":node_loss
        }
        dataset.update(total_dataset)
        self.logger.decrease_indent()
        return dataset

    def _merge_batch_results(self, batch_results):
        """
        合并所有 batch 的结果为一个完整的 dataset。
        """
        merged_data = {field: np.concatenate([res[field] for res in batch_results], axis=0) for field in batch_results[0].keys()}
        return merged_data