import os
import torch
import glob
import re


class BesetEpochPicker:
    def __init__(self, gnnExpeiment_Config, exp,train_set,val_set) -> None:
        self.gnnExpeiment_Config = gnnExpeiment_Config
        self.EPOCHS = self.gnnExpeiment_Config.trainingArgs.EPOCHS
        self.patience = self.gnnExpeiment_Config.trainingArgs.patience
        self.path = self.gnnExpeiment_Config.datapath_to_model
        self.train_set = train_set
        self.val_set = val_set
        self.exp = exp
        self.loss_values = []
        self.best = self.EPOCHS + 1
        self.best_epoch = 0
        self.clean_dir()
    def clean_dir(self):
        files = glob.glob(self.path + '*.pkl')
        for file in files:
                os.remove(file)

    def __deleteOtherEpoch(self):
        best_epoch_file_name = self.path+str(self.best_epoch) + '.pkl'
        # 把bestepoch之前的都删掉
        files = glob.glob(self.path + '*.pkl')
        for file in files:
            if file != best_epoch_file_name:
                os.remove(file)

    def pickBestEpoch(self):
        self.bad_counter = 0
        # 只留一个epoch就是最好的epoch
        for cur_epoch in range(self.EPOCHS):
            self.exp.visdomDrawer.init_window()
            loss, acc = self.exp.train(cur_epoch, self.train_set, self.val_set)
            # self.exp.low_the_lr()
            self.exp.visdomDrawer.draw_epoch(loss,acc,cur_epoch)
            self.loss_values.append(loss)

            torch.save(self.exp.model.state_dict(), self.path + '{}.pkl'.format(cur_epoch))
            if self.loss_values[-1] < self.best:
                self.best = self.loss_values[-1]
                self.best_epoch = cur_epoch
                self.bad_counter = 0
            else:
                self.bad_counter += 1

            if self.bad_counter == self.patience:
                break
        self.__deleteOtherEpoch()
        BestEpoch_file_name = self.path + "BestEpoch.pkl"
        os.rename(self.path + os.listdir(self.path)[0], BestEpoch_file_name)
        return BestEpoch_file_name