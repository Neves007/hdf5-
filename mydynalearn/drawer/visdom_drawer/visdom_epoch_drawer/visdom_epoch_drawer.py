from visdom import Visdom
import torch
import numpy as np

class VisdomEpochDrawer:
    def __init__(self) -> None:

        self.wind = Visdom()
        # 初始化窗口参数
        self.wind.scatter([[0,0.],[0,0.]], win='epoch_loss', opts=dict(title='epoch loss', legend=['epoch loss'],                    xtickmin=0, # 坐标设置
                    xtickmax=30,
                    xtickstep=5,
                    ytickmin=0,
                    ytickmax=1))
        self.wind.scatter([[0,0.],[0,0.]], win='epoch_acc', opts=dict(title='epoch acc', legend=['epoch acc'],                    xtickmin=0, # 坐标设置
                    xtickmax=30,
                    xtickstep=5,
                    ytickmin=0,
                    ytickmax=1))

    def draw_epoch(self,loss,acc,epoch):
        self.wind.scatter([[epoch, loss.data.item()]], win='epoch_loss', update='append')
        self.wind.scatter([[epoch, acc.data.item()]], win='epoch_acc', update='append')

