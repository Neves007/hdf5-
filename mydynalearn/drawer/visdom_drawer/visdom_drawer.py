from visdom import Visdom
import torch
import numpy as np

class VisdomDrawer:
    def __init__(self,dynamics) -> None:

        self.wind = Visdom()
        # 初始化窗口参数
        self.wind.line([0.], [0.],win = 'train_loss',opts = dict(title = 'train_loss',legend = ['train_loss'],                    xtickmin=0, # 坐标设置
                            ytickmin=0,
                            ytickmax=1))
        self.wind.line([0.], [0.],win = 'train_acc',opts = dict(title = 'train_acc',legend = ['train_acc'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
        self.wind.line([0.], [0.],win = 'val_loss',opts = dict(title = 'val_loss',legend = ['val_loss'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
        self.wind.line([0.], [0.],win = 'val_acc',opts = dict(title = 'val_acc',legend = ['val_acc'],                    xtickmin=0, # 坐标设置
                    ytickmin=0,
                    ytickmax=1))
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

    def init_window(self):
        # 初始化窗口参数
        self.wind.line([0.], [0.],win = 'train_loss',opts = dict(title = 'train_loss',legend = ['train_loss']))
        self.wind.line([0.], [0.],win = 'train_acc',opts = dict(title = 'train_acc',legend = ['train_acc']))
        self.wind.line([0.], [0.],win = 'val_loss',opts = dict(title = 'val_loss',legend = ['val_loss']))
        self.wind.line([0.], [0.],win = 'val_acc',opts = dict(title = 'val_acc',legend = ['val_acc']))

    def draw_acc_loss(self,train_loss, train_acc, val_loss,val_acc, time_index):
        opts = dict(
                    ytickmin=0,
                    ytickmax=1,
                    ytickstep=0.1,
                    )

        self.wind.line([train_loss.data.item()], [time_index], win='train_loss', opts=opts,update='append')
        self.wind.line([train_acc.data.item()], [time_index], win='train_acc',opts=opts, update='append')
        self.wind.line([val_loss.data.item()], [time_index], win='val_loss', opts=opts,update='append')
        self.wind.line([val_acc.data.item()], [time_index], win='val_acc', opts=opts,update='append')

    def get_performance_data(self,x,predict_TP, y_ob, fun_TP):
        self.STATES_MAP = {"S": 0, "I": 1}  # [S,I]
        with torch.no_grad():
            pre_labels = predict_TP.max(1)[1].type_as(y_ob)
            ob_labels = y_ob.max(1)[1].type_as(y_ob)
            right_prediction = pre_labels == ob_labels

            S_S = torch.where((x[:, self.STATES_MAP["S"]] == 1) & (ob_labels == 0))[0]
            S_I = torch.where((x[:, self.STATES_MAP["S"]] == 1) & (ob_labels == 1))[0]
            I_S = torch.where((x[:, self.STATES_MAP["I"]] == 1) & (ob_labels == 0))[0]
            I_I = torch.where((x[:, self.STATES_MAP["I"]] == 1) & (ob_labels == 1))[0]
            # 混淆矩阵
            S_S_fun_pre = torch.cat((fun_TP[S_S,0].view(-1,1), predict_TP[S_S,0].view(-1,1)),dim=1)
            S_I_fun_pre = torch.cat((fun_TP[S_I,1].view(-1,1), predict_TP[S_I,1].view(-1,1)),dim=1)
            I_S_fun_pre = torch.cat((fun_TP[I_S,0].view(-1,1), predict_TP[I_S,0].view(-1,1)),dim=1)
            I_I_fun_pre = torch.cat((fun_TP[I_I,1].view(-1,1), predict_TP[I_I,1].view(-1,1)),dim=1)
            S_S_fun_pre = S_S_fun_pre.cpu().numpy()
            S_I_fun_pre = S_I_fun_pre.cpu().numpy()
            I_S_fun_pre = I_S_fun_pre.cpu().numpy()
            I_I_fun_pre = I_I_fun_pre.cpu().numpy()
        return S_S_fun_pre, S_I_fun_pre, I_S_fun_pre, I_I_fun_pre

    def draw_performance(self,x,predict_TP, y_ob, fun_TP):
        S_S_fun_pre, S_I_fun_pre, I_S_fun_pre, I_I_fun_pre = self.get_performance_data(x,predict_TP, y_ob, fun_TP)

        red =np.array([ 242,0,0]).reshape(1,3)
        green =np.array([ 146,195,47]).reshape(1,3)
        orange =np.array([ 242,157,0]).reshape(1,3)
        b =np.array([ 66,123,171]).reshape(1,3)
        colors = np.concatenate((red, green, orange, b), axis=0).astype(np.int16)

        S_S_draw_data = {"X":S_S_fun_pre,"Y":(1*np.ones(S_S_fun_pre.shape[0])).astype(np.int16),"color":red,"mark":'cross'}
        S_I_draw_data = {"X":S_I_fun_pre,"Y":(2*np.ones(S_I_fun_pre.shape[0])).astype(np.int16),"color":green,"mark":"star"}
        I_S_draw_data = {"X":I_S_fun_pre,"Y":(3*np.ones(I_S_fun_pre.shape[0])).astype(np.int16),"color":orange,"mark":"triangle-left"}
        I_I_draw_data = {"X":I_I_fun_pre,"Y":(4*np.ones(I_I_fun_pre.shape[0])).astype(np.int16),"color":b,"mark":"x"}

        X = np.concatenate((S_S_draw_data['X'],S_I_draw_data['X'],I_S_draw_data['X'],I_I_draw_data['X']))
        Y = np.concatenate((S_S_draw_data['Y'],S_I_draw_data['Y'],I_S_draw_data['Y'],I_I_draw_data['Y']))
        corrcoef = np.corrcoef(X.T)[0,1]
        opts = dict(title='Train performance, r = {:0.4f}'.format(corrcoef),
                    legend=["S_S","S_I","I_S","I_I"],
                    xtickmin=0, # 坐标设置
                    xtickmax=1,
                    xtickstep=0.1,
                    ytickmin=0,
                    ytickmax=1,
                    ytickstep=0.1,
                    markercolor=colors,
                    markersize=5,
                    )
        self.wind.scatter(X=X,Y=Y, win='Train performance', opts=opts)

        # opts['markersymbol'] = "star"
        # opts['markercolor'] = green
        # self.wind.scatter(X=S_I_X, win='Train performance', opts=opts)
        # self.wind.scatter(X=I_S_X, win='Train performance', opts=opts)
        # self.wind.scatter(X=I_I_X, win='Train performance', opts=opts)

    def draw_epoch(self,loss,acc,epoch):
        self.wind.scatter([[epoch, loss.data.item()]], win='epoch_loss', update='append')
        self.wind.scatter([[epoch, acc.data.item()]], win='epoch_acc', update='append')

