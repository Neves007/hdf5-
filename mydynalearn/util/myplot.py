import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt

class myplot():
    def __init__(self, x, y, xlabel="x", ylabel="y"):
        if torch.is_tensor(x):
            self.x = x.cpu().numpy()
        else:
            self.x = x
        if torch.is_tensor(x):
            self.y = y.cpu().numpy()
        else:
            self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
    def plot(self):
        plt.plot(self.x,self.y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

    def bar(self):
        plt.bar(self.x,self.y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()