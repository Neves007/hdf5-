import matplotlib.pyplot as plt
import torch
class FigBetaRho():
    def __init__(self,dynamics, x, stady_rho_dict):
        self.x = x
        self.stady_rho_dict = stady_rho_dict
        self.state_list = dynamics.STATES_MAP.keys()
        self.dynamic_name = dynamics.NAME
        self.num_fig = len(self.state_list)

    def save_fig(self,fig_file):
        self.fig.savefig(fig_file)
        plt.close(self.fig)
    def draw(self):
        self.fig, ax = plt.subplots(1, self.num_fig, figsize=(5 * self.num_fig, 4.5))
        for i, state in enumerate(self.state_list):
            y = self.stady_rho_dict[state]
            ax[i].set_title('{} dynamic model'.format(self.dynamic_name))
            ax[i].plot(self.x, y,marker='^')  # Plot x vs. y with circle markers
            ax[i].set_xlabel("Effective Infection Rate")  # X-axis label
            ax[i].set_ylabel("$\\rho_{{{:s}}}$".format(state))  # Y-axis label
            ax[i].set_ylim([0, 1])  # Set the limits for the y-axis
            ax[i].set_xlim([0, self.x[-1]])  # Set the limits for the y-axis
            ax[i].grid(True)  # Show grid
        plt.tight_layout()

