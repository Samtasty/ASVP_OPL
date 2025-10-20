import sys
import numpy as np

class LossHistory(object):

    def __init__(self, name):
        self.name = name
        self.crm_loss = []
        self.online_loss = []
        self.betas = []
        self.n_samples = []
        self.n_actions = []
        self.cumulated_loss = []
        self.losses_baseline = []
        self.losses_skyline = []
        self.regret = []

    def update(self, beta, online_loss, regret, crm_loss, cumulated_losses, losses_baseline, losses_skyline, n_samples):
        self.betas += [beta]
        self.online_loss += [online_loss]
        self.crm_loss += [crm_loss]
        self.losses_baseline += [losses_baseline]
        self.losses_skyline += [losses_skyline]
        self.n_samples += [n_samples]
        self.cumulated_loss += [np.sum(self.cumulated_loss) + cumulated_losses]
        self.regret += [np.sum(self.regret) + regret * n_samples]

    def show_last(self):
        print(
            '<', self.name,
            'IPW loss: %.5f' % self.crm_loss[-1],
            'Online loss: %.5f' % self.online_loss[-1],
            '|theta|=%.2f' % np.sqrt((self.betas[-1] ** 2).sum()),
            'n=%d' % sum(self.n_samples[:-1]),
            '>',
            file=sys.stderr
        )