import torch
import torch.nn as nn
import math


class CauchyLoss(nn.Module):
    def __init__(self, gamma=1):
        super(CauchyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        diff = inputs - targets
        loss = torch.mean(0.5 * torch.log((self.gamma + diff ** 2) / self.gamma))
        return loss


import numpy as np
from sklearn.metrics import mutual_info_score


class MILoss(nn.Module):
    def __init__(self, loss_l, loss_a):
        super(MILoss, self).__init__()
        self.loss_l = loss_l
        self.loss_a = loss_a

    def compute_mi(self, X, Y, num_bins=30):
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy().flatten()

        # 初始化互信息
        MI_total = 0.0

        # 对每个特征进行互信息计算
        for i in range(X_np.shape[1]):  # 遍历每个特征
            x_feature = X_np[:, i]  # 取出当前特征
            # 计算联合直方图
            hist_2d, x_edges, y_edges = np.histogram2d(x_feature, Y_np, bins=num_bins)

            # 计算边际分布
            p_x = hist_2d.sum(axis=1)
            p_y = hist_2d.sum(axis=0)
            p_xy = hist_2d.flatten()

            # 计算互信息
            p_x = p_x / p_x.sum()  # 归一化
            p_y = p_y / p_y.sum()  # 归一化
            p_xy = p_xy / p_xy.sum()  # 归一化

            # 避免对数的计算出现 zero
            p_x[p_x == 0] = 1e-10
            p_y[p_y == 0] = 1e-10
            p_xy[p_xy == 0] = 1e-10

            MI = 0.0
            for j in range(num_bins):
                for k in range(num_bins):
                    MI += p_xy[j * num_bins + k] * np.log(p_xy[j * num_bins + k] / (p_x[j] * p_y[k]))

            MI_total += MI

        return MI_total

    def forward(self, y_true, y_pred, u_attr, s_attr, u_vec, s_vec):
        bins = 30

        L_PT = torch.mean(torch.abs(y_true - y_pred))

        L_MI_uq = 0
        L_MI_sq = 0
        L_MI_uq += self.compute_mi(u_vec, y_pred)
        L_MI_sq += self.compute_mi(s_vec, y_pred)

        L_MI_us = 0
        for i in range(u_vec.size(1)):
            u_vec_i = u_vec[i].detach().cpu().numpy().flatten()
            s_vec_i = s_vec[i].detach().cpu().numpy().flatten()

            u_vec_binned = np.digitize(u_vec_i, bins=np.linspace(np.min(u_vec_i), np.max(u_vec_i), bins))
            s_vec_binned = np.digitize(s_vec_i, bins=np.linspace(np.min(s_vec_i), np.max(s_vec_i), bins))

            L_MI_us += mutual_info_score(u_vec_binned, s_vec_binned)

        return L_PT + self.loss_l * (L_MI_uq + L_MI_sq - self.loss_a * L_MI_us)
