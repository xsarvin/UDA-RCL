import torch
from torch.utils.data import Dataset
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import networkx as nx


class EventDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集。

        参数:
        data: 数据列表，可以是任何形式，例如特征向量列表。
        transform: 可选的变换（transform），用于对数据进行预处理。
        """
        self.data = data

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本。

        参数:
        idx: 样本的索引。

        返回:
        一个包含样本数据和标签的元组（如果适用）。
        """
        # 获取原始数据

        log_e = self.data[idx][0]
        log_c = self.data[idx][1]
        metric_e = self.data[idx][2]
        metric_c = self.data[idx][3]
        trace_e = self.data[idx][4]
        trace_c = self.data[idx][5]
        adj = self.data[idx][6]
        label = self.data[idx][7]
        adj = np.where(adj > 0, 1, 0)

        # return torch.Tensor(fault), torch.Tensor(adj), label, contain_event
        return np.array(log_e), np.array(metric_e), np.array(trace_e), np.array(log_c), np.array(metric_c), np.array(trace_c), np.array(adj), np.array(label)


class VariableDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集。

        参数:
        data: 数据列表，可以是任何形式，例如特征向量列表。
        transform: 可选的变换（transform），用于对数据进行预处理。
        """
        self.data = data

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本。

        参数:
        idx: 样本的索引。

        返回:
        一个包含样本数据和标签的元组（如果适用）。
        """
        # 获取原始数据

        fault_feature = self.data[idx][0]
        adj = self.data[idx][2]
        label = self.data[idx][3]

        return torch.FloatTensor(fault_feature), torch.tensor(adj), label
