import pandas as pd
from scipy.interpolate import griddata
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from model import AnomalyTransformer


def evaluate(topk, training=True):
    top_value = [0] * 10
    for top in topk:
        for i in range(top - 1, 10):
            top_value[i] += 1
    top_prob = np.array(top_value) / len(topk)
    avg_5 = np.array(top_prob[:5]).mean()
    result = {"top1": top_prob[0], "top3": top_prob[2], "top5": top_prob[4], "avg5": avg_5}
    return result


class MinMaxScalar:
    def __init__(self, fault_min=1e9, fault_max=0):
        self.max_ = fault_max
        self.min_ = fault_min

    def fit_transform(self, data):
        self.min_ = np.where(data.min(0) > self.min_, self.min_, data.min(0))
        self.max_ = np.where(data.max(0) > self.max_, data.max(0), self.max_)
        return (data - self.min_) / (self.max_ - self.min_ + 1e-12)

    def transform(self, data):
        return (data - self.min_) / (self.max_ - self.min_ + 1e-12)


def compute_normal_data(normal_data, args):
    logs, metrics, traces = normal_data
    if args.metric_trace:
        variables = metrics
    elif args.log_trace:
        variables = logs
    else:
        variables = np.concatenate([logs, metrics], -1)

    trace, span_time, trace_error = traces
    trace_np = np.array(list(trace.values()))
    span_time = np.array(list(span_time.values()))
    edges = np.concatenate((span_time, trace_np), -1)
    edges[edges == 0] = np.nan
    variables[variables == 0] = np.nan
    v_mean = np.nan_to_num(np.nanmean(variables, 0), 0)
    v_sigma = np.nan_to_num(np.nanstd(variables, 0), 0)
    v_min = np.nan_to_num(np.nanmin(variables, 0), 0)
    v_max = np.nan_to_num(np.nanmax(variables, 0), 0)
    e_mean = np.nan_to_num(np.nanmean(edges, 0), 0)
    e_sigma = np.nan_to_num(np.nanstd(edges, 0), 0)
    e_min = np.nan_to_num(np.nanmin(edges, 0), 0)
    e_max = np.nan_to_num(np.nanmax(edges, 0), 0)
    del normal_data, edges, variables
    return v_mean, v_sigma, v_min, v_max, e_mean, e_sigma, e_min, e_max


class ATDataset(Dataset):
    def __init__(self, data, label, args, step=1):
        self.data = data
        self.label = label
        self.args = args
        self.step = step

    def __len__(self):
        return (len(self.data) - self.args.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.data[index:index + self.args.win_size]), np.float32(
            self.label[index:index + self.args.win_size])


def train(train_data, args):
    train_label = np.zeros(len(train_data))
    model = AnomalyTransformer(win_size=args.win_size, enc_in=args.input_c, c_out=args.output_c, e_layers=3).to(
        args.device)
    train_dataset = ATDataset(train_data, train_label, args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for i, (input_data, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            input = input_data.float().to(args.device)
            output, series, prior, _ = model(input)
            rec_loss = F.mse_loss(output, input)
            rec_loss.backward()
            optimizer.step()

    return model


def test(train_data, test_data, model, args):
    # 初始化模型、数据集、异常分数评判标准
    model.eval()
    train_label = np.zeros(len(train_data))
    train_dataset = ATDataset(train_data, train_label, args)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_label = np.zeros(len(test_data))
    test_dataset = ATDataset(test_data, test_label, args, args.win_size)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.MSELoss(reduce=False)

    # 计算阈值

    attens_energy = []
    energy2 = []
    for i, (input_data, labels) in enumerate(test_data_loader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)
        loss = torch.mean(criterion(input, output), dim=-1)
        cri = loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

        loss2 = criterion(input, output)
        cri2 = loss2.detach().cpu().numpy()
        energy2.append(cri2)
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    b, w, d = energy2[0].shape
    energy2 = np.concatenate(energy2, axis=0).reshape(-1, d)
    test_energy2 = np.array(energy2)

    attens_energy = []
    energy2 = []
    for i, (input_data, labels) in enumerate(train_data_loader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)
        loss = torch.mean(criterion(input, output), dim=-1)
        cri = loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

        loss2 = criterion(input, output)
        cri2 = loss2.detach().cpu().numpy()
        energy2.append(cri2)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = attens_energy
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    # combined_energy = np.concatenate([test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - args.anormly_ratio)
    print("Threshold :", thresh)

    b, w, d = energy2[0].shape
    energy2 = np.concatenate(energy2, axis=0).reshape(-1, d)
    train_energy2 = np.array(energy2)
    combined_energy2 = np.concatenate([train_energy2, test_energy2], axis=0)
    # combined_energy2 = np.concatenate([test_energy2], axis=0)
    thresh2 = np.percentile(combined_energy2, 100 - args.anormly_ratio, axis=0)
    thresh2 = np.expand_dims(thresh2, axis=0)

    # 计算异常分数
    attens_energy = []
    energy2 = []
    for i, (input_data, labels) in enumerate(test_data_loader):
        input = input_data.float().to(args.device)
        output, series, prior, _ = model(input)
        loss = torch.mean(criterion(input, output), dim=-1)
        cri = loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

        loss2 = criterion(input, output)
        cri2 = loss2.detach().cpu().numpy()
        energy2.append(cri2)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)

    energy2 = np.concatenate(energy2, axis=0).reshape(-1, d)
    test_energy2 = np.array(energy2)
    thresh2 = np.repeat(thresh2, len(test_energy2), axis=0)

    # 根据阈值计算分数
    pred = (test_energy > thresh).astype(int)

    pred2 = (test_energy2 > thresh2).astype(int)

    return pred, test_energy, thresh, pred2, test_energy2, thresh2


def load_data(fault_data, scalar, args, time_step=None):
    logs = fault_data["logs"]
    metric = fault_data["metrics"]  # [time_step,pod_num,resource_metric_num]
    fault_span_time = fault_data["trace_span_time"]
    fault_trace = fault_data["trace"]
    fault_error = fault_data["trace_error"]
    graph_adj = np.where(fault_trace.sum(0))
    if args.metric_trace:
        variable = metric
    elif args.log_trace:
        variable = logs
    else:
        variable = np.concatenate((logs, metric), -1)

    time_step, pod_num = variable.shape[0], variable.shape[1]
    time_np = np.array(fault_span_time)
    time_np[time_np == 0] = np.nan
    time_np = np.nan_to_num(np.nanmean(time_np, 1), 0)
    rpc_np = np.array(fault_trace)
    rpc_np[rpc_np == 0] = np.nan
    rpc_np = np.nan_to_num(np.nanmean(rpc_np, 1), 0)
    edges = np.concatenate((time_np, rpc_np, fault_error), -1).reshape(time_step, pod_num, -1)
    test_data = np.concatenate([edges, variable], -1).reshape(time_step, -1)
    test_data = interpolate_zeros(test_data)
    test_data = scalar.transform(test_data)

    return test_data, graph_adj


def interpolate_zeros(data):
    data[data == 0] = np.nan
    df = pd.DataFrame(data)
    df_interpolated = df.interpolate(method='linear', limit_direction='both')
    df_interpolated.fillna(0, inplace=True)
    return df_interpolated.values


def predict_fault_point(variables, fault_error, edges, scalar):
    time_step, pod_num, _ = variables.shape
    v = scalar.variable_transform(variables)

    e = scalar.edge_transform(edges)

    def compute_dis(middle, var, eg, trace_error):
        v[v == 0] = np.nan  # 2021 AIops Dataset has part of the missing data

        v_dis = np.nan_to_num(np.nanstd(v[middle:], 0)).sum() * (time_step - middle) + np.nan_to_num(
            np.nanstd(v[:middle], 0), 0).sum() * middle
        v_distance = v_dis.sum()
        eg[eg == 0] = np.nan
        e_distance = np.nan_to_num(np.nanstd(eg[middle:], 0)).sum() * (time_step - middle) + np.nan_to_num(
            np.nanstd(eg[:middle], 0), 0).sum() * middle

        trace_error_dis = trace_error[middle:].std(0) * (time_step - middle) + middle * trace_error[:middle].std(0)
        trace_error_dis = trace_error_dis.sum()
        return v_distance + e_distance + trace_error_dis

    # predict the fault point according to the distribution difference ############
    max_distance = np.inf
    fault_point = 0
    for i in range(2, time_step - 1):
        dis = compute_dis(i, v, e, fault_error)
        if dis < max_distance:
            fault_point = i
            max_distance = dis

    print(fault_point)

    return fault_point


def anomaly_detect_3sigma(variables, edges, v_mean, v_sigma, e_mean, e_sigma, args, fault_point):
    time_step, pod_num, _ = variables.shape

    # 3sigma anomaly detect
    v_time, pod_index, v_index = np.where(variables - v_mean > v_sigma * 3)

    anomaly_index = tuple(zip(pod_index, v_index))

    e_time, call_pod, callee_pod = np.where(edges - e_mean > e_sigma * 3)
    edge = []
    for i in range(len(call_pod)):
        edge.append((call_pod[i], callee_pod[i]))

    if args.filter_anomaly:
        # filter noise according to distribution difference
        variables_before = variables[:fault_point]
        variables_before[variables_before == 0] = np.nan
        variables_before = np.nan_to_num(np.nanmean(variables_before, 0), 0)

        variables_after = variables[fault_point:]
        variables_after[variables_after == 0] = np.nan
        variables_after = np.nan_to_num(np.nanmean(variables_after, 0), 0)

        edges_before = edges[:fault_point]
        edges_before[edges_before == 0] = np.nan
        edges_before = np.nan_to_num(np.nanmean(edges_before, 0), 0)
        edges_after = edges[fault_point:]
        edges_after[edges_after == 0] = np.nan
        edges_after = np.nan_to_num(np.nanmean(edges_after, 0), 0)
        filter_pod_index, filter_v_index = np.where(
            abs(variables_after - v_mean) / (v_sigma + 1e-9) > 3 * abs(variables_before - v_mean) / (
                    v_sigma + 1e-9))

        filter_anomaly_index = tuple(zip(filter_pod_index, filter_v_index))

        index = []
        for i in range(len(anomaly_index)):
            if anomaly_index[i] in filter_anomaly_index:
                index.append(i)
        pod_index = [pod_index[i] for i in index]
        v_index = [v_index[i] for i in index]

        filter_call_pod, filter_callee_pod = np.where(
            abs(edges_after - e_mean) / (e_sigma + 1e-9) > 3 * abs(edges_before - e_mean) / (e_sigma + 1e-9))

        filter_edge = []
        for i in range(len(filter_call_pod)):
            if edges_before[filter_call_pod[i], filter_callee_pod[i]] == 0 or edges_after[
                filter_call_pod[i], filter_callee_pod[i]] == 0:

                if edges_after[filter_call_pod[i], filter_callee_pod[i]] == 0:
                    continue
                else:
                    filter_edge.append((filter_call_pod[i], filter_callee_pod[i]))
            else:
                filter_edge.append((filter_call_pod[i], filter_callee_pod[i]))

        anomaly_edge = [e for e in edge if e in filter_edge]

    else:
        anomaly_edge = edge

    edges_np = np.zeros((pod_num, pod_num))
    for i in range(len(anomaly_edge)):
        if anomaly_edge[i][1] >= pod_num:

            edges_np[anomaly_edge[i][0]][anomaly_edge[i][1] - pod_num] += 1
        else:

            edges_np[anomaly_edge[i][0]][anomaly_edge[i][1]] += 1

    ser_anomaly_count = {}
    for i in range(len(pod_index)):
        if pod_index[i] not in ser_anomaly_count.keys():
            ser_anomaly_count[pod_index[i]] = 1
        else:

            ser_anomaly_count[pod_index[i]] += 1

    return ser_anomaly_count, edges_np, pod_index, v_index


def plot():
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置随机种子以便结果可复现
    np.random.seed(0)
    for j in range(40):
        # 假设我们有5个指标的数据
        num_indicators = 50
        num_points = 60

        # 生成正常数据（0-30s）
        train_data = train_data.reshape(720, 40, -1)
        test_data = test_data.reshape(30, 40, -1)
        normal_data = train_data[:30, j, -50:]

        # 生成异常数据（30-60s），为了区分，可以增加一个偏移量
        abnormal_data = test_data[:, j, -50:]
        data = np.concatenate([normal_data, abnormal_data], 0)
        # 创建时间点
        time_point = np.linspace(0, 60, num_points)  # 正常数据的时间点

        # 创建图形
        plt.figure(figsize=(12, 8))

        # 绘制正常数据
        for i in range(num_indicators):
            plt.plot(time_point, data[:, i], marker='o')

        # 添加图例
        plt.legend(loc='upper right')

        # 添加标题和轴标签
        plt.title('Normal and Abnormal Data Over Time of {}'.format(j))
        plt.xlabel('Time (s)')
        plt.ylabel('Value')

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.show()


class LambdaRankLoss(nn.Module):
    def __init__(self, k=10):
        super(LambdaRankLoss, self).__init__()
        self.k = k  # 考虑的列表长度

    def forward(self, y_pred, y_true):
        # 计算排序后的索引
        y_pred = F.softmax(y_pred, -1)
        sorted_indices = torch.argsort(y_pred, dim=1, descending=True)
        # 获取排序后的预测得分
        sorted_scores = torch.gather(y_pred, 1, sorted_indices)
        # 获取排序后的真正标签
        sorted_labels = torch.gather(y_true, 1, sorted_indices)

        # 计算折扣累积增益（DCG）
        gains = (2 ** sorted_labels - 1) * sorted_scores
        discounts = torch.log2(torch.arange(1, self.k + 1, device=y_pred.device).float() + 1).unsqueeze(0)
        dcg = torch.sum(gains / discounts, dim=1)

        # 计算理想折扣累积增益（IDCG）
        perfect_scores, _ = torch.sort(y_true, descending=True, dim=1)
        perfect_gains = (2 ** perfect_scores - 1) * torch.arange(1, self.k + 1, device=y_true.device).float().unsqueeze(
            0)
        perfect_dcg = torch.sum(perfect_gains / discounts, dim=1)

        # 计算归一化折扣累积增益（NDCG）
        ndcg = dcg / perfect_dcg.clamp(min=1e-8)

        # 计算损失
        loss = 1 - ndcg
        return loss.mean()


#
# def one_hot_encoding(numbers, depth):
#     # numbers: 要编码的数字列表
#     # depth: 编码的深度（即二进制的位数）
#
#     one_hot_array = np.eye(depth)[numbers]
#     return one_hot_array.tolist()
def one_hot_encode(labels, num_classes):
    # 创建一个全0的矩阵，行数等于标签的数量，列数等于分类的总数
    one_hot_list = [1 if i in labels else 0 for i in range(num_classes)]
    return one_hot_list


def compute_topk(batch_scores, src_label, training):
    sorted_indices = batch_scores.argsort(-1, True)
    # 找到标签在排序后的索引中的位置
    ranking = []
    for i in range(len(sorted_indices)):
        sort_rank = sorted_indices[i]
        for j in range(len(sort_rank)):
            if sort_rank[j] in src_label[i]:
                ranking.append(j + 1)

    # ranking = torch.where(sorted_indices == torch.tensor(src_label).unsqueeze(-1))[1] + 1  # 加1是因为排名从1开始

    result=evaluate(np.array(ranking), training)
    return result


def collate(samples):
    fault_feature, graphs, label = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    fault_feature = torch.FloatTensor(np.array(fault_feature))
    labels = torch.tensor(label)
    return fault_feature, batched_graph, labels


def augment_data(fault_data):
    augmented_data = []
    for i in range(len(fault_data)):
        ori_data, ori_adj, ori_label, failure_type = fault_data[i]

        for j in range(len(fault_data)):
            extra_data, extra_adj, extra_label, _ = fault_data[j]

            new_data = np.concatenate((ori_data[:80, ori_label, :], extra_data[-60:, extra_label, :]), 0)

            augmented_data.append([new_data, ori_data, ori_adj, 1, failure_type])
        for p in range(ori_data.shape[1]):
            if ori_label == p:
                augmented_data.append([ori_data[:, p, :], ori_data, ori_adj, 1, failure_type])
            else:
                augmented_data.append([ori_data[:, p, :], ori_data, ori_adj, 0, failure_type])
    return augmented_data
