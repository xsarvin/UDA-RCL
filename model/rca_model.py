import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=-1)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    """GAT模型"""

    def __init__(self, args, event_num):
        super(GAT, self).__init__()
        self.embedding = nn.Embedding(event_num, args.embed_dim)
        self.log_encoder = nn.Linear(args.seq_len * (args.embed_dim + args.use_event_number), args.hidden_dim)
        self.metric_encoder = nn.Linear(args.seq_len * (args.embed_dim + args.use_event_number), args.hidden_dim)
        self.trace_encoder = nn.Linear(args.seq_len * (args.embed_dim + args.use_event_number), args.hidden_dim)
        self.shared_encoder = nn.Linear(args.modal_num * args.seq_len * (args.embed_dim + args.use_event_number),
                                        3 * args.hidden_dim)

        self.decoder = nn.Linear(args.modal_num * args.hidden_dim, 1)
        self.max_event_len = args.event_len
        self.embed_dim = args.embed_dim
        self.use_event_number = args.use_event_number
        self.gat1 = GraphAttentionLayer(args.modal_num * args.hidden_dim, 1, 0, 0.2)
        self.gat2 = GraphAttentionLayer(args.modal_num * args.hidden_dim, args.modal_num * args.hidden_dim, 0, 0.2)
        self.linear1 = nn.Linear(args.modal_num * args.hidden_dim * args.pod_num, args.pod_num)
        self.linear2 = nn.Linear(args.pod_num, args.pod_num)
        self.linear3 = nn.Linear(1, 1)
        self.use_shared_encoder = args.use_shared_encoder
        self.only_use_gat = args.only_use_gat
        self.use_gat_linear = args.use_gat_linear
        self.use_embed_domain = args.use_embed_domain
        self.iter = args.pagerank_iter
        self.alpha = nn.Parameter(torch.empty(1))
        self.eps=args.epsilon
        torch.nn.init.uniform_(self.alpha, a=0, b=1)

    def forward(self, l_e, m_e, t_e, l_c, m_c, t_c, adj):
        # x.shape: batch_size pod_num seq_len event_len embed_dim
        bs, pn, _ = l_e.shape

        log_r = self.embedding(l_e)
        metric_r = self.embedding(m_e)
        trace_r = self.embedding(t_e)
        if self.use_event_number:
            log_r = torch.cat([log_r, (l_c / self.max_event_len).unsqueeze(-1)], -1)
            metric_r = torch.cat([metric_r, (m_c / self.max_event_len).unsqueeze(-1)], -1)
            trace_r = torch.cat([trace_r, (t_c / self.max_event_len).unsqueeze(-1)], -1)

        if self.use_shared_encoder:
            fusion_r = torch.cat([log_r, metric_r, trace_r], -1)
            fusion_r = self.shared_encoder(fusion_r.reshape(bs, pn, -1))
        else:
            log_encoded = self.log_encoder(log_r.reshape(bs, pn, -1))
            metric_encoded = self.metric_encoder(metric_r.reshape(bs, pn, -1))
            trace_encoded = self.trace_encoder(trace_r.reshape(bs, pn, -1))
            fusion_r = torch.cat([log_encoded, metric_encoded, trace_encoded], -1)

        if self.only_use_gat:
            x = self.gat1(fusion_r, adj)
            # x = out.max(-1)[0]
        elif self.use_gat_linear:
            out = self.gat2(fusion_r, adj)
            x = fusion_r + out
            # x = x.max(-1)[0]
            x = self.linear1(x.reshape(bs, -1))
        else:
            x = self.decoder(fusion_r)

            x_min = x.min(1)[0].unsqueeze(1).repeat(1, pn, 1)
            x_max = x.max(1)[0].unsqueeze(1).repeat(1, pn, 1)
            x = (x - x_min) / (x_max - x_min + 1e-5)
            x = self.pagerank(x, adj,self.eps)
        if self.use_shared_encoder and not self.use_embed_domain:
            return x, fusion_r
        elif not self.use_shared_encoder and not self.use_embed_domain:
            return x, log_encoded, metric_encoded, trace_encoded
        elif self.use_embed_domain:
            return x, log_r, metric_r, trace_r

    def pagerank(self, x, adj, epsilon=10e-5):
        """
        可微分的PageRank算法。

        参数:
        - x: 初始PageRank向量，形状为 (num_nodes,)
        - adj: 邻接矩阵，形状为 (num_nodes, num_nodes)
        - max_iter: 最大迭代次数
        - alpha: 阻尼系数
        - epsilon: 收敛阈值

        返回:
        - pagerank: 最终的PageRank向量
        """
        alpha = F.sigmoid(self.alpha)
        num_nodes = x.shape[1]
        # 初始化PageRank向量
        # pagerank = x.clone().detach().requires_grad_(True)
        pagerank = x
        # 迭代计算PageRank
        for _ in range(self.iter):
            # 计算转移概率矩阵
            transition_matrix = adj / (adj.sum(dim=-1, keepdim=True) + 1e-5)
            # 处理没有出链的节点（Dead Ends）
            transition_matrix[torch.isnan(transition_matrix)] = 1 / num_nodes
            transition_matrix[torch.isinf(transition_matrix)] = 1 / num_nodes

            # 计算PageRank更新
            # new_pagerank = alpha * torch.matmul(transition_matrix, pagerank) + (1 - alpha) / num_nodes
            new_pagerank = alpha * torch.matmul(transition_matrix, pagerank) + (1 - alpha) * x
            # 检查收敛性
            if torch.norm(new_pagerank - pagerank) < epsilon:
                break
            pagerank = new_pagerank
        return pagerank


class GRL(Function):
    @staticmethod
    def forward(ctx, i):
        # ctx.save_for_backward(i)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors
        return grad_output * -1


class DomainClassifier(nn.Module):
    def __init__(self, args):
        super(DomainClassifier, self).__init__()
        self.log_mlp1 = nn.Linear(args.hidden_dim, 2)
        self.metric_mlp1 = nn.Linear(args.hidden_dim, 2)
        self.trace_mlp1 = nn.Linear(args.hidden_dim, 2)
        self.log_mlp2 = nn.Linear((args.embed_dim + args.use_event_number) * args.seq_len, 2)
        self.metric_mlp2 = nn.Linear((args.embed_dim + args.use_event_number) * args.seq_len, 2)
        self.trace_mlp2 = nn.Linear((args.embed_dim + args.use_event_number) * args.seq_len, 2)
        self.mlp = nn.Linear(3 * args.hidden_dim, 2)
        self.use_embed_domain = args.use_embed_domain

    def fusion_forward(self, x, constant=1):
        x = GRL.apply(x)

        x = x.mean(1)

        x = self.mlp(x)

        return x

    def forward(self, log, metric, trace, constant=1):
        bn = log.shape[0]

        log = GRL.apply(log)
        metric = GRL.apply(metric)
        trace = GRL.apply(trace)

        log = log.mean(1)
        metric = metric.mean(1)
        trace = trace.mean(1)
        if self.use_embed_domain:
            log_out = self.log_mlp2(log.reshape(bn, -1))
            metric_out = self.metric_mlp2(metric.reshape(bn, -1))
            trace_out = self.trace_mlp2(trace.reshape(bn, -1))
        else:
            log_out = self.log_mlp1(log)
            metric_out = self.metric_mlp1(metric)
            trace_out = self.trace_mlp1(trace)

        return log_out, metric_out, trace_out
