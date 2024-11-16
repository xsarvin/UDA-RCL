import os
import pickle
import networkx as nx
import numpy as np
from utils_final import *
from rca_dataset import EventDataset
from model.rca_model import GAT, DomainClassifier
from config import AIops2022, AIops2021, SN, TT
import torch
import time


def val(args, t_loader, net, optimizer_net):
    net.eval()
    score_epoch = []
    label_epoch = []
    ######################test####################################
    for (l_e, m_e, t_e, l_c, m_c, t_c, tgt_adj, tgt_label) in t_loader:
        bs, pn, _ = l_e.shape
        l_e = l_e.to(args.device)
        m_e = m_e.to(args.device)
        t_e = t_e.to(args.device)
        l_c = l_c.to(args.device)
        m_c = m_c.to(args.device)
        t_c = t_c.to(args.device)
        tgt_adj = tgt_adj.to(args.device)

        # 获取源域和目标域的特征
        optimizer_net.zero_grad()

        if args.use_shared_encoder:
            result, _ = net(l_e, m_e, t_e, l_c, m_c, t_c, tgt_adj)
        else:
            result, log_r, metric_r, trace_r = net(l_e, m_e, t_e, l_c, m_c, t_c, tgt_adj)

        # 训练分类器

        label_epoch.append(tgt_label)
        score_epoch.append(result.reshape(bs, pn))
        # 训练分类器

    return compute_topk(torch.cat(score_epoch, 0), torch.cat(label_epoch, 0), training=False)


def root_cause(args):
    with open("./{}/".format(args.s_dataset) + args.fault_vector + ".pkl", "rb") as f:
        s_data = pickle.load(f)

    s_faults_data, s_event_dict = s_data

    s_num = len(s_faults_data)

    s_train_dataset = EventDataset(s_faults_data[:int(s_num * args.ratio)])
    s_test_dataset = EventDataset(s_faults_data[int(s_num * args.ratio):])

    s_train_loader = DataLoader(s_train_dataset, batch_size=args.batch_size, shuffle=True)
    s_test_loader = DataLoader(s_test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    net = GAT(args, len(s_event_dict)).to(args.device)

    domain_net = DomainClassifier(args).to(args.device)
    optimizer_net = torch.optim.Adam([{'params': net.parameters()},
                                      {'params': domain_net.parameters()}], lr=0.1)
    best_result = {}
    train_t = 0
    for epoch in range(args.epochs):
        loss_epoch = 0

        net.train()

        train_start = time.time()
        for (l_e, m_e, t_e, l_c, m_c, t_c, src_adj, src_label) in s_train_loader:
            # 获取源域和目标域的特征
            bs, pn, _ = l_e.shape
            l_e = l_e.to(args.device)
            m_e = m_e.to(args.device)
            t_e = t_e.to(args.device)
            l_c = l_c.to(args.device)
            m_c = m_c.to(args.device)
            t_c = t_c.to(args.device)
            src_adj = src_adj.to(args.device)
            optimizer_net.zero_grad()
            if args.use_shared_encoder:
                src_result, src_feature = net(l_e, m_e, t_e, l_c, m_c, t_c, src_adj)
            else:
                src_result, log_r, metric_r, trace_r = net(l_e, m_e, t_e, l_c, m_c, t_c, src_adj)

            # 训练分类器
            s_class_loss = criterion(src_result.reshape(bs, pn), src_label.to(int))

            loss = s_class_loss

            loss_epoch += loss
            loss.backward()
            optimizer_net.step()
        train_t += (time.time() - train_start)
        test_st = time.time()
        result = val(args, s_test_loader, net, optimizer_net)
        test_epoch = time.time() - test_st
        print("train_t:{} test_t:{}".format(train_t, test_epoch))
        # print(result)
        if best_result == {}:
            best_result = result
        else:
            if best_result["avg5"] < result["avg5"]:
                best_result = result
    print("top1:{} --top3:{} --top5:{} --avg5:{}".format(best_result["top1"], best_result["top3"], best_result["top5"],
                                                         best_result["avg5"]))
    # result = val(args, s_test_loader, net, optimizer_net)
    # print(result)
    return best_result
