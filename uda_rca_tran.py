import os
import pickle
import networkx as nx
import numpy as np
from utils_final import *
from rca_dataset import EventDataset
from model.rca_model import GAT, DomainClassifier
from config import AIops2022, AIops2021, SN, TT, Micross
import torch


def val(args, t_loader, net, optimizer_net):
    net.eval()
    score_epoch = []
    label_epoch = []
    ######################test####################################
    for (t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c, tgt_adj, tgt_label) in t_loader:
        bs, pn, _ = t_log_e.shape

        # 获取源域和目标域的特征
        optimizer_net.zero_grad()
        if args.use_shared_encoder and not args.use_embed_domain:
            result, _ = net(t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c, tgt_adj)
        else:
            result, _, _, _ = net(t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c, tgt_adj)
        # 训练分类器

        label_epoch.append(tgt_label)
        score_epoch.append(result.reshape(bs, pn))
        # 训练分类器

    return compute_topk(torch.cat(score_epoch, 0), torch.cat(label_epoch, 0), training=False)


def root_cause(args):
    with open("./{}/".format(args.s_dataset) + args.fault_vector + ".pkl",
              "rb") as f:
        s_data = pickle.load(f)

    with open("./{}/".format(args.t_dataset) + args.fault_vector + ".pkl",
              "rb") as f:
        t_data = pickle.load(f)

    s_faults_data, s_event_dict = s_data
    t_faults_data, t_event_dict = t_data

    s_dataset = EventDataset(s_faults_data)
    t_dataset = EventDataset(t_faults_data)

    s_loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True)
    t_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    d_criterion = nn.CrossEntropyLoss()

    net = GAT(args, event_num=len(s_event_dict)).to(args.device)
    domain_net = DomainClassifier(args).to(args.device)
    optimizer_net = torch.optim.Adam([{'params': net.parameters()},
                                      {'params': domain_net.parameters()},
                                      ], lr=0.1)
    best_result = {}
    ##################train phase####################
    for epoch in range(args.epochs):
        loss_epoch = 0
        s_class_loss_epoch = 0
        domain_loss_epoch = 0
        net.train()
        for (s_log_e, s_metric_e, s_trace_e, s_log_c, s_metric_c, s_trace_c, src_adj, src_label), (
                t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c, tgt_adj, tgt_label) in zip(s_loader,
                                                                                                           t_loader):
            # 获取源域和目标域的特征
            optimizer_net.zero_grad()

            bs, pn, _ = s_log_e.shape
            s_log_e = s_log_e.to(args.device)
            s_metric_e = s_metric_e.to(args.device)
            s_trace_e = s_trace_e.to(args.device)
            s_log_c = s_log_c.to(args.device)
            s_metric_c = s_metric_c.to(args.device)
            s_trace_c = s_trace_c.to(args.device)
            src_adj = src_adj.to(args.device)

            t_log_e = t_log_e.to(args.device)
            t_metric_e = t_metric_e.to(args.device)
            t_trace_e = t_trace_e.to(args.device)
            t_log_c = t_log_c.to(args.device)
            t_metric_c = t_metric_c.to(args.device)
            t_trace_c = t_trace_c.to(args.device)
            tgt_adj = tgt_adj.to(args.device)

            if args.use_shared_encoder and not args.use_embed_domain:
                result, src_feature = net(s_log_e, s_metric_e, s_trace_e, s_log_c, s_metric_c,
                                          s_trace_c,
                                          src_adj)
                _, tgt_feature = net(t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c,
                                     tgt_adj)

                domain_src = domain_net.fusion_forward(src_feature)
                domain_tgt = domain_net.fusion_forward(tgt_feature)

                domain_loss = d_criterion(domain_src, torch.ones(domain_src.size(0)).long()) + \
                              d_criterion(domain_tgt, torch.zeros(domain_tgt.size(0)).long())


            else:
                result, s_log_r, s_metric_r, s_trace_r = net(s_log_e, s_metric_e, s_trace_e, s_log_c, s_metric_c,
                                                             s_trace_c,
                                                             src_adj)
                _, t_log_r, t_metric_r, t_trace_r = net(t_log_e, t_metric_e, t_trace_e, t_log_c, t_metric_c, t_trace_c,
                                                        tgt_adj)

                s_log_out, s_metric_out, s_trace_out = domain_net(s_log_r, s_metric_r, s_trace_r)
                t_log_out, t_metric_out, t_trace_out = domain_net(t_log_r, t_metric_r, t_trace_r)

                log_domain_loss = d_criterion(s_log_out, torch.ones(s_log_out.size(0)).long()) + d_criterion(
                    t_log_out, torch.zeros(t_log_out.size(0)).long())

                metric_domain_loss = d_criterion(s_metric_out, torch.ones(s_metric_out.size(0)).long()) + d_criterion(
                    t_metric_out, torch.zeros(t_metric_out.size(0)).long())

                trace_domain_loss = d_criterion(s_trace_out, torch.ones(s_trace_out.size(0)).long()) + d_criterion(
                    t_trace_out, torch.zeros(t_trace_out.size(0)).long())

                domain_loss = log_domain_loss + metric_domain_loss + trace_domain_loss

            # 训练分类器
            s_class_loss = criterion(result.reshape(bs, pn), src_label.to(int))

            if args.domin_adaptation:
                loss = s_class_loss + args.beta * domain_loss
            else:
                loss = s_class_loss

            loss_epoch += loss
            domain_loss_epoch += domain_loss
            s_class_loss_epoch += s_class_loss
            loss.backward()
            optimizer_net.step()

            s_loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True)
            t_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True)

        result = val(args, t_loader, net, optimizer_net)
        # print(result)
        if best_result == {}:
            best_result = result
        else:
            if best_result["avg5"] < result["avg5"]:
                best_result = result
    result = val(args, t_loader, net, optimizer_net)

    if best_result == {}:
        best_result = result
    else:
        if best_result["avg5"] < result["avg5"]:
            best_result = result
    print("top1:{} --top3:{} --top5:{} --avg5:{}".format(best_result["top1"], best_result["top3"], best_result["top5"],
                                                         best_result["avg5"]))

    return best_result
