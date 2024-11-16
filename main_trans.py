import pickle
from config import *
import argparse
import numpy as np
from uda_rca_tran import root_cause
from unsupervised_rca import personalized_random_walk
from microscope import microscope
from msrank import msrank
from PDiagnose import PDiagnose
import torch
import random
import os
import json


def save(best_res, config):
    filename = f"./result/_{config.method}_{config.s_dataset}_{config.t_dataset}_bs{config.batch_size}_emd{config.embed_dim}_hid{config.hidden_dim}_beta{config.beta}_el{config.event_len}_eps{config.epsilon}.txt"
    # filename = f"./result/_{config.method}_{config.s_dataset}_{config.t_dataset}_bs{config.batch_size}_emd{config.embed_dim}_hid{config.hidden_dim}_beta{config.beta}_el{config.event_len}.txt"
    with open(filename, "w") as fw:
        json.dump(best_res, fw)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default="uda_rca",
                    choices=["uda_rca", "per_random_walk", "microscope", "msrank", "PDiagnose"], help="method choice")
parser.add_argument('-s', '--s_dataset', type=str, default="AIops2021", choices=["AIops2022", "SN", "Micross", "TT","AIops2021"],
                    help="dataset choice")
parser.add_argument('-t', '--t_dataset', type=str, default="AIops2022", choices=["AIops2022", "SN", "Micross", "TT","AIops2021"],
                    help="dataset choice")
parser.add_argument('--self', type=int, default=0)
parser.add_argument('-f', '--fault_event', type=str, default="fault_events")
parser.add_argument('--fault_vector', type=str, default="fault_vector")
parser.add_argument('--fault_data', type=str, default="fault_data.pkl")
parser.add_argument('--anomaly_thresh', type=int, default=0, help='')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--metric_trace', type=str2bool, default=False)
parser.add_argument('--log_trace', type=str2bool, default=False)
parser.add_argument('--log_metric', type=str2bool, default=False)
parser.add_argument('--use_shared_encoder', type=str2bool, default=False)
parser.add_argument('--use_embed_domain', type=str2bool, default=True)
parser.add_argument('--lr', type=float, default=1e-5, help='')
parser.add_argument('--batch_size', type=int, default=45, help='')
parser.add_argument('--pod_num', type=int, default=40, help='')
parser.add_argument('--modal_num', type=int, default=3, help='')
parser.add_argument('--ratio', type=float, default=0.9, help='')
parser.add_argument('--use_event_number', type=int, default=1, help='')
parser.add_argument('--epochs', type=int, default=0, help='')
parser.add_argument('--embed_dim', type=int, default=5, help='')
parser.add_argument('--hidden_dim', type=int, default=5, help='')
parser.add_argument('--beta', type=float, default=1, help='')
parser.add_argument('--pagerank_iter', type=float, default=100, help='')
parser.add_argument('--epsilon', type=float, default=1e-5, help='')
parser.add_argument('--event_len', type=int, default=20, help='')
parser.add_argument('--seq_len', type=int, default=6, help='')
parser.add_argument('--domin_adaptation', type=int, default=1, help='')
parser.add_argument('--only_use_gat', type=str2bool, default=False)
parser.add_argument('--use_gat_linear', type=str2bool, default=False)
args = parser.parse_args()
seed_everything(2024)
if torch.cuda.is_available():
    args.device = 'cuda' if args.device == 'cuda' else 'cpu'
if args.method == "uda_rca":
    result = root_cause(args)
elif args.method == "per_random_walk":
    personalized_random_walk(args)
elif args.method == "microscope":
    microscope(args)
elif args.method == "msrank":
    msrank(args)
elif args.method == "PDiagnose":
    result = PDiagnose(args)

save(result, args)
