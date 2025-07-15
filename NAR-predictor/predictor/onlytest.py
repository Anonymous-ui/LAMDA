import os
import time
import copy
import random
import torch
import logging
import argparse
import numpy as np
from scipy.stats import stats
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from model import SRLoss
from dataset import GraphLatencyDataset, AccuracyDataset, FixedLengthBatchSampler

from transformers import get_linear_schedule_with_warmup




class Metric(object):
    def __init__(self):
        self.all = self.init_pack()
        self.plts = {}

    def init_pack(self):
        return {
            'ps' : [],
            'gs' : [],
            'cnt': 0,
            'apes': [],                                # absolute percentage error
            'errbnd_cnt': np.array([0.0, 0.0, 0.0]),   # error bound count
            'errbnd_val': np.array([0.1, 0.05, 0.01]), # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / gs[i]
            pack['errbnd_cnt'][ape <= pack['errbnd_val']] += 1
            pack['apes'].append(ape)
            pack['ps'].append(ps[i])
            pack['gs'].append(gs[i])
        pack['cnt'] += len(ps)

    def measure_pack(self, pack):
        acc = np.mean(pack['apes'])
        err = (pack['errbnd_cnt'] / pack['cnt'])
        tau = stats.kendalltau(pack['gs'], pack['ps']).correlation
        return acc, err, tau

    def update(self, ps, gs, plts=None):
        self.update_pack(ps, gs, self.all)
        if plts:
            for idx, plt in enumerate(plts):
                if plt not in self.plts:
                    self.plts[plt] = self.init_pack()
                self.update_pack([ps[idx]], [gs[idx]], self.plts[plt])

    def get(self, plt=None):
        if plt is None:
            return self.measure_pack(self.all)
        else:
            return self.measure_pack(self.plts[plt])


def test_single(self, data1, data2, n_edges, y):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    self.model.eval()

    metric = Metric()
    infer_time = 0

    with torch.no_grad():
        torch.cuda.empty_cache()

        # 开始计时
        time_i_1 = time.time()
        # 模型推理
        pred_cost = self.model(data1, data2, n_edges)
        time_i_2 = time.time()

        # 计算推理时间
        infer_time += time_i_2 - time_i_1

        # 打印并记录预测结果
        content = f"test pred_cost {pred_cost} label: {y}"
        with open("D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset5\\unseen_structure\\result\\resnet18.txt",
                  "a") as file:
            file.write(content + "\n")
        print("test pred_cost", pred_cost, "label:", y)

        # 转换结果以更新指标
        gs = pred_cost.data.cpu().numpy()[:, 0].tolist()
        ps = y.data.cpu().numpy()[:, 0].tolist()
        plts = None
        metric.update(ps, gs, plts)

        # 获取并打印指标
        acc, err, tau = metric.get()
        print("acc:", acc)

    self.logger.info(" ------------------------------------------------------------------")
    self.logger.info(" * MAPE: {:.5f}".format(acc))
    self.logger.info(" * ErrorBound: {}".format(err))
    self.logger.info(" * Kendall's Tau: {}".format(tau))
    self.logger.info(" ------------------------------------------------------------------")

    self.logger.info(" Average Latency : {:.8f} ms".format(infer_time * 1000))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed_all(time.time())

    return acc, err, tau


test_single()