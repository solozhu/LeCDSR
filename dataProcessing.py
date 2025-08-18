"""
    Data loader 
"""

import random
import torch
import numpy as np
import codecs
import pandas as pd
from torch.utils.data import Dataset


class DataSet(Dataset):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, domains, batch_size, opt, evaluation, pred_domain='x'):
        # 初始化一系列参数
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.domains = domains
        self.pred_domain = pred_domain
        if opt.model == None:
            opt.model = 'other'

        if domains == "Toy-Game":  # 这里是因为所构造的数据集使用了CSV格式保存，与其他数据及不同，因此直接将计算结果赋值
            opt.source_item_num = 86491
            opt.target_item_num = 22832
            opt.user_num = 15574
        else:
            # ************* item_id *****************
            opt.source_item_num = self.read_item("./dataset/" + domains + "/Alist.txt")
            opt.target_item_num = self.read_item("./dataset/" + domains + "/Blist.txt")

            # ************* user id *****************
            opt.user_num = self.read_user("../../dataset/" + domains + "/userlist.txt")

        # ************* sequential data *****************
        train_data = "./dataset/" + domains + "/traindata.txt"
        valid_data = "./dataset/" + domains + "/validdata.txt"
        test_data = "./dataset/" + domains + "/testdata.txt"

        if opt.model == 'other':
            if evaluation == -1:  # 构造训练集
                self.train_data, self.train_user = self.read_train_data(train_data)
                data = self.preprocess()
            elif evaluation == 2:  # 构造验证集
                self.test_data, self.test_user = self.read_test_data(valid_data)
                data = self.preprocess_for_predict()
            else:  # 构造测试集
                self.test_data, self.test_user = self.read_test_data(test_data)
                data = self.preprocess_for_predict()
        elif opt.model == 'SASRec':
            if evaluation == -1:  # 构造训练集
                self.train_data, self.train_user = self.read_train_data(train_data)
                data = self.preprocess_for_SASRec()
            elif evaluation == 2:  # 构造验证集
                self.test_data, self.test_user = self.read_test_data(valid_data)
                data = self.preprocess_for_SASRec_precidt()
            else:  # 构造测试集
                self.test_data, self.test_user = self.read_test_data(test_data)
                data = self.preprocess_for_SASRec_precidt()
        elif opt.model == 'LLM_ECDSR':
            if evaluation == -1:  # 构造训练集
                self.train_data, self.train_user = self.read_train_data(train_data)
                data = self.preprocess_for_llmecdsr()
            elif evaluation == 2:  # 构造验证集
                self.test_data, self.test_user = self.read_test_data(valid_data)
                data = self.preprocess_for_predict()
            else:  # 构造测试集
                self.test_data, self.test_user = self.read_test_data(test_data)
                data = self.preprocess_for_predict()
        self.data = data

    def read_item(self, fname):
        """
            统计 item 个数
        """
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_user(self, fname):
        """
            统计 user 个数
        """
        user_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                user_number += 1
        return user_number

    def read_train_data(self, train_file):
        """
            读取训练集
        """
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")
                user.append(int(line[0]))

                line = line[2:]  # 交互的一系列物品
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=lambda x: x[1])  # 按照时间顺序排列

                seq = []
                for r in res:
                    seq.append(r[0])  # 只保留 item id
                train_data.append(seq)

        return train_data, user

    def read_test_data(self, test_file):
        """
            读取验证or测试集
        """
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            user = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")
                user.append(int(line[0]))
                line = line[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=lambda x: x[1])

                seq = []
                for r in res:
                    seq.append(r[0])

                # denoted the corresponding validation/test entry
                test_data.append(seq)
        return test_data, user

    def preprocess_for_predict(self):

        """ 与 preprocess 类似"""

        if "Enter" in self.domains:
            max_len = 30
            self.opt.maxlen = 30
            self.opt.L = 30
        else:
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15

        padding = self.opt.source_item_num + self.opt.target_item_num
        processed = []
        for d, user in zip(self.test_data, self.test_user):  # the pad is needed! but to be careful.
            seq = d[:-1]
            gt = d[-1]
            # 识别gt所属的域，若与预测域不同，则该条数据不用于预测域数据集的构建
            if gt >= self.opt.source_item_num and self.pred_domain == 'x':
                continue
            if gt < self.opt.source_item_num and self.pred_domain == 'y':
                continue
            position = [0] * max_len

            if len(seq) < max_len:
                seq = [padding] * (max_len - len(seq)) + seq

            index_x = 0
            index_y = 0
            # 为了标识两域的不同位置编码，负数表示Y域
            for id in range(1, max_len):
                # position[-id] = id
                if seq[-id] == padding:
                    position[-id] = 0
                elif seq[-id] >= self.opt.source_item_num:  # Y域
                    index_y += 1
                    position[-id] = index_y
                else:
                    index_x += 1
                    position[-id] = index_x

            # 验证集/测试集 随机抽取负样本999个
            negative_sample = []
            for i in range(999):
                while True:
                    if self.pred_domain == 'y':  # in Y domain, the validation/test negative samples
                        sample = random.randint(self.opt.source_item_num, padding - 1)
                        if sample != gt:
                            negative_sample.append(sample)
                            break
                    else:  # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt.source_item_num - 1)
                        if sample != gt:
                            negative_sample.append(sample)
                            break

            # seq: test data cross
            # [gt]: ground truth
            # user: the user corresponding to the test data
            # negative_sample: negative sample

            processed.append([seq, [gt], position, [user], negative_sample])

        return processed

    def preprocess(self):
        """ 构造成训练需要的格式 """
        if "Enter" in self.domains:  # Entertainment-Education 跨域场景序列长度为30
            max_len = 30
            self.opt.maxlen = 30
            self.opt.L = 30
        else:  # Food-Kitchen、Movie-Book 场景序列长度为15
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15

        processed = []
        padding = self.opt.source_item_num + self.opt.target_item_num
        for d, user in zip(self.train_data, self.train_user):  # the pad is needed! but to be careful.
            g = d[-1]  # ground truth，最新的交互物品即为需要预测的物品
            seq = d[:-1]  # delete the ground truth，去除 ground truth 后的物品序列（跨域序列），跨域 train data
            # 位置序列初始化
            position = [0] * max_len

            # 序列长度不足 max_len 的，用固定值填充补全序列长度
            if len(seq) < max_len:
                seq = [padding] * (max_len - len(seq)) + seq

            index_x = 0
            index_y = 0
            # 为了标识两域的不同位置编码，负数表示Y域
            for id in range(1, max_len):
                # position[-id] = id
                if seq[-id] == padding:
                    position[-id] = 0
                elif seq[-id] >= self.opt.source_item_num:  # Y域
                    index_y += 1
                    position[-id] = index_y
                else:
                    index_x += 1
                    position[-id] = index_x

            # 随机抽取负样本，这些 negative sample 与 ground truth 一起，模型为这些 item 打分做 top k 推荐
            negative_sample = []
            for i in range(self.opt.neg_samples):
                while True:
                    if g >= self.opt.source_item_num:  # ground truth in Y domain, the validation/test negative samples
                        sample = random.randint(self.opt.source_item_num, padding - 1)
                        if sample != g:
                            negative_sample.append(sample)
                            break
                    else:  # ground truth in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt.source_item_num - 1)
                        if sample != g:
                            negative_sample.append(sample)
                            break

            # seq: train data cross domain
            # [g]: ground truth for cross-domain
            # [user]: the user corresponding to the train data
            # negative_sample: negative samples

            processed.append([seq, [g], position, [user], negative_sample])

        return processed

    def preprocess_for_llmecdsr(self):
        """ 构造成训练需要的格式 """
        if "Enter" in self.domains:  # Entertainment-Education 跨域场景序列长度为30
            max_len = 30
            self.opt.maxlen = 30
            self.opt.L = 30
        else:  # Food-Kitchen、Movie-Book 场景序列长度为15
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15

        processed = []
        padding = self.opt.source_item_num + self.opt.target_item_num
        for d, user in zip(self.train_data, self.train_user):  # the pad is needed! but to be careful.
            while len(d) >= 3:
                g = d[-1]  # ground truth，最新的交互物品即为需要预测的物品
                seq = d[:-1]  # delete the ground truth，去除 ground truth 后的物品序列（跨域序列），跨域 train data
                # 位置序列初始化
                position = [0] * max_len

                # 序列长度不足 max_len 的，用固定值填充补全序列长度
                if len(seq) < max_len:
                    seq = [padding] * (max_len - len(seq)) + seq

                index_x = 0
                index_y = 0
                # 为了标识两域的不同位置编码，负数表示Y域
                for id in range(1, max_len):
                    # position[-id] = id
                    if seq[-id] == padding:
                        position[-id] = 0
                    elif seq[-id] >= self.opt.source_item_num:  # Y域
                        index_y += 1
                        position[-id] = index_y
                    else:
                        index_x += 1
                        position[-id] = index_x

                # 随机抽取负样本，这些 negative sample 与 ground truth 一起，模型为这些 item 打分做 top k 推荐
                negative_sample = []
                for i in range(self.opt.neg_samples):
                    while True:
                        if g >= self.opt.source_item_num:  # ground truth in Y domain, the validation/test negative samples
                            sample = random.randint(self.opt.source_item_num, padding - 1)
                            if sample != g:
                                negative_sample.append(sample)
                                break
                        else:  # ground truth in X domain, the validation/test negative samples
                            sample = random.randint(0, self.opt.source_item_num - 1)
                            if sample != g:
                                negative_sample.append(sample)
                                break

                # seq: train data cross domain
                # [g]: ground truth for cross-domain
                # [user]: the user corresponding to the train data
                # negative_sample: negative samples

                processed.append([seq, [g], position, [user], negative_sample])
                d = d[:-1]

        return processed

    def preprocess_for_SASRec(self):
        """用于为SASRec系列的模型生成数据集
            格式为：
            u: [batch_size]
            seq: [batch_size, maxlen]
            pos: [batch_size, maxlen]
            neg: [batch_size, maxlen]
            其中正例表示交互序列每一个位置的ID及其之前ID(子序列)所对应的正例，负例同样
            模型采用BCE,因此一个正例对应一个负例
        """
        max_len = 15
        self.opt.maxlen = 15
        self.opt.L = 15

        processed = []
        padding = self.opt.source_item_num + self.opt.target_item_num
        for d, user in zip(self.train_data, self.train_user):
            seq = d[:-1]  # 去除最后一个item作为预测目标
            pos = d[1:]   # 正例为序列中每个位置的下一个item
            neg = []      # 负例列表
            
            # 为每个位置生成负例
            for i in range(len(pos)):
                while True:
                    if pos[i] >= self.opt.source_item_num:  # Y域
                        sample = random.randint(self.opt.source_item_num, padding - 1)
                    else:  # X域
                        sample = random.randint(0, self.opt.source_item_num - 1)
                    if sample != pos[i] and sample not in seq[:i+1]:  # 确保负例不在序列中
                        neg.append(sample)
                        break
            
            # 序列长度不足时进行padding
            if len(seq) < max_len:
                seq = [padding] * (max_len - len(seq)) + seq
            if len(pos) < max_len:
                pos = [padding] * (max_len - len(pos)) + pos
            if len(neg) < max_len:
                neg = [padding] * (max_len - len(neg)) + neg
            
            processed.append([seq, pos, [user], neg])
        return processed

    def preprocess_for_SASRec_precidt(self):
        """用于为SASRec系列的模型生成数据集
            格式为：
            u: [batch_size]
            seq: [batch_size, maxlen]
            pos: [batch_size, maxlen]
            neg: [batch_size, maxlen]
            其中正例表示交互序列每一个位置的ID及其之前ID(子序列)所对应的正例，负例同样
            模型采用BCE,因此一个正例对应一个负例
        """
        max_len = 15
        self.opt.maxlen = 15
        self.opt.L = 15

        processed = []
        padding = self.opt.source_item_num + self.opt.target_item_num
        for d, user in zip(self.test_data, self.test_user):  # the pad is needed! but to be careful.
            pos = d[-1]
            seq = d[:-1]
            if pos >= self.opt.source_item_num and self.pred_domain == 'x':
                continue
            if pos < self.opt.source_item_num and self.pred_domain == 'y':
                continue
            if len(seq) < max_len:
                seq = [padding] * (max_len - len(seq)) + seq

            # 随机抽取负样本，这些 negative sample 与 ground truth 一起，模型为这些 item 打分做 top k 推荐
            negative_sample = []
            for i in range(999):
                while True:
                    if pos >= self.opt.source_item_num:  # ground truth in Y domain, the validation/test negative samples
                        sample = random.randint(self.opt.source_item_num, padding - 1)
                        if sample not in d:
                            negative_sample.append(sample)
                            break
                    else:  # ground truth in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt.source_item_num - 1)
                        if sample not in d:
                            negative_sample.append(sample)
                            break

            # seq: train data cross domain
            # [pos]: ground truth for cross-domain
            # [user]: the user corresponding to the train data
            # negative_sample: negative samples

            processed.append([seq, [pos], [user], negative_sample])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


def read_train_data(train_file):
    """
        读取训练集
    """

    def takeSecond(elem):
        return elem[1]

    with codecs.open(train_file, "r", encoding="utf-8") as infile:
        train_data = []
        user = []
        for id, line in enumerate(infile):
            res = []
            line = line.strip().split("\t")
            user.append(int(line[0]))

            line = line[2:]  # 交互的一系列物品
            for w in line:
                w = w.split("|")
                res.append((int(w[0]), int(w[1])))
            res.sort(key=takeSecond)  # 按照时间顺序排列

            res_2 = []
            for r in res:
                res_2.append(r[0])  # 只保留 item id
            train_data.append(res_2)

    return train_data, user



if __name__ == "__main__":
    domain = ['Food-Kitchen', 'Movie-Book', 'Toy-Game']
    # for domain_name in domain:
    #     data_spilt(domain=domain_name, remain_domain="A")
    #     data_spilt(domain=domain_name, remain_domain="B")
