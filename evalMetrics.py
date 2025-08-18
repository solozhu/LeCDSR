import math
import numpy as np

'''
Hit Rate存在一些问题，HR计算方法为每一个序列中命中次数之和除以所有序列长度之和。
由于实验设置为每个序列只有一个真实值，因此分子为命中次数之和，而分母存在问题。
分母没有求所有序列长度之和，而是使用了序列个数之和，即为测试集大小。
当前计算方法相当于计算了平均命中率

MeanReciprocalRank， MRR的计算没有逻辑问题。

NDCG：论文中公式与其他资料中公式不同，且代码中并没有体现分母中的log2，分子为1， 此处可能是因为论文中写的是log(2, base=2)==1
根据其他资料中的公式，可以认为这里将真实值的相关性定为1，其他值相关性定为0，可以推导出以下计算方法。
HGN中NDCG中IDCG认为是推荐列表全部命中, 这里认为真实值排在第一位。
'''
def HR_at_k(predicted, topk):
    # 注意这里 predicted 里面都是index，不是具体的item ID
    sum_HR = 0.0
    for i in range(len(predicted)):
        # 由于我们在候选物品中把 ground truth 放到了第一个位置上，
        # 所以其 index 为 0，所以我们只需要看top-k预测结果中是否有 0 ，有则命中
        act_set = set([0])
        pred_set = set(predicted[i][:topk])
        sum_HR += len(act_set & pred_set)
    return sum_HR / float(len(predicted))


def MRR_(predicted):
    res = 0.0
    for i in range(len(predicted)):
        pred_list = predicted[i]
        val = 0.0
        for j in range(0, len(pred_list)):
            if pred_list[j] == 0:
                val = 1.0 / (j + 1)
                break
        res += val
    return res / float(len(predicted))


def ndcg_at_k(predicted, topk):
    res = 0.0
    act_set = set([0])
    for i in range(len(predicted)):
        dcg_k = sum([int(predicted[i][j] in act_set) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k
    return res / float(len(predicted))


def ndcg_k(predicted, topk):
    """参考HGN中的计算NDCG方法
    """
    res = 0
    actual = set([0])
    for i in range(len(predicted)):
        idcg = idcg_k(topk)
        dcg_k = sum([int(predicted[i][j] in actual) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(predicted))


def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res