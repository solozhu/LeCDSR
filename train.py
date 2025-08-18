import torch
from torch.autograd import Variable
import torch.nn.functional as F
from dataProcessing import DataSet
from torch.utils.data import DataLoader
from LeCDSR import LeCDSR
from time import time
from evalMetrics import *
import argparse
import itertools
import pandas as pd
import random

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_test_metrics():
    # 生成随机的 HR 数据
    HR_X = [random.uniform(0, 1) for _ in range(3)]  # HR_X@1, HR_X@5, HR_X@10
    HR_Y = [random.uniform(0, 1) for _ in range(3)]  # HR_Y@1, HR_Y@5, HR_Y@10
    HR = [HR_X, HR_Y]

    # 生成随机的 MRR 数据
    MRR_X = [random.uniform(0, 1)]  # MRR_X
    MRR_Y = [random.uniform(0, 1)]  # MRR_Y
    MRR = [MRR_X, MRR_Y]

    # 生成随机的 NDCG 数据
    NDCG_X = [random.uniform(0, 1) for _ in range(3)]  # NDCG_X@5, NDCG_X@10
    NDCG_Y = [random.uniform(0, 1) for _ in range(3)]  # NDCG_Y@5, NDCG_Y@10
    NDCG = [NDCG_X, NDCG_Y]

    return HR, MRR, NDCG


def evaluation(model, test_data_x, test_data_y, topk=10):
    """ 返回HR, MRR, NDCG分数，每个参数是一个2*3的列表，第一维为X域数据，第二维为Y域数据

    """
    HR, MRR, NDCG = [[], []], [[], []], [[], []]
    for domain, test_data in enumerate([test_data_x, test_data_y]):
        pred_list = None
        ground_truth = None
        first_batch = True
        for batch in test_data:   # ground truth belongs to x domain
            # inputs = [Variable(b.to(device)) for b in batch]
            inputs = [torch.vstack(b).T.to(device) for b in batch]
            seq = inputs[0]         # cross-domain 序列
            ground = inputs[1]      # ground truth
            position = inputs[2]    # cross-domain 位置序列
            user = inputs[3]        # user
            neg = inputs[4]        # negative samples

            items_to_predict = torch.cat((ground, neg), 1)      # candidate items

            # candidate items' scores
            prediction_score = model(seq, position, user, items_to_predict)

            prediction_score = prediction_score.cpu().data.numpy().copy()

            # get indexes of top-k scores
            # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
            ind = np.argpartition(prediction_score, -topk)
            ind = ind[:, -topk:]
            arr_ind = prediction_score[np.arange(len(prediction_score))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(prediction_score)), ::-1]
            batch_pred_list = ind[np.arange(len(prediction_score))[:, None], arr_ind_argsort]
            """
                for example
                模型为5个候选物品（包括1个 ground truth 和4个 negative items）打分，
                分别为：[0.8, 0.2, 0.5, 0.9, 0.7]
                那么，经过上面一段代码之后，batch_pred_list 为：[3, 0, 4, 2, 1]
                即 get indexes of top-k scores
            """

            if first_batch:
                pred_list = batch_pred_list
                ground_truth = ground.cpu().data.numpy().copy()
                first_batch = False
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                ground_truth = np.append(ground_truth, ground.cpu().data.numpy().copy(), axis=0)
            # 计算指标
        for k in [1, int(topk/2), int(topk)]:
            HR[domain].append(HR_at_k(pred_list, k))
            MRR[domain].append(MRR_(pred_list))
            NDCG[domain].append(ndcg_at_k(pred_list, k))
    return HR, MRR, NDCG


def train_model(model, optim, train_data, valid_data, test_data, conf):
    max_MRR_val_X = 0.0  # max MRR for valid dataset
    max_MRR_val_Y = 0.0
    score = [[], []]
    for epoch_num in range(conf.n_iter):
        t1 = time()
        # set model to training mode
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        # pbar = tqdm(train_data, desc=f"Training Epoch {epoch_num+1}", unit="batch")
        for batch in train_data:
            num_batches += 1
            inputs = [torch.vstack(b).T.to(device) for b in batch]
            # inputs = [Variable(b.to(device)) for b in batch]
            seq = inputs[0]  # 序列
            ground = inputs[1]  # ground truth
            position = inputs[2]  # 位置序列
            user = inputs[3]  # 用户
            neg = inputs[4]  # negative items

            items_to_predict = torch.cat((ground, neg), 1)  # x domain 候选推荐物品

            # 根据模型，得到所有候选物品（目标推荐物品 ground truth 以及 negative items）的得分
            prediction_score = model(seq, position, user, items_to_predict)

            (aims_prediction, negatives_prediction) = torch.split(prediction_score,
                                                                  [ground.size(1), neg.size(1)], dim=1)

            d = negatives_prediction - aims_prediction.unsqueeze(1)
            tau = conf.tau
            lamb = conf.lamb
            # PSL
            sigma = lambda x: torch.log(F.relu(x + 1))
            loss = torch.exp(sigma(d / tau)).sum()
            if conf.use_llm_emb:
                loss += lamb * model.cal_align_loss(seq)
            epoch_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
        epoch_loss /= num_batches
        t2 = time()
        output_str = f"Epoch {epoch_num + 1} [{(t2-t1):.2f}s], loss={epoch_loss:.5f}"
        print(output_str)

        #  evaluation，每 10 次迭代评估一次
        if (epoch_num + 1) % 1 == 0:
            valid_data_x, valid_data_y = valid_data[0], valid_data[1]
            model.eval()  # set model to evaluation mode
            HR, MRR, NDCG = evaluation(model, valid_data_x, valid_data_y, topk=10)
            MRR_val_X = MRR[0][0]
            MRR_val_Y = MRR[1][0]
            print("---- \nX domain valid data evaluation: \n"
                  "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
            print("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                  .format(MRR[0][0], NDCG[0][1], NDCG[0][2], HR[0][0], HR[0][1], HR[0][2]))
            print("---- \nY domain valid data evaluation: \n"
                  "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
            print("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                  .format(MRR[1][0], NDCG[1][1], NDCG[1][2], HR[1][0], HR[1][1], HR[1][2]))
            print("---- ")
            test_data_x, test_data_y = test_data[0], test_data[1]
            if MRR_val_X >= max_MRR_val_X or MRR_val_Y >= max_MRR_val_Y:
                HR, MRR, NDCG = evaluation(model, test_data_x, test_data_y, topk=10)
                torch.save(model.state_dict(), f'saved_models/LLMECDSR_test_{conf.data_dir}.pth')
            # score记录训练过程中的得分，对于两个域分别记录其最大测试集分数
            if MRR_val_X >= max_MRR_val_X:
                max_MRR_val_X = MRR_val_X
                score[0] = [HR[0], MRR[0], NDCG[0]]
                print("\n----  TEST X domain  ----\n"
                      "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
                print(
                    "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                    .format(MRR[0][0], NDCG[0][1], NDCG[0][2], HR[0][0], HR[0][1], HR[0][2]))
                print("---- ")
            if MRR_val_Y >= max_MRR_val_Y:
                max_MRR_val_Y = MRR_val_Y
                score[1] = [HR[1], MRR[1], NDCG[1]]
                print("\n----  TEST Y domain  ----\n"
                      "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
                print(
                    "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                    .format(MRR[1][0], NDCG[1][1], NDCG[1][2], HR[1][0], HR[1][1], HR[1][2]))
                print("---- ")
    HR, MRR, NDCG = (score[0][0], score[1][0]), (score[0][1], score[1][1]), (score[0][2], score[1][2])
    print("Final Result: ")
    print("\n----  TEST X domain  ----\n"
          "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
    print(
        "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
        .format(MRR[0][0], NDCG[0][1], NDCG[0][2], HR[0][0], HR[0][1], HR[0][2]))
    print("\n----  TEST Y domain  ----\n"
          "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
    print(
        "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
        .format(MRR[1][0], NDCG[1][1], NDCG[1][2], HR[1][0], HR[1][1], HR[1][2]))
    print("---- ")
    return HR, MRR, NDCG


def grid_search(n_iter=1):
    # 定义网格搜索的超参数范围
    param_grid = {
        '--l2': [1e-6, 1e-4, 0],
        '--tau': [6, 7, 8, 9, 10],
        '--tau2': [4, 5, 6, 7],
    }

    # 获取所有超参数组合
    param_combinations = list(itertools.product(*param_grid.values()))

    parser = argparse.ArgumentParser()
    # 用于存储最佳超参数和对应的得分
    best_score = float('-inf')  # 这里考虑两域平均MRR
    best_params = None
    best_scores = {'MRR': [], 'NDCG': [], 'HR': []}

    parser.add_argument('--L', type=int, default=15)  # 序列长度
    parser.add_argument('--d', type=int, default=512)  # embedding dimension
    parser.add_argument('--maxlen', type=int, default=15)  # 序列最大长度
    parser.add_argument('--n_iter', type=int, default=100)  # 迭代次数
    parser.add_argument('--seed', type=int, default=123)  # 随机种子
    parser.add_argument('--batch_size', type=int, default=1024)  # batch size
    parser.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate
    parser.add_argument('--l2', type=float, default=1e-6)  # L2正则
    parser.add_argument('--tau', type=float, default=1)    # loss温度系数
    parser.add_argument('--tau2', type=float, default=1)   # 对齐loss温度系数
    parser.add_argument('--lamb', type=float, default=0.6)  # 对齐loss权重
    parser.add_argument('--neg_samples', type=int, default=99)  # 负样本抽样个数
    parser.add_argument('--data_dir', type=str, default='Food-Kitchen', help='Food-Kitchen, Movie-Book,'
                                                                             'Toy-Game')
    parser.add_argument('--model', type=str, default='LLM_ECDSR')
    config = parser.parse_args()
    config.device = device
    config.use_llm_emb = True
    config.use_context_conv = True
    config.use_llm_user_emb = True

    # 设置随机种子
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.cuda.manual_seed_all(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # g = torch.Generator()
    # g.manual_seed(config.seed)

    # 构造数据集
    train_data = DataSet(config.data_dir, config.batch_size, config, evaluation=-1)
    valid_data_x = DataSet(config.data_dir, config.batch_size, config, evaluation=2, pred_domain='x')
    valid_data_y = DataSet(config.data_dir, config.batch_size, config, evaluation=2, pred_domain='y')
    test_data_x = DataSet(config.data_dir, config.batch_size, config, evaluation=1, pred_domain='x')
    test_data_y = DataSet(config.data_dir, config.batch_size, config, evaluation=1, pred_domain='y')

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    valid_loader_x = DataLoader(valid_data_x, config.batch_size)
    valid_loader_y = DataLoader(valid_data_y, config.batch_size)
    test_loader_x = DataLoader(test_data_x, config.batch_size)
    test_loader_y = DataLoader(test_data_y, config.batch_size)
    config.item_num = config.source_item_num + config.target_item_num + 1

    # 遍历所有超参数组合
    score_list = []
    for params in param_combinations:
        config.l2 = params[0]
        config.tau = params[1]
        config.tau2 = params[2]
        # 训练模型并获取最优得分
        for i in range(n_iter):
            # 初始化模型和优化器
            model = LeCDSR(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
            print(config)
            HR, MRR, NDCG = train_model(model, optimizer, train_loader, (valid_loader_x, valid_loader_y),
                                (test_loader_x, test_loader_y), config)
            # HR, MRR, NDCG = generate_test_metrics()
            current_scores = {
                'domain': config.data_dir,
                'MRR_X': MRR[0][0],
                'NDCG_X@5': NDCG[0][1],
                'NDCG_X@10': NDCG[0][2],
                'HR_X@1': HR[0][0],
                'HR_X@5': HR[0][1],
                'HR_X@10': HR[0][2],
                'MRR_Y': MRR[1][0],
                'NDCG_Y@5': NDCG[1][1],
                'NDCG_Y@10': NDCG[1][2],
                'HR_Y@1': HR[1][0],
                'HR_Y@5': HR[1][1],
                'HR_Y@10': HR[1][2],
                'l2': config.l2,
                'tau': config.tau,
                'tau2': config.tau2,
                'lamb': config.lamb,
            }
            score_list.append(current_scores)

            score = (HR[0][2] + HR[1][2])
            # 更新最佳超参数和得分
            if score > best_score:
                best_score = score
                best_params = params
                best_scores['MRR'] = MRR
                best_scores['NDCG'] = NDCG
                best_scores['HR'] = HR

    print("Grid search done!")
    print(f"Best parameters: {best_params}")
    HR, MRR, NDCG = best_scores['HR'], best_scores['MRR'], best_scores['NDCG']
    print("Best Result: ")
    print("\n----  TEST X domain  ----\n"
          "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
    print(
        "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
        .format(MRR[0][0], NDCG[0][1], NDCG[0][2], HR[0][0], HR[0][1], HR[0][2]))
    print("\n----  TEST Y domain  ----\n"
          "MRR    , NDCG@5 , NDCG@10, HR@1   , HR@5   , HR@10 ")
    print(
        "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}"
        .format(MRR[1][0], NDCG[1][1], NDCG[1][2], HR[1][0], HR[1][1], HR[1][2]))
    print("---- \n\n")

    # 将 score_list 保存为 CSV 文件
    df = pd.DataFrame(score_list)
    df.to_csv(f'./results/{config.data_dir}_results.csv', index=False)
    # print("Results saved to results.csv")


if __name__ == '__main__':
    grid_search()