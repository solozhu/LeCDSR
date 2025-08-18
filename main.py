import argparse
from train import *
import datetime
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=15)  # 序列长度
    parser.add_argument('--d', type=int, default=512)  # embedding dimension
    parser.add_argument('--maxlen', type=int, default=15)  # 序列最大长度
    parser.add_argument('--n_iter', type=int, default=10)  # 迭代次数
    parser.add_argument('--seed', type=int, default=123)  # 随机种子
    parser.add_argument('--batch_size', type=int, default=1024)  # batch size
    parser.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate
    parser.add_argument('--l2', type=float, default=1e-6)  # L2正则
    parser.add_argument('--tau', type=float, default=3)    # loss温度系数
    parser.add_argument('--tau2', type=float, default=3)   # 对齐loss温度系数
    parser.add_argument('--lamb', type=float, default=0.3)  # 对齐loss权重
    parser.add_argument('--neg_samples', type=int, default=99)  # 负样本抽样个数
    parser.add_argument('--data_dir', type=str, default='Toy-Game', help='Food-Kitchen, Movie-Book,'
                                                                                        'Toy-Game')
    parser.add_argument('--model', type=str, default='LLM_ECDSR')
    config = parser.parse_args()
    config.device = device
    config.use_llm_emb = True
    config.use_llm_user_emb = True
    config.use_context_conv = True

    # 构造用于训练、验证、测试的数据格式
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

    print("Data loading done!")

    # 模型初始化
    model = LeCDSR(config)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    print("recommendation: -------------------------------------------------------------------")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(config)
    print(device)

    # 训练
    train_model(model, optimizer, train_loader, (valid_loader_x, valid_loader_y),
                (test_loader_x, test_loader_y), config)


if __name__ == '__main__':
    main()
