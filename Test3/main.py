import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math  # 确保导入 math 模块

from config import Config
from models import CP, AggregationModel
from utils import accuracy, EarlyStopping
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--mm', type=int, default=0)
parser.add_argument('--dd', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuid)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')

if __name__ == "__main__":
    Models = ['CP']
    Datasets = ['Abilene']
    config = Config('./data/' + Datasets[args.dd] + '.ini')
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_ratio = 0.2
    checkpoint_dir = ('./results/checkpoint/{}_{}_{}.pt').format(Models[args.mm], Datasets[args.dd], train_ratio)

    # 加载采样后的数据
    tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals = config.Sampling(train_ratio)

    # 初始化4个CP模型（对应四个标签）
    cp_models = [CP(config.num_dim, config.num_emb).cuda() for _ in range(4)]
    optimizers = [optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) for model in cp_models]
    criterion = nn.MSELoss()

    # 为每个标签创建独立的 EarlyStopping 对象
    early_stopping = [EarlyStopping(patience=args.patience, verbose=True, save_path=f'{checkpoint_dir}_label{label}.pt') for label in range(4)]
    stop_flags = [False] * 4  # 初始时，所有模型的 stop_flags 都是 False

    # 在 for epoch 循环前初始化最佳分数
    for label in range(4):
        if early_stopping[label].best_score is None:
            early_stopping[label].best_score = float('inf')  # 初始化为无穷大

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')

        for label in range(4):
            if stop_flags[label]:
                continue

            cp_models[label].train()
            train_loss = 0

            if label < 3:  # 标签模型（CP1, CP2, CP3）
                selected_data = tr_idxs[tr_idxs[:, 4] == label]
                selected_vals = tr_vals[tr_idxs[:, 4] == label]
            else:  # CP4 使用全数据
                selected_data = tr_idxs
                selected_vals = tr_vals

            if len(selected_data) == 0:
                continue

            # 随机打乱数据索引
            random_perm_idx = np.random.permutation(len(selected_data))
            selected_data = selected_data[random_perm_idx]
            selected_vals = selected_vals[random_perm_idx]

            num_batches = int(math.ceil(float(len(selected_data)) / float(config.batch_size)))
            for n in range(num_batches):
                batch_data = selected_data[n * config.batch_size : (n + 1) * config.batch_size]
                batch_vals = selected_vals[n * config.batch_size : (n + 1) * config.batch_size]

                i = batch_data[:, 0]
                j = batch_data[:, 1]
                k = batch_data[:, 2]

                optimizers[label].zero_grad()
                xijk = cp_models[label](i, j, k)[1] 
                loss = criterion(xijk, batch_vals)
                loss.backward()
                optimizers[label].step()

                train_loss += loss

            train_loss /= num_batches
            print(f"Label {label} - Train Loss: {train_loss}")

            # 验证模型
            cp_models[label].eval()
            valid_loss = 0
            if label < 3:
                selected_data = va_idxs[va_idxs[:, 4] == label]
                selected_vals = va_vals[va_idxs[:, 4] == label]
            else:  # CP4 使用全验证集
                selected_data = va_idxs
                selected_vals = va_vals

            if len(selected_data) > 0:
                i = selected_data[:, 0]
                j = selected_data[:, 1]
                k = selected_data[:, 2]
                with torch.no_grad():
                    xijk = cp_models[label](i, j, k)[1] 
                    valid_loss = criterion(xijk, selected_vals)

            # 保存当前最佳模型参数
            if valid_loss < early_stopping[label].best_score:
                early_stopping[label].best_score = valid_loss
                torch.save(cp_models[label].state_dict(), f'{checkpoint_dir}_label{label}.pt')

            early_stopping[label](valid_loss, cp_models[label])
            if early_stopping[label].early_stop:
                print(f"Early stopping for label {label}")
                stop_flags[label] = True

        if all(stop_flags):
            print("All models have early stopped. Ending training.")
            break

    # 聚合前的测试前加载早停保存的最佳模型参数
    for label in range(4):
        checkpoint_path = f'{checkpoint_dir}_label{label}.pt'
        if os.path.exists(checkpoint_path):
            cp_models[label].load_state_dict(torch.load(checkpoint_path))

    cp_models = [model.eval() for model in cp_models]  # 确保所有模型都处于评估模式


    # 初始化聚合模型
    aggregation_model = AggregationModel(config.num_emb).cuda()

    # 设置聚合模型的优化器、损失函数和早停机制
    optimizer = optim.AdamW(aggregation_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./results/aggregation_model_best.pt')

    # 初始化最佳分数
    early_stopping.best_score = float('inf')  # 无穷大，表示初始时验证损失无上限

    # 聚合模型训练循环
    for epoch in range(config.epochs):
        aggregation_model.train()
        train_loss = 0

        # 随机打乱数据索引
        random_perm_idx = np.random.permutation(len(tr_idxs))
        tr_idxs = tr_idxs[random_perm_idx]
        tr_vals = tr_vals[random_perm_idx]

        num_batches = int(math.ceil(float(len(tr_idxs)) / float(config.batch_size)))

        for n in range(num_batches):
            batch_data = tr_idxs[n * config.batch_size : (n + 1) * config.batch_size]
            batch_vals = tr_vals[n * config.batch_size : (n + 1) * config.batch_size]

            i = batch_data[:, 0]
            j = batch_data[:, 1]
            k = batch_data[:, 2]

            # 获取每个CP模型的预测值
            cp_embed = [cp_models[label](i, j, k)[0] for label in range(4)]  # 只取 cp_embed 部分

            # 聚合模型的预测
            aggregated_output = aggregation_model(cp_embed[0], cp_embed[1], cp_embed[2], cp_embed[3])

            # 计算损失并优化聚合模型
            loss = criterion(aggregated_output, batch_vals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= num_batches
        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss}")

        # 验证聚合模型
        aggregation_model.eval()
        valid_loss = 0
        with torch.no_grad():
            num_batches_val = math.ceil(len(va_idxs) / config.batch_size_eval)
            for n in range(num_batches_val):
                batch_data = va_idxs[n * config.batch_size_eval : (n + 1) * config.batch_size_eval]
                batch_vals = va_vals[n * config.batch_size_eval : (n + 1) * config.batch_size_eval]

                i = batch_data[:, 0]
                j = batch_data[:, 1]
                k = batch_data[:, 2]

                # 获取每个CP模型的预测值
                cp_embed = [cp_models[label](i, j, k)[0] for label in range(4)]

                # 聚合模型的验证预测
                aggregated_output = aggregation_model(cp_embed[0], cp_embed[1], cp_embed[2], cp_embed[3])
                loss = criterion(aggregated_output, batch_vals)
                valid_loss += loss

        # 如果验证损失更好，保存当前模型
        if valid_loss < early_stopping.best_score:
            early_stopping.best_score = valid_loss
            # 保存聚合模型
            torch.save(aggregation_model.state_dict(), './results/aggregation_model_best.pt')

        # 检查是否需要早停
        early_stopping(valid_loss, aggregation_model)
        if early_stopping.early_stop:
            print("Early stopping for aggregation model.")
            break

    # 聚合模型测试
    print("Testing after aggregation...")

    aggregation_model.eval()
    with torch.no_grad():
        Estimated_after = []
        for n in range(config.num_batch_test):
            te_idxs_batch = te_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]

            # 获取每个模型的预测值
            cp_embed = [cp_models[label](te_idxs_batch[:, 0], te_idxs_batch[:, 1], te_idxs_batch[:, 2])[0] for label in range(4)]

            # 聚合模型的预测
            aggregated_output = aggregation_model(cp_embed[0], cp_embed[1], cp_embed[2], cp_embed[3])
            Estimated_after.extend(aggregated_output.cpu().numpy())

    Estimated_after = np.asarray(Estimated_after)
    test_nmae_after, test_nrmse_after = accuracy(Estimated_after, te_vals)

    print(f"聚合后 - Test NMAE: {test_nmae_after:.4f}, Test NRMSE: {test_nrmse_after:.4f}")
