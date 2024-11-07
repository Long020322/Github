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

    train_ratio = 0.1
    checkpoint_dir = ('./results/checkpoint/{}_{}_{}.pt').format(Models[args.mm], Datasets[args.dd], train_ratio)

    # 加载采样后的数据
    tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals = config.Sampling(train_ratio)

    # 初始化3个CP模型（对应三个标签）
    cp_models = [CP(config.num_dim, config.num_emb).cuda() for _ in range(3)]
    optimizers = [optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) for model in cp_models]
    criterion = nn.L1Loss()

    # 为每个标签创建独立的 EarlyStopping 对象
    early_stopping = [EarlyStopping(patience=args.patience, verbose=True, save_path=f'{checkpoint_dir}_label{label}.pt') for label in range(3)]
    stop_flags = [False] * 3  # 初始时，所有模型的 stop_flags 都是 False

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')

        for label in range(3):
            if stop_flags[label]:
                continue

            cp_models[label].train()
            train_loss = 0

            # 选取对应标签的数据
            selected_data = tr_idxs[tr_idxs[:, 4] == label]
            selected_vals = tr_vals[tr_idxs[:, 4] == label]

            if len(selected_data) == 0:
                continue

            # 随机打乱数据索引
            random_perm_idx = torch.randperm(len(selected_data))
            selected_data = selected_data[random_perm_idx]
            selected_vals = selected_vals[random_perm_idx]

            num_batches = math.ceil(len(selected_data) / config.batch_size)

            for n in range(num_batches):
                batch_data = selected_data[n * config.batch_size : (n + 1) * config.batch_size]
                batch_vals = selected_vals[n * config.batch_size : (n + 1) * config.batch_size]

                i = batch_data[:, 0]
                j = batch_data[:, 1]
                k = batch_data[:, 2]

                optimizers[label].zero_grad()
                predict = cp_models[label](i, j, k)
                loss = criterion(predict, batch_vals)
                loss.backward()
                optimizers[label].step()

                train_loss += loss.item()

            train_loss /= num_batches
            print(f"Label {label} - Train Loss: {train_loss}")

            # 验证模型
            cp_models[label].eval()
            valid_loss = 0
            selected_data = va_idxs[va_idxs[:, 4] == label]
            selected_vals = va_vals[va_idxs[:, 4] == label]

            if len(selected_data) > 0:
                i = selected_data[:, 0]
                j = selected_data[:, 1]
                k = selected_data[:, 2]
                with torch.no_grad():
                    valid_output = cp_models[label](i, j, k)
                    valid_loss = criterion(valid_output, selected_vals).item()

            # 早停检查
            early_stopping[label](valid_loss, cp_models[label])

            if early_stopping[label].early_stop:
                print(f"Early stopping for label {label}")
                stop_flags[label] = True  # 标记该模型不再继续训练

        # 如果所有模型都已经早停，则提前结束整个训练过程
        if all(stop_flags):
            print("All models have early stopped. Ending training.")
            break

    # 聚合前的测试
    print("Testing before aggregation...")

    # 聚合前的测试前加载早停保存的最佳模型参数
    for label in range(3):
        checkpoint_path = f'{checkpoint_dir}_label{label}.pt'
        if os.path.exists(checkpoint_path):
            cp_models[label].load_state_dict(torch.load(checkpoint_path))

    cp_models = [model.eval() for model in cp_models]  # 确保所有模型都处于评估模式

    # 计算聚合前的NMAE和NRMSE
    with torch.no_grad():
        Estimated_before = []
        for n in range(config.num_batch_test):
            te_idxs_batch = te_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            labels = te_idxs_batch[:, 4]

            for i in range(len(te_idxs_batch)):
                label = labels[i].item()
                i_idx = te_idxs_batch[i, 0].unsqueeze(0)
                j_idx = te_idxs_batch[i, 1].unsqueeze(0)
                k_idx = te_idxs_batch[i, 2].unsqueeze(0)

                predict = cp_models[label](i_idx, j_idx, k_idx)
                Estimated_before += predict.cpu().numpy().tolist()

    Estimated_before = np.asarray(Estimated_before)
    test_nmae_before, test_nrmse_before = accuracy(Estimated_before, te_vals)

    # 初始化聚合模型
    aggregation_model = AggregationModel().cuda()

    # 设置聚合模型的优化器、损失函数和早停机制
    optimizer = optim.AdamW(aggregation_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.L1Loss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='./results/aggregation_model_best.pt')

    # 聚合模型训练循环
    for epoch in range(config.epochs):
        aggregation_model.train()
        train_loss = 0

        # 随机打乱数据索引
        random_perm_idx = torch.randperm(len(tr_idxs))
        tr_idxs = tr_idxs[random_perm_idx]
        tr_vals = tr_vals[random_perm_idx]

        num_batches = math.ceil(len(tr_idxs) / config.batch_size)

        for n in range(num_batches):
            batch_data = tr_idxs[n * config.batch_size : (n + 1) * config.batch_size]
            batch_vals = tr_vals[n * config.batch_size : (n + 1) * config.batch_size]

            i = batch_data[:, 0]
            j = batch_data[:, 1]
            k = batch_data[:, 2]

            # 获取每个CP模型的预测值
            outputs = [cp_models[label](i, j, k) for label in range(3)]

            # 聚合模型的预测
            aggregated_output = aggregation_model(outputs[0], outputs[1], outputs[2])

            # 计算损失并优化聚合模型
            loss = criterion(aggregated_output, batch_vals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= num_batches
        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}")

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
                outputs = [cp_models[label](i, j, k) for label in range(3)]

                # 聚合模型的验证预测
                aggregated_output = aggregation_model(outputs[0], outputs[1], outputs[2])
                loss = criterion(aggregated_output, batch_vals)
                valid_loss += loss.item()

            valid_loss /= num_batches_val
            print(f"Epoch {epoch + 1}/{config.epochs} - Validation Loss: {valid_loss:.4f}")

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
            outputs = [cp_models[label](te_idxs_batch[:, 0], te_idxs_batch[:, 1], te_idxs_batch[:, 2]) for label in range(3)]

            # 聚合模型的预测
            aggregated_output = aggregation_model(outputs[0], outputs[1], outputs[2])
            Estimated_after.extend(aggregated_output.cpu().numpy())

    Estimated_after = np.asarray(Estimated_after)
    test_nmae_after, test_nrmse_after = accuracy(Estimated_after, te_vals)
    
    # 输出汇总结果
    print("\n测试结果汇总：")
    print(f"聚合前 - Test NMAE: {test_nmae_before:.4f}, Test NRMSE: {test_nrmse_before:.4f}")
    print(f"聚合后 - Test NMAE: {test_nmae_after:.4f}, Test NRMSE: {test_nrmse_after:.4f}")
