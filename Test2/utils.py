import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
# import os
# import scipy.stats
# from sklearn.metrics import ndcg_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        # if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)     # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


# def sampling(data_path, train_ratio, loc_path):
#     data = np.load(data_path)
#     max_value = np.max(data)
#     data = data / max_value
#     location = np.load(loc_path).astype(int)
#     num_sample = int(np.ceil(np.max(location) * train_ratio))
#     train_mask = ((location > 0) & (location <= num_sample)).astype(np.float32)
#     test_mask = (location > num_sample).astype(np.float32)
#     train_data = train_mask * data
#     test_data = test_mask * data
#     num_train = np.sum(train_mask)
#     num_test = np.sum(test_mask)
#     return train_data, train_mask, test_data, test_mask, num_train, num_test, max_value


def accuracy(predict, data):
    error = np.abs(predict - data)
    NMAE = np.sum(error) / np.sum(np.abs(data))
    NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
    return NMAE, NRMSE


# def accuracy(predict, data, num, num_time):
#     mask = (data > 0).astype(float)
#     error = np.multiply(mask, np.abs(predict - data))
#     NMAE = np.sum(error) / np.sum(np.abs(data))
#     NRMSE = np.sqrt(np.sum(error ** 2) / np.sum(data ** 2))
#     KL = scipy.stats.entropy(data.reshape(-1), (mask * predict).reshape(-1))
#     # NDCG = ndcg_score(data.reshape(num_time, -1), (mask * predict).reshape(num_time, -1), k=10)
#     NDCG = ndcg_score(data.transpose(2, 0, 1).reshape(num_time, -1), (mask * predict).transpose(2, 0, 1).reshape(num_time, -1), k=10)
#     MAE = np.sum(error) / num
#     RMSE = np.sqrt(np.sum(error ** 2) / num)
#     return NMAE, NRMSE, KL, NDCG, MAE, RMSE


def record(record_file, results):
    with open(record_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)
        csvfile.close()


def tensor2tuple(tensor_):
    tuples = []
    I, J, K = tensor_.shape  # 确保张量是 (12, 12, time_slots)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                value = tensor_[i, j, k]
                if value != 0:  # 过滤掉值为0的元素
                    tuples.append([i, j, k, value])
    
    return np.array(tuples)

def tuple2tensor(indices, values):
    I = np.max(indices[:, 0])+1
    J = np.max(indices[:, 1])+1
    K = np.max(indices[:, 2])+1
    tensor_ = np.zeros((I, J, K))
    for n in range(len(values)):
        [i, j, k] = indices[n].tolist()
        tensor_[i, j, k] = values[n]
    return tensor_


def idx2seq(index, period=5):
    seq = []
    # for k in index:
    #     if k < period:
    #         tmp_seq = [0] * (period - k) + list(range(k))
    #     else:
    #         tmp_seq = list(range(k - period, k))
    for k in index:
        tmp_seq = list(range(k - period, k))
        seq += [tmp_seq]
    seq = np.asarray(seq)
    seq = np.clip(seq, 0, None)
    return seq

# def loc2mat():
#     data = np.load('./data/Harvard72Store.npy')
#     data = np.round(data, 4)
#     data[64, data[64] > 1e3] = 0        # 4e3
#     np.save('../tucker_compress/data/Harvard72.npy', data)
#
#     data = tensor2tuple(data)
#     loc = np.random.permutation(len(data))
#     matrix = tuple2tensor(data[:, 0:3].astype(int), loc+1)
#     np.save('../tucker_compress/data/Harvard72_loc.npy', matrix)





'''====================读取Matlab数据===================='''
# from scipy.io import loadmat
# m = loadmat('E:/Workspace/Matlab/TensorDecomposition/DataSets/Harvard_224_224_72.mat')
# data = m['data']
# np.save('E:/Workspace/Python/AECGAN/data/Harvard/data.npy', data)

'''====================保存Matlab数据===================='''
# import scipy.io as io
# mat_path = 'E:/Workspace/Matlab/data'
# io.savemat(mat_path, {'data': data})

