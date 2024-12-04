import configparser
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import torch
import math
import json
import tensorly as tl
from utils import idx2seq, tensor2tuple, tuple2tensor
from tensorly.decomposition import parafac

class Config(object):

    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.batch_size = conf.getint("Model_Setup", "bs")
        self.batch_size_eval = conf.getint("Model_Setup", "bs_eval")
        self.seed = conf.getint("Model_Setup", "seed")

        # Dataset
        self.num_dim = np.array(json.loads(conf.get("Data_Setting", "ndim")))
        self.num_emb = conf.getint("Data_Setting", "nemb")
        self.num_embs = np.array(json.loads(conf.get("Data_Setting", "nembs")))
        self.period = conf.getint("Data_Setting", "period")
        self.channels = conf.getint("Data_Setting", "channels")

        self.data_path = conf.get("Data_Setting", "data_path")
        self.location_path = conf.get("Data_Setting", "location_path")

        self.num_batch = None
        self.num_batch_vali = None
        self.num_batch_test = None
        self.num_train = None
        self.max_value = None

    # def split(self):
    #     # 加载数据
    #     data = np.load(self.data_path)  # (428536, 4)
    #     # 计算数据
    #     data_val = data[:, -1]
    #     median = np.median(data_val)
    #     mean = np.mean(data_val)
    #     percentile_90 = np.percentile(data_val, 90)
    #     # 初始化 idxs
    #     idxs = np.zeros_like(data)
    #     # 使用布尔索引进行条件赋值
    #     idxs[data_val > percentile_90, 3] = 1
    #     idxs[data_val > mean, 2] = 1
    #     idxs[data_val > median, 1] = 1
    #     idxs[data_val > 0, 0] = 1
    #     # idxs[data_val > percentile_90, :] = 1
    #     # idxs[(data_val <= percentile_90) & (data_val > mean), :3] = 1
    #     # idxs[(data_val <= mean) & (data_val > median), :2] = 1
    #     # idxs[data_val <= median, 0] = 1
    #     # 拼接数据
    #     data = np.hstack((data, idxs))

    #     return data


    def Sampling(self, train_ratio):
        data = np.load(self.data_path)  #(428536, 4)
        data = self.add_labels_to_data(data, 3, 100)
        loc = np.load(self.location_path)

        data = data[loc]
        num_train = int(np.ceil(len(data) * train_ratio))
        num_test = len(data) - num_train
        num_valid = int(num_train * 0.05)
        num_train -= num_valid
        train_data = data[:num_train]
        valid_data = data[num_train: num_train+num_valid]
        test_data = data[num_train+num_valid:]
        max_value = np.max(train_data[:, -1])
        # max_value = 1


        tr_idxs = torch.from_numpy(train_data[:].astype(int)).cuda().long()
        tr_vals = torch.from_numpy(train_data[:, 3] / max_value).float().cuda()
        va_idxs = torch.from_numpy(valid_data[:].astype(int)).cuda().long()
        va_vals = torch.from_numpy(valid_data[:, 3] / max_value).float().cuda()
        te_idxs = torch.from_numpy(test_data[:].astype(int)).cuda().long()
        te_vals = test_data[:, 3] / max_value

        self.num_batch = int(math.ceil(float(len(train_data)) / float(self.batch_size)))
        self.num_batch_vali = int(math.ceil(float(len(valid_data)) / float(self.batch_size_eval)))
        self.num_batch_test = int(math.ceil(float(num_test) / float(self.batch_size_eval)))
        self.num_train = num_train
        self.max_value = max_value

        # 统计训练集中每个标签的数据数量
        label_counts = [torch.sum(tr_idxs[:, 4] == label).item() for label in range(3)]
        print(f"Training Data Label Counts: {label_counts}")

        return tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals

    def add_labels_to_data(self, original_data, n_clusters, time_slots):

            # 将数组分割为 indices 和 values
            indices = original_data[:, :3]  # 取前3列作为索引
            values = original_data[:, 3]    # 第4列作为值
            indices = indices.astype(int)

            # 使用 tuple2tensor 函数转换为三维张量
            tensor = tuple2tensor(indices, values)
            
            # 重塑张量
            data = tensor.reshape(144, 3000)

            # 提取前 time_slots 个时隙的数据
            data_for_kmeans = data[:, :time_slots]

            # KMeans 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data_for_kmeans)

            def evaluate_clusters(data, labels):
                sil_score = silhouette_score(data[:, :3], labels)
                ch_score = calinski_harabasz_score(data[:, :3], labels)
                print(f'Silhouette Score: {sil_score:.4f}, CH Score: {ch_score:.4f}')

            # 评估 tr_idxs 中的前三列数据的聚类效果
            evaluate_clusters(data, labels)

            # 创建路径到标签的映射
            path_labels = {}
            for i in range(144):
                path_labels[(i // 12, i % 12)] = labels[i]  # 路径是 (行, 列)

            # 初始化一个新列以存储聚类标签
            label_column = np.full(original_data.shape[0], -1)  # 默认值为 -1

            # 遍历原始数据，填充标签列
            for i in range(original_data.shape[0]):
                path = (original_data[i, 0], original_data[i, 1])  # 获取路径
                if path in path_labels:
                    label_column[i] = path_labels[path]  # 填充标签

            # 将标签添加到原始数据中
            new_data = np.column_stack((original_data, label_column))  # 合并

            # 返回新数据
            return new_data

    def plot_errors(self,vaule, estimate, dataset_name, save_path):
        absolute_errors = vaule - estimate
        relative_errors = abs(absolute_errors) / vaule

        # 绘制绝对误差的散点图
        plt.figure(figsize=(10, 5))
        plt.scatter(vaule, absolute_errors, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')  # 0误差线
        plt.xlabel('Actual Values')
        plt.ylabel('Absolute Errors')
        plt.title('Absolute Errors vs Actual Values')               
        plt.grid()
        plt.savefig(f'{save_path}/absolute_errors_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 过滤相对误差
        threshold = 10  # 根据情况选择合适的阈值
        filtered_indices = relative_errors < threshold

        # 绘制相对误差的散点图
        plt.figure(figsize=(10, 5))
        plt.scatter(vaule[filtered_indices], relative_errors[filtered_indices], alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')  # 0相对误差线
        plt.xlabel('Actual Values')
        plt.ylabel('Relative Error')
        plt.title(f'{dataset_name} Relative Error vs Actual Values')
        plt.grid()
        plt.savefig(f'{save_path}/relative_error_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    # def cluster_data(self, data, n_clusters):
    #     # 对数据进行 KMeans 聚类
    #     data = data.reshape(-1, 1)
    #     kmeans = KMeans(n_clusters, random_state=0)
    #     kmeans.fit(data)
        
    #     # 获取每个数据点的聚类标签
    #     labels = kmeans.labels_
        
    #     # 获取每个聚类的边界（最小值和最大值），并按最小值排序
    #     boundaries = []
    #     for i in range(n_clusters):
    #         cluster_data = data[labels == i]  # 获取属于当前类的数据点
    #         min_val = cluster_data.min()
    #         max_val = cluster_data.max()
    #         boundaries.append((min_val, max_val))
        
    #     # 按照区间的最小值对 boundary 进行排序
    #     sorted_boundaries = sorted(boundaries, key=lambda x: x[0])
        
    #     return sorted_boundaries, labels  # 返回排序后的边界和标签

    # def calculate_interval_overlap(self, original_data, filled_data, n_clusters):
    #     # 第一步：用原始数据聚类，并按区间最小值排序
    #     original_sorted_boundaries, original_labels = self.cluster_data(original_data, n_clusters)
        
    #     # 第二步：用填充后的数据再次聚类，并按区间最小值排序
    #     filled_sorted_boundaries, filled_labels = self.cluster_data(filled_data, n_clusters)
        
    #     # 输出每个聚类区间的范围
    #     print("第一步聚类区间（从小到大排序）:", original_sorted_boundaries)
    #     print("第二步聚类区间（从小到大排序）:", filled_sorted_boundaries)

    #     overlap_rates = []

    #     # 计算每个区间的重合率
    #     for i in range(n_clusters):
    #         # 获取第一次聚类中属于区间 i 的所有数据点的索引
    #         original_cluster_indices = [idx for idx, label in enumerate(original_labels) if label == i]
            
    #         # 获取这些数据点在第二次聚类中的标签
    #         matching_points = 0
    #         for idx in original_cluster_indices:
    #             if filled_labels[idx] == i:  # 检查该数据点在第二次聚类中是否还在相同的区间
    #                 matching_points += 1
            
    #         # 计算该区间的重合率
    #         total_points_in_cluster = len(original_cluster_indices)
    #         if total_points_in_cluster > 0:
    #             overlap_percentage = matching_points / total_points_in_cluster * 100
    #         else:
    #             overlap_percentage = 0.0
            
    #         overlap_rates.append(overlap_percentage)
    #         print(f"区间 {i+1} 的重合率为: {overlap_percentage:.2f}%")

    #     return overlap_rates


