import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import Config
from models import CP, MergL, Tucker, LTP
from utils import accuracy, record, EarlyStopping
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--mm', type=int, default=8)
parser.add_argument('--dd', type=int, default=0)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuid)
    torch.set_default_dtype(torch.float32)  # 设置默认数据类型为 float32
    torch.set_default_device('cuda')  # 设置默认设备为 CUDA


if __name__ == "__main__":
    mm = args.mm   # 0-CP, 1-Merg, 2-MergL,3-Tucker, 4-LTP, 5-NTC, 6-NTM, 7-CoSTCo, 8-ImprovedCP
    Models = ['CP', 'MergL','Tucker', 'LTP']
    dd = args.dd   # 0-Abilene, 1-Geant, 2-WSdreamtp, 3-PlanetLab, 4-Harvard72, 5-Seattle, 6-MBD, 7-PMU
    Datasets = ['Abilene', 'Geant', 'WSdreamtp', 'PlanetLab', 'Harvard72', 'Seattle', 'MBD', 'PMU']
    config = Config('./data/' + Datasets[dd] + '.ini')
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    train_ratio = 0.5
    error_list = []
    checkpoint_dir = ('./results/checkpoint/{}_{}_{}.pt').format(Models[mm], Datasets[dd], train_ratio)


    print("Dataset - " + Datasets[dd])
    tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals = config.Sampling(train_ratio)

    print("Model - " + Models[mm])
    if   mm == 0:
        model = CP(config.num_dim, config.num_emb)
    elif mm == 1:
        model = MergL(config.num_dim, config.num_emb)       
    elif mm == 2:
        model = Tucker(config.num_dim, config.num_embs)
    elif mm == 3:
        model = LTP(config.num_dim, config.num_emb, config.period)
    else:
        raise Exception("Illegal option!")

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=checkpoint_dir)


    print('Training...')                
    for epoch in range(config.epochs):
        model.train()          
        random_perm_idx = np.random.permutation(int(config.num_train))
        for n in range(config.num_batch):
            batch_set_idx = random_perm_idx[n * config.batch_size: (n + 1) * config.batch_size]
            tr_idxs_batch = tr_idxs[batch_set_idx]
            i = tr_idxs_batch[:, 0]
            j = tr_idxs_batch[:, 1]
            k = tr_idxs_batch[:, 2]

        
            val = tr_vals[batch_set_idx]
            optimizer.zero_grad()  # 把所有变量的梯度初始化为0

            predict = model(i, j, k)
            loss = criterion(predict, val)     
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{config.epochs}]    train_loss: {loss}')

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for n in range(config.num_batch_vali):
                idxs_batch = va_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
                i = idxs_batch[:, 0]
                j = idxs_batch[:, 1]
                k = idxs_batch[:, 2]

                val = va_vals[n * config.batch_size_eval: (n + 1) * config.batch_size_eval] #torch.Size([1024])
                valid_output = model(i, j, k)
                valid_loss += criterion(valid_output, val)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


    print("Testing...")
    model.load_state_dict(torch.load(checkpoint_dir))  # 获得 early stopping 时的模型参数

    model.eval()  # 不启用BatchNormalization和Dropout
    with torch.no_grad():
        Estimated0 = []
        for n in range(config.num_batch):
            idxs_batch = tr_idxs[n * config.batch_size: (n + 1) * config.batch_size]
            i = idxs_batch[:, 0]
            j = idxs_batch[:, 1]
            k = idxs_batch[:, 2]

            predict = model(i, j, k)
            Estimated0 += predict.cpu().numpy().tolist()
    # Estimated0 = np.clip(np.asarray(Estimated0), 0, 1)  # * config.max_value
    Estimated0 = np.asarray(Estimated0)

    model.eval()  # 不启用BatchNormalization和Dropout
    with torch.no_grad():
        Estimated1 = []
        for n in range(config.num_batch_vali):
            idxs_batch = va_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            i = idxs_batch[:, 0]
            j = idxs_batch[:, 1]
            k = idxs_batch[:, 2]
            
            predict = model(i, j, k)
            Estimated1 += predict.cpu().numpy().tolist()

    # Estimated1 = np.clip(np.asarray(Estimated1), 0, 1)
    Estimated1 = np.asarray(Estimated1)

    model.eval()  # 不启用BatchNormalization和Dropout
    with torch.no_grad():
        Estimated2 = []
        for n in range(config.num_batch_test):
            idxs_batch = te_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            i = idxs_batch[:, 0]
            j = idxs_batch[:, 1]
            k = idxs_batch[:, 2]

            predict = model(i, j, k)
            Estimated2 += predict.cpu().numpy().tolist()
    # Estimated = np.clip(np.asarray(Estimated), 0, 1)  
    Estimated2 = np.asarray(Estimated2)

    config.calculate_interval_overlap(tr_vals.cpu().numpy(), Estimated0 ,3)
    config.calculate_interval_overlap(va_vals.cpu().numpy(), Estimated1 ,3)
    config.calculate_interval_overlap(te_vals, Estimated2 ,3)


    #检验保序性
    # value = va_vals.cpu().numpy() 
    # estimate = Estimated1
    # config.check_order(value, estimate,boundaries)
    # print("----------------")
    # b = config.Kmeans1(Estimated0)
    # config.check_order(value, estimate, b)


    #计算NMAE和NRMSE
    train_nmae, train_nrmse = accuracy(np.concatenate((Estimated0, Estimated1)),
                                       np.concatenate((tr_vals.cpu().detach().numpy(), va_vals.cpu().detach().numpy())))
    test_nmae, test_nrmse = accuracy(Estimated2, te_vals)
    total_nmae, total_nrmse = accuracy(np.concatenate((Estimated0, Estimated1, Estimated2)),
                                       np.concatenate((tr_vals.cpu().detach().numpy(), va_vals.cpu().detach().numpy(), te_vals)))
                                       
    # #计算大值的NMAE和NRMSE
    # tr_w4 = tr_idxs[:, 7].cpu().detach().numpy()
    # va_w4 = va_idxs[:, 7].cpu().detach().numpy()
    # te_w4 = te_idxs[:, 7].cpu().detach().numpy()

    # Esbig0 = tr_w4 * Estimated0
    # Esbig1 = va_w4 * Estimated1
    # Esbig  = te_w4 * Estimated
    # train_nmae_big, train_nrmse_big = accuracy(np.concatenate((Esbig0, Esbig1)), 
    #                                            np.concatenate((tr_w4 * tr_vals.cpu().detach().numpy(), va_w4 * va_vals.cpu().detach().numpy())))
    # test_nmae_big, test_nrmse_big = accuracy(Esbig, te_w4 * te_vals)
    # total_nmae_big, total_nrmse_big = accuracy(np.concatenate((Esbig0, Esbig1, Esbig)),
    #                                            np.concatenate((tr_w4 * tr_vals.cpu().detach().numpy(), va_w4 *  va_vals.cpu().detach().numpy(), te_w4 * te_vals)))
    
    print("Sampling rate %.2f is done!" % train_ratio)
    print("Learning rate {} is done!".format(config.lr))


    print('{:10}\ttrain nmae\ttest nmae\tall nmae\ttrain nrmse\ttest nrmse\tall nrmse'.format(Datasets[dd]))
    print('{:14}\t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}'
          .format(Models[mm], train_nmae, test_nmae, total_nmae, train_nrmse, test_nrmse, total_nrmse))
    
    # print('{:10}\ttra nmae big\tte nmae big\tall nmae big\ttra nrmse big\tte nrmse big\tall nrmse big'.format(Datasets[dd]))
    # print('{:14}\t{:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}\t    {:5.4f}'
    #       .format(Models[mm], train_nmae_big, test_nmae_big, total_nmae_big, train_nrmse_big, test_nrmse_big, total_nrmse_big))    
    

    # error_list += [np.asarray([train_nmae, test_nmae, total_nmae, train_nrmse, test_nrmse, total_nrmse])]
    # record('./results/{}_{}_{}_error.csv'.format(Models[mm], Datasets[dd], train_ratio), np.asarray(error_list))



    # store_1 = np.concatenate((Estimated0, Estimated1, Estimated))
    # store_2 = np.concatenate((tr_idxs.cpu().numpy(), va_idxs.cpu().numpy(), te_idxs.cpu().numpy()))
    # np.save('./results/{}_{}_{}_estimate.npy'.format(Models[mm], Datasets[dd], train_ratio), store_1)
    # np.save('./results/{}_{}_{}_index.npy'.format(Models[mm], Datasets[dd], train_ratio), store_2)
    

