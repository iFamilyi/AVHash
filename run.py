import argparse
import os
import random
import json
import numpy as np
import torch
from dataset import load_data
from train import train_model
import argparse


def getfileindex(result_log_dir):
    try:
        files = os.listdir(result_log_dir)
        fileindex = sorted([int(x.split('.')[0]) for x in files ])[-1]
    except:
        fileindex = 0

    return fileindex+1

def run():

    parser = argparse.ArgumentParser(description='Script parameters')
    # 添加命令行参数
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use')
    parser.add_argument('--data_set_config', type=str, default='./Json/fcvid.json', help='Path to dataset config file')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--epoch', type=int, default=300, help='Maximum number of iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=3346, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hashcode_size', type=int, default=16, help='Hashcode size')
    parser.add_argument('--weight_path', type=str, default="", help='Path to load weight')
    parser.add_argument('--result_log_dir', type=str, default="", help='Path to log')
    parser.add_argument('--result_weight_dir', type=str, default="", help='Path to weight')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用参数
    device = args.device
    data_set_config = args.data_set_config
    num_workers = args.num_workers
    epoch = args.epoch
    lr = args.lr
    seed = args.seed
    batch_size = args.batch_size
    hashcode_size = args.hashcode_size
    result_log_dir = args.result_log_dir
    result_weight_dir = args.result_weight_dir
    weight_path = args.weight_path

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    train_val_data_path = (config['Image'],config['Audio'])

    print("train device:", device)
    print("data_path:", train_val_data_path)
    print("num_workers: ", num_workers)
    print("lr:",lr)
    print("hashcode_size:",hashcode_size)

    # Load dataset
    train_dataloader, val_dataloader, data_dataloader = load_data(
        data_set_config = data_set_config   # 修改， Anet_z 中的所有权重都大于1
        ,batch_size = batch_size
        ,num_workers = num_workers                # 修改
        ,pn=5
    )


    fileindex = str(getfileindex(result_log_dir))
    print(fileindex)

    with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
        f.write('train device: {}\n'.format(device))
        f.write('data_path: {}\n'.format(train_val_data_path))
        f.write('seed = {}\n'.format(seed))
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('num_workers = {}\n'.format(num_workers))
        f.close()

    # Training
    train_model(
        train_loader=train_dataloader
        , q_image_loader=val_dataloader
        , r_image_loader=data_dataloader
        , hashembeding_dim=hashcode_size
        , device=device
        , max_iter= epoch
        , lr= lr
        , fileindex=fileindex
        , result_log_dir=result_log_dir
        , result_weight_dir=result_weight_dir
        , weight_path = weight_path
    )

if __name__ == '__main__':

    run()

