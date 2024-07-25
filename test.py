import random
import numpy as np
import eval as Eval
import torch
from model import MMVH
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import h5py
import argparse


class QRDataset(Dataset):
    def __init__(self, config_file, image_features, audio_features, phase='retrieval'): # val, test, dataset
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.dataset = config['dataset']
        # 数据名称
        if phase == 'val':
            self.list_file = config['val_list']
        elif phase == 'test':
            self.list_file = config['test_list']
        elif phase == 'retrieval':
            self.list_file = config['retrieval_list']
        self.num_classes = config['num_class']
        self.image_features = image_features
        self.audio_features = audio_features


        self.video_data = self.load_videos_from_file(self.list_file) # 名称加载成字典
        self.indexes = self._make_indexes(self.video_data) # (类别，名称)

    def load_videos_from_file(self,file_path):
        video_data = {}
        if self.dataset == 'actnet':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, frame_count, class_label = line.strip().split(',')[0:3]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    # Append the video information to the class list
                    video_data[class_label].append(video_name)
        elif self.dataset == 'fcvid':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, class_label = line.strip().split(',')[0:2]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

                    # Append the video information to the class list
                    video_data[class_label].append(video_name)

        return video_data

    def _make_indexes(self, video_data):

        indexes = []
        for class_label, videos in video_data.items():
            for i in range(len(videos)):
                indexes.append((i, class_label))
        print('video number:%d' % (len(indexes)))
        return indexes


    def __getitem__(self, index):
        video_index, class_label = self.indexes[index] #
        video = self.video_data[class_label][video_index] # video 是一个视频名字

        image_features = self.image_features[video]
        audio_features = self.audio_features[video]

        if self.dataset == 'actnet':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=self.num_classes).float() # 标签独热码
        elif self.dataset == 'fcvid':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label-1), num_classes=self.num_classes).float() # 标签独热码

        return image_features,audio_features,target_onehot, index

    def __len__(self):
        return len(self.indexes)

def load_h5_file_to_memory(image_features_path , audio_faetures_path):
    image_features = {}
    audio_features = {}
    with h5py.File(image_features_path, 'r') as h5_file1:
        for key in h5_file1.keys():
            image_features[key] = torch.tensor(h5_file1[key]['vectors'][...])


    with h5py.File(audio_faetures_path, 'r') as h5_file2:
        for key in h5_file2.keys():
            audio_features[key] = torch.tensor(h5_file2[key]['vectors'][...])
    return image_features, audio_features


def load_test_data(data_set_config,batch_size,num_workers):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_features, audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    query_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_features,
                  audio_features=audio_features,
                  phase='test',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_features,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return  query_dataloader, retrival_dataloader

def test_model(
        q_image_loader=None
        , r_image_loader=None
        , bert_config='./BertConfig'
        , cls_path='./cls.pt'
        , hashembeding_dim=64
        , device='cuda:6'
        , weight_path = None
):

    # Device
    use_device = torch.device(device)

    model = MMVH(model_config_path=bert_config, cls_path=cls_path,hashcode_size=hashembeding_dim).to(device)
    for name, param in model.named_parameters():
        print(name)
    state_dict = torch.load(weight_path,map_location=device)
    model.load_state_dict(state_dict['berthash'],strict=True)

    # Val or Test:
    model.eval()
    with torch.no_grad():

        q_image_code, q_image_targets = generate_code(model, q_image_loader, hashembeding_dim, use_device)
        r_image_code, r_image_targets = generate_code(model, r_image_loader, hashembeding_dim, use_device)

    topks = [5, 10, 20, 40, 60, 80, 100,200]
    PKs = []
    RKs = []
    for topk in topks:
        PK = Eval.precision_k(
            q_image_code.to(use_device),
            r_image_code.to(use_device),
            q_image_targets.to(use_device),
            r_image_targets.to(use_device),
            topk
        )

        RK = Eval.recall_k(
            q_image_code.to(use_device),
            r_image_code.to(use_device),
            q_image_targets.to(use_device),
            r_image_targets.to(use_device),
            topk
        )
        PKs.append(PK)
        RKs.append(RK)

    P_result = ' '.join(['P@{}:{:.4f}'.format(x, y) for x, y in zip(topks, PKs)])
    print(P_result)
    R_result = ' '.join(['R@{}:{:.4f}'.format(x, y) for x, y in zip(topks, RKs)])
    print(R_result)

    mAPs = []
    topks = [5, 10, 20, 40, 60, 80, 100, 200, None]
    for topk in topks:
        mAP = Eval.mean_average_precision(
            q_image_code.to(use_device),
            r_image_code.to(use_device),
            q_image_targets.to(use_device),
            r_image_targets.to(use_device),
            device=use_device,
            topk=topk
        )
        mAPs.append(mAP)

    map_result = ' '.join(['mAP@{}:{:.4f}'.format(x, y) for x, y in zip(topks, mAPs)])
    print(map_result)

    P, R = Eval.pr_curve_3(
        q_image_code.to(use_device),
        r_image_code.to(use_device),
        q_image_targets.to(use_device),
        r_image_targets.to(use_device),
        topK=-1
        # use_device
    )
    print(P)
    print(R)

def generate_code(model, dataloader, code_length, device):

    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        numclass = dataloader.dataset.num_classes
        code = torch.zeros([N, code_length])
        target = torch.zeros([N, numclass])
        for image_features, audio_features,  tar, index in dataloader:
            # for data, data_mask, tar, index in dataloader:
            image_features = image_features.to(device)
            audio_features = audio_features.to(device)

            _,_,hash_code = model(image_features, audio_features)
            code[index, :] = hash_code.sign().cpu()
            target[index, :] = tar.clone().cpu()
    torch.cuda.empty_cache()
    return code, target


def run():

    parser = argparse.ArgumentParser(description='Script parameters')
    # 添加命令行参数
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use')
    parser.add_argument('--data_set_config', type=str, default='./Json/fcvid.json', help='Path to dataset config file')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--seed', type=int, default=3346, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hashcode_size', type=int, default=16, help='Hashcode size')
    parser.add_argument('--weight_path', type=str, default="", help='Path to load weight')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用参数
    device = args.device
    data_set_config = args.data_set_config
    num_workers = args.num_workers
    seed = args.seed
    batch_size = args.batch_size
    hashcode_size = args.hashcode_size
    weight_path = args.weight_path

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    print("train device:", device)
    print("num_workers: ", num_workers)
    print("hashcode_size:",hashcode_size)

    # Load dataset
    val_dataloader, data_dataloader = load_test_data(
        data_set_config = data_set_config
        ,batch_size = batch_size
        ,num_workers = num_workers
    )

    # Training
    test_model(
        q_image_loader=val_dataloader
        , r_image_loader=data_dataloader
        , hashembeding_dim=hashcode_size
        , device=device
        , weight_path = weight_path
    )

if __name__ == '__main__':

    run()

