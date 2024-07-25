import random
import json
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class TrainDataset(Dataset):  #  video_name, frame_count, class_label = line.strip().split(',')[0:3]

    def __init__(self, config_file, image_features, audio_features, pn):
        with open(config_file, 'r') as f:
            config = json.load(f)
            f.close()
        self.dataset = config['dataset']
        self.image_features = image_features
        self.audio_features = audio_features
        self.list_file = config['train_list']
        self.pn = pn

        self.data = self.load_videos_from_file(self.list_file)
        self.classes = list(self.data.keys())
        self.indexes = self._make_indexes(self.data)

    def load_videos_from_file(self,file_path):
        video_data = {}
        if self.dataset == 'actnet':
            with open(file_path, 'r') as file:
                for line in file:
                    video_name, frame_count, class_label = line.strip().split(',')[0:3]
                    class_label = int(class_label)  # Convert class label to integer

                    if class_label not in video_data:
                        video_data[class_label] = []

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
        return indexes


    def select_n_neg_elements(self, divtKeys, N):
        selected_keys = random.sample(divtKeys, N)
        all_elements = []
        for key in selected_keys:
            all_elements.extend(random.sample(self.data[key],1))
        return all_elements

    def __getitem__(self, index):

        # anchor
        class_index, class_label = self.indexes[index]
        anchor = self.data[class_label][class_index] # video 的名字
        # positive
        pos_class_index = (class_index + random.randint(1, len(self.data[class_label]) - 1)) % len(
                self.data[class_label])
        pos = self.data[class_label][pos_class_index]
        # negtive
        neg_num = self.pn - 1
        neg_classes = [cls for cls in self.classes if cls != class_label]
        negs = self.select_n_neg_elements(neg_classes, neg_num)

        anchorI, anchorA = self.image_features[anchor], self.audio_features[anchor]
        posI, posA = self.image_features[pos], self.audio_features[pos]
        negI = []
        negA = []
        for neg in negs:
            negI.append(self.image_features[neg])
            negA.append(self.audio_features[neg])

        return anchorI, anchorA, posI, posA, torch.stack(negI,dim=0), torch.stack(negA,dim=0)

    def __len__(self):

        return len(self.indexes)


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
        video = self.video_data[class_label][video_index]

        image_features = self.image_features[video]
        audio_features = self.audio_features[video]

        if self.dataset == 'actnet':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=self.num_classes).float() # 标签独热码
        elif self.dataset == 'fcvid':
            target_onehot = torch.nn.functional.one_hot(torch.tensor(class_label-1), num_classes=self.num_classes).float() # 标签独热码

        return image_features,audio_features,target_onehot, index

    def __len__(self):
        return len(self.indexes)


def load_h5_file_to_memory(image_features_path,audio_faetures_path):
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

def load_data(data_set_config,batch_size,num_workers,pn):


    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    image_fratures,audio_features = load_h5_file_to_memory(config['Image'],config['Audio'])

    train_dataloader = DataLoader(
        TrainDataset(config_file=data_set_config,
                     image_features=image_fratures,
                     audio_features=audio_features,
                     pn=pn,
                     ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    query_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='val',
                 ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrival_dataloader = DataLoader(
        QRDataset(config_file=data_set_config,
                  image_features=image_fratures,
                  audio_features=audio_features,
                  phase='retrieval',
                  ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, query_dataloader, retrival_dataloader
