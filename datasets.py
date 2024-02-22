import os
import sys
from PIL import Image
from time import time

import torch
import torchvision
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
from simclr.modules.transformations import TransformsSimCLR
import random
import transforms


def get_loader(dataset, batch_size, num_workers, labeled_ind, use_gpu):

    pin_memory = True if use_gpu else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        sampler=SubsetRandomSampler(labeled_ind),
    )

    return dataloader


def get_myloader(dataset, batch_size, num_workers, labeled_ind, use_gpu):

    pin_memory = True if use_gpu else False
    dataset = MyDataset(dataset)
    unlabeledloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        sampler=SubsetRandomSampler(labeled_ind),
    )
    return unlabeledloader


def get_dataset(dataset):

    if dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        trainset = torchvision.datasets.CIFAR100("../Data/CIFAR100", train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100("../Data/CIFAR100", train=False, download=False, transform=transform_test)
    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10("../Data/CIFAR10", train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10("../Data/CIFAR10", train=False, download=False, transform=transform_test)
    elif dataset == "tinyimagenet":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),  # 随机裁剪为64x64大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])

        transform_val = transforms.Compose([
            transforms.Resize(64),  # 调整大小为64x64
            transforms.CenterCrop(64),  # 中心裁剪为64x64
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])

        trainset = torchvision.datasets.ImageFolder('../Data/tiny-imagenet-200/train', transform=transform_train)
        testset = TinyImageNet('../Data/tiny-imagenet-200/', train=False, transform=transform_val)

    return trainset, testset


def get_openset(trainset, testset, known_class, init_percent):

    print("openset here!")
    labeled_ind_train, unlabeled_ind_train, known_nums = filter_known_unknown_10percent(trainset, known_class, init_percent)
    unlabeled_ind_test = filter_known_unknown(testset, known_class)

    return labeled_ind_train, unlabeled_ind_train, unlabeled_ind_test, known_nums


def filter_known_unknown(dataset, known_class):
    filter_ind = []
    for i in range(len(dataset.targets)):
        c = dataset.targets[i]
        if c < known_class:
            filter_ind.append(i)
    return filter_ind


def filter_known_unknown_10percent(dataset, known_class, init_percent):
    filter_ind = []
    unlabeled_ind = []
    for i in range(len(dataset.targets)):
        c = dataset.targets[i]
        if c < known_class:
            filter_ind.append(i)
        else:
            unlabeled_ind.append(i)

    # 随机选
    known_nums = len(filter_ind)
    random.shuffle(filter_ind)
    labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
    unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
    return labeled_ind, unlabeled_ind, known_nums


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indices = list(range(len(data)))

    def __getitem__(self, index):
        data = self.data[index]
        return self.indices[index], data


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        self.targets = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                            tar = self.class_to_tgt_idx[tgt]
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                            tar = self.class_to_tgt_idx[self.val_img_to_class[fname]]
                        self.images.append(item)
                        self.targets.append(tar)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt