import torch
import socket
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os
from torchvision.datasets import CIFAR100
import torchvision.transforms.functional as TF
import random
from io import BytesIO
from PIL import Image, ImageChops
from helper.util import AverageMeter, accuracy
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

from multiprocessing import Pool
from multiprocessing import set_start_method

import torch.multiprocessing as mp
import time
from functools import partial
from torch.utils.data import DataLoader
import sys
import pickle
import math
import copy

# from helper.loops_ckd import validate_ckd 
import torch.nn as nn
from random import shuffle

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def transform_apply(image, test_transform, aug_parameters=None, compress=False):
    try:
        if (aug_parameters is not None) and (not np.isnan(aug_parameters).any()) and aug_parameters[-1] != -2:
            # print(list(aug_parameters))
            qf, i, j, h, w, rand_HorFlip = list(aug_parameters)
            if compress:
                buffer = BytesIO()
                image.save(buffer, 'JPEG', quality=int(qf), subsampling=0)
                image = Image.open(buffer).convert("RGB")
            
            padding = 4
            output_size = (32, 32)
            image = TF.pad(image, (padding, padding, padding, padding))
            image = TF.resized_crop(image, i, j, h, w, output_size, interpolation=TF.InterpolationMode.BILINEAR)
            
            if rand_HorFlip > 0.5:
                image = TF.hflip(image)
    except:
        print("Error in transform_apply")
    # Transform to tensor
    # image = TF.to_tensor(image)
    # norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # image = norm(image)
    image = test_transform(image)
    return image, [-2]*5


def transform_new_(image, test_transform ):

    train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                        ])

    image = train_transform(image)        
    aug_parameters = [-2] * 5
    return image, aug_parameters


def transform_new(image, test_transform):
    # it works for CIFAR100
    # crop = transforms.RandomCrop(32,  padding=4)
    # i, j, h, w  = crop.get_params(image, output_size=(32,32))

    # ONLY CIFAR 100
    padding = 4
    output_size = (32, 32)
    image = TF.pad(image, (padding, padding, padding, padding))
    crop = transforms.RandomCrop(output_size[0])
    i, j, h, w  = crop.get_params(image, output_size)
    image = TF.resized_crop(image, i, j, h, w, output_size, interpolation=TF.InterpolationMode.BILINEAR)
            
    
    rand_HorFlip = random.random()
    # Random horizontal flipping
    if rand_HorFlip > 0.5:
        image = TF.hflip(image)

    # Random vertical flipping
    # if random.random() > 0.5:
    #     image = TF.vflip(image)
    #     image1 = TF.vflip(image1)

    aug_parameters = [i, j, h, w, rand_HorFlip] 

    # Transform to tensor
    # image = TF.to_tensor(image)
    # norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # image = norm(image)

    image = test_transform(image)
    return image, aug_parameters

def kl_divergence(reference_tensor, tensors):
    """
    Calculates the KL divergence between the input tensors and the reference tensor.

    Args:
        reference_tensor (torch.Tensor): The reference tensor.
        *tensors (torch.Tensor): A variable number of input tensors to compare to the reference tensor.

    Returns:
        A list of KL divergence values for each input tensor relative to the reference tensor.
    """
    # Iterate over the input tensors and calculate the KL divergence relative to the reference tensor
    kl_divs = torch.empty(len(tensors), dtype=float)
    if torch.cuda.is_available():
        kl_divs = kl_divs.cuda()
    for idx in range(len(tensors)):
        # Calculate the KL divergence between the probability distributions
        kl_divs[idx] = F.kl_div(torch.log(tensors[idx]), reference_tensor, reduction='batchmean').item()  
    return kl_divs


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def checkSimilar(image1, image2):
    # Assuming you have two PIL.Image.Image objects: image1 and image2
    diff_image = ImageChops.difference(image1, image2)  # Subtract image2 from image1
    # Calculate the rms (root mean square) value of the difference image
    h = diff_image.histogram()
    squared_diff = sum((value * ((idx % 256) ** 2)) for idx, value in enumerate(h))
    rms = math.sqrt(squared_diff / float(image1.size[0] * image1.size[1]))
    if rms > 0.01:
        print("Check Similar inputs: ", rms)


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0,
                 CKD_dict=None, data_ckd=None):
        
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        print("==> CRD default")

        # if self.train:
        #     num_samples = len(self.train_data)
        #     label = self.train_labels
        # else:
        #     num_samples = len(self.test_data)
        #     label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)


        self.transform = transform
        self.train = train
        self.CKD_dict = CKD_dict
        self.root = root
        
        # This is a container that hold the compressed data 
        self.data_ckd = data_ckd


        self.train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                ])

        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])

        if CKD_dict is not None:
            self.num_epochs = len(CKD_dict)
            self.epoch_index = [i for i in range(0, self.num_epochs)]
            shuffle(self.epoch_index)
            self.epoch_count = 0
            self.epoch = self.epoch_index[self.epoch_count]
            self.aug_parameters = self.CKD_dict[str(self.epoch)]
            print("CKD -- CRD ==> EPOCH {} is loaded".format(self.epoch_count))
            self.loadCompressedSet()
        
        if self.CKD_dict is not None:
            print("CKD -- CRD --> CRD ==> for Training : (sample, sample_ckd, target, index)")
        else:
            print("CKD -- CRD ==> Normal")


    def __getitem__(self, index):
        # New version of torchvision datasets only has .data, .target, not train_data and test_data
        target = self.targets[index]
        target = torch.tensor(target)

        sample_ckd, _ = self.transform_apply(self.Cifar_Compress[index], self.test_transform, \
                                                    aug_parameters=self.aug_parameters[index])
        
        sample, _ = self.transform_apply(self.Cifar_Compress[index][-1], self.test_transform, \
                                                            aug_parameters=self.aug_parameters[index])

        if not self.is_sample:
            # directly return
            return sample, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            # return img, target, index, sample_idx
            return sample, sample_ckd, target, index, sample_idx
    
    def __len__(self):
        return len(self.data)
    
    def loadCompressedSet(self):
        save_dir = self.root+'/cifar-100-python/train_compressed_delta_5.pickle'
        self.Cifar_Compress = pickle.load(open(save_dir, 'rb'))
        print("CKD -- CRD ==> Compressed data is loaded")
    
    def incermentEpoch(self):
        self.epoch_count += 1
        self.epoch = self.epoch_index[self.epoch_count]
        self.aug_parameters = self.CKD_dict[str(self.epoch)]
        if (self.epoch_count >= self.num_epochs):
            shuffle(self.epoch_index)
            self.epoch_count = 0
        print("CKD -- CRD ==> EPOCH {} is loaded".format(self.epoch_count))    
    
    def transform_apply(self, image, test_transform, aug_parameters=None):
        image = copy.copy(image)
        try:
            if (aug_parameters is not None) and (not np.isnan(aug_parameters).any()) and aug_parameters[-1] != -2:
                qf, i, j, h, w, rand_HorFlip = list(aug_parameters)
                
                if isinstance(image, dict):
                    image = image[qf]
                
                padding = 4
                output_size = (32, 32)
                image = TF.pad(image, (padding, padding, padding, padding))
                image = TF.resized_crop(image, i, j, h, w, output_size, interpolation=TF.InterpolationMode.BILINEAR)
            
                if rand_HorFlip > 0.5:
                    image = TF.hflip(image)
        except:
            print("Error in transform_apply")

        image = test_transform(image)
        return image, [-2]*5
    
class CIFAR100Dataset_simple(datasets.CIFAR100):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, CKD_dict=None, data_ckd=None):
        # print("... CIFAR100Dataset_simple ...")
        super().__init__(root=root, train=train, download=download,
                transform=transform, target_transform=target_transform)
        
        self.transform = transform
        self.train = train
        self.CKD_dict = CKD_dict
        self.root = root
        
        # This is a container that hold the compressed data 
        self.data_ckd = data_ckd


        self.train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                ])

        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])

        if CKD_dict is not None:
            self.num_epochs = len(CKD_dict)
            self.epoch_index = [i for i in range(0, self.num_epochs)]
            shuffle(self.epoch_index)
            self.epoch_count = 0
            self.epoch = self.epoch_index[self.epoch_count]
            self.aug_parameters = self.CKD_dict[str(self.epoch)]
            print("CKD ==> EPOCH {} is loaded".format(self.epoch_count))
            self.loadCompressedSet()
        
        if self.data_ckd is not None:
            print("CKD ==> After Selecting : (sample, data_ckd[index], target, index)")
        if self.CKD_dict is not None:
            print("CKD ==> for Training : (sample, sample_ckd, target, index)")
        else:
            print("CKD ==> Normal")
    
    def __len__(self):
        return len(self.data)

    def loadCompressedSet(self):
        save_dir = self.root+'/cifar-100-python/train_compressed_delta_5.pickle'
        self.Cifar_Compress = pickle.load(open(save_dir, 'rb'))
        print("CKD ==> Compressed data is loaded")
    
    def incermentEpoch(self):
        self.epoch_count += 1
        self.epoch = self.epoch_index[self.epoch_count]
        self.aug_parameters = self.CKD_dict[str(self.epoch)]
        if (self.epoch_count >= self.num_epochs):
            shuffle(self.epoch_index)
            self.epoch_count = 0
        print("CKD ==> EPOCH {} is loaded".format(self.epoch_count))    
    
    def transform_apply(self, image, test_transform, aug_parameters=None):
        image = copy.copy(image)
        try:
            if (aug_parameters is not None) and (not np.isnan(aug_parameters).any()) and aug_parameters[-1] != -2:
                qf, i, j, h, w, rand_HorFlip = list(aug_parameters)
                
                if isinstance(image, dict):
                    image = image[qf]
                
                padding = 4
                output_size = (32, 32)
                image = TF.pad(image, (padding, padding, padding, padding))
                image = TF.resized_crop(image, i, j, h, w, output_size, interpolation=TF.InterpolationMode.BILINEAR)
            
                if rand_HorFlip > 0.5:
                    image = TF.hflip(image)
        except:
            # breakpoint()
            print("Error in transform_apply")
        # Transform to tensor
        # image = TF.to_tensor(image)
        # norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        # image = norm(image)
        image = test_transform(image)
        return image, [-2]*5


    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]
        target = torch.tensor(target)
        
        if self.train:
            if self.data_ckd is not None:
                 return sample, self.data_ckd[index], target, index
            elif self.CKD_dict is not None:

                #  compare that the loaded image is similar to the self.data[index]
                # checkSimilar(Image.fromarray(sample), self.Cifar_Compress[index][-1])

                sample_ckd, _ = self.transform_apply(self.Cifar_Compress[index], self.test_transform, \
                                                                 aug_parameters=self.aug_parameters[index])
                sample, _ = self.transform_apply(self.Cifar_Compress[index][-1], self.test_transform, \
                                                                 aug_parameters=self.aug_parameters[index])

                # sample = self.train_transform(self.Cifar_Compress[index][-1])

                return sample, sample_ckd, target, index
            elif self.transform is not None:
                return self.transform(Image.fromarray(sample)), target, index
            else:
                return sample, target, index

        else:
            if self.data_ckd is not None:
                 return sample, self.data_ckd[index], target, index
            elif self.CKD_dict is not None:
                sample_ckd, _ = transform_apply(Image.fromarray(sample), self.test_transform, \
                                                                 aug_parameters=None)  
                sample, _ = transform_apply(Image.fromarray(sample), self.test_transform, \
                                                                 aug_parameters=None)  
                return sample, sample_ckd, target
            elif self.transform is not None: 
                return self.transform(Image.fromarray(sample)), target, index
            else:
                return sample, target
            

class CKD_selector_parallel(object):
    def __init__(self, dataset_size, model=None, delta=5, train=False, ckd_batch_size=100, batch_size=64, num_workers=16, \
                 mode="online", ckd_model_t_path="", distill='', k=16384):
        print("... CKD_selector_parallel ...")
        self.train = train
        self.model = model
        self.delta = delta
        self.num_workers = num_workers
        self.k_crd = k
        self.QF_range = range(0,100+self.delta, self.delta)
        # self.QF_range = []
        self.num_channels, self.height, self.width = 3 , 32 , 32
        self.ckd_batch_size=ckd_batch_size
        self.batch_size=batch_size
        self.dataset_size = dataset_size

        self.distill = distill
        
        if model is not None and torch.cuda.is_available():
            self.model.cuda()


        self.train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                ])

        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])

        if self.train:
            self.transform = transform_new
            self.shuffle = True
            self.ckd_model_t_path = ckd_model_t_path.replace("<train/val>", "train")
            print("==> Random Augmentation")
        
        else:
            self.transform = transform_apply
            self.shuffle = False
            self.ckd_model_t_path = ckd_model_t_path.replace("<train/val>", "val")
            print("==> Validation Augmentation")
        
        if mode == "online":
            # self.loadCompressedSet()
            self.callable_func = self.online_generation
        
        elif mode == "save_ckd" :
            self.callable_func = self.save_ckd
        
        elif mode == "loadOffline":
            if not os.path.isfile(self.ckd_model_t_path):
                raise ValueError("The offline stage did not run .. please run generate mode")
            CKD_dict_loaded = np.load(self.ckd_model_t_path, allow_pickle=True)
            self.CKD_dict_loaded = dict(CKD_dict_loaded)
            self.callable_func = self.loadOffline
            
        
        # Change the loaded data everytime
        self.data_folder = get_data_folder()

        # save_dir = self.data_folder+'/cifar-100-python/train_compressed_delta_{}.npy'.format(self.delta)
        # self.DataBatch_CKD = np.load(save_dir, allow_pickle=True)

    def loadCompressedSet(self):
        save_dir = self.data_folder+'/cifar-100-python/train_compressed_delta_5.pickle'
        self.Cifar_Compress = pickle.load(open(save_dir, 'rb'))
        print("CKD ==> Compressed data is loaded")

    def __call__(self, *args, **kwargs):
        return self.callable_func(*args, **kwargs)


    def loadOffline(self):
        if os.path.isfile(self.ckd_model_t_path):
            print("Offline CKD ==> file loading")
            CKD_dict = np.load(self.ckd_model_t_path, allow_pickle=True)
            CKD_dict = dict(CKD_dict)
            print("CKD Samples ==> ", len(CKD_dict.keys()) )
        else:
            raise "Does not exist: {}".format(self.ckd_model_t_path)
        
        if self.distill != 'crd':
            self.ckd_set = CIFAR100Dataset_simple(root=self.data_folder,
                                                    download=False,
                                                    train=self.train, 
                                                    CKD_dict=CKD_dict)
        else:
            # These are the deafault the 
            self.ckd_set = CIFAR100InstanceSample(root=self.data_folder,
                                                    download=False,
                                                    train=self.train,
                                                    k=self.k_crd,
                                                    mode='exact',
                                                    CKD_dict=CKD_dict)
            
        self.ckd_loader_new = DataLoader (self.ckd_set, \
                                        batch_size=self.batch_size, \
                                        shuffle=self.shuffle, \
                                        num_workers=self.num_workers,
                                        pin_memory=True)
        return self.ckd_loader_new
    
    def save_ckd(self, process_id, lock):
        self.CKD_operator()
        if lock is not None:
            # Acquire lock
            with lock:
                self.saveDict(process_id)
        else:
            self.saveDict(process_id)

    def saveDict(self, process_id):
        CKD_dict = {}
        CKD_dict_loaded = {} 
        print("Process", process_id, "acquired lock")
        if os.path.isfile(self.ckd_model_t_path):
            print("CKD ==> file loading")
            CKD_dict_loaded = np.load(self.ckd_model_t_path, allow_pickle=True)
            CKD_dict_loaded = dict(CKD_dict_loaded) 
            CKD_dict[str(len(CKD_dict_loaded.keys()))] = self.aug_parameters_CKD
            CKD_dict.update(dict(CKD_dict_loaded))          
        else:
            CKD_dict["0"] = self.aug_parameters_CKD
    
        # Perform file write
        with open(self.ckd_model_t_path, 'wb') as f:
            np.savez(f, **CKD_dict)

        # Save updated data to the .npz file in binary write mode
        print("CKD ==> file saved : ", self.ckd_model_t_path)
        print("CKD Samples ==> ", CKD_dict.keys() )
        
        # Release lock
        print("Process", process_id, "released lock")

    def originaAcc(self):
        ckd_set = CIFAR100Dataset_simple(root=self.data_folder,
                                            download=False,
                                            train=self.train,
                                            transform=self.train_transform)
        
        # ckd_set.data = ckd_set.data[:1000]
        # ckd_set.targets = ckd_set.targets[:1000]
        
        ckd_loader = DataLoader (ckd_set, \
                                batch_size=self.batch_size,\
                                shuffle=False, \
                                num_workers=8)
        criterion_cls = nn.CrossEntropyLoss()
        teacher_acc, _, _ = validate_ckd(ckd_loader, self.model, criterion_cls)
        print('coded teacher accuracy: ',teacher_acc)

    def createContainers(self):        
        # Container that store the training data after CKD
        self.DataBatch =  torch.zeros(self.ckd_batch_size, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.DataBatch_orginal = torch.zeros(self.ckd_batch_size, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.DataBatch_aug_parameters = np.empty((self.ckd_batch_size, 6))
        

        # Container that store the data for model inference
        self.data_tensor = torch.empty(self.ckd_batch_size * (len(self.QF_range)+1), self.num_channels, self.height, self.width, dtype=torch.float32)
        self.target_tensor = torch.empty(self.ckd_batch_size * (len(self.QF_range)+1), dtype=torch.long)
        
        self.target_tmp = torch.empty((len(self.QF_range)+1), dtype=torch.long)
        self.aug_parameters = np.empty((self.ckd_batch_size * (len(self.QF_range)+1), 6))

        # num_channels, height, width = 3 , 32 , 32
        self.DataBatch_CKD =  torch.empty(self.dataset_size, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.DataBatch_ORG =  torch.empty(self.dataset_size, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.aug_parameters_CKD = np.empty((self.dataset_size, 6))

    def CKD_operator(self):
        print("CKD ==> Start")
        # self.originaAcc()
        # self.createContainers()
        # self.loadCompressedSet()

        start = time.time()
        ckd_set = CIFAR100Dataset_simple(root=self.data_folder,
                                            download=False,
                                            train=self.train)
        
        # ckd_set.data = ckd_set.data[:1000]
        # ckd_set.targets = ckd_set.targets[:1000]
        
        ckd_loader = DataLoader (ckd_set, \
                                batch_size=self.ckd_batch_size, \
                                collate_fn=custom_collate_fn, \
                                shuffle=False, \
                                num_workers=1)
        
        batch_time = AverageMeter()
        end = time.time()
        
        with torch.no_grad():
            for idx, (input ,target) in enumerate(ckd_loader):
                st = self.ckd_batch_size * idx
                ed = st +  len(input)
                self.DataBatch_CKD[st:ed], self.DataBatch_ORG[st:ed], self.aug_parameters_CKD[st:ed] = \
                    self.groupCal(input, target, len(target), st, ed)      
                batch_time.update(time.time() - end)
                end = time.time()
                # print info
                if idx % self.batch_size == 0:
                    print('CKD ==> Epoch: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} per iteration'
                           .format(idx, len(ckd_loader), batch_time=batch_time))
                    sys.stdout.flush()
        print("CKD ==> Total Time: %.2f "%(time.time() - start))

        # DEBUGING
        # just to verify the improvement of the new created set
        ckd_set = CIFAR100Dataset_simple(root=self.data_folder,
                                            download=False,
                                            train=self.train,
                                            data_ckd=[])
        
        # Load the dataset object with the new compressed data
        ckd_set.data_ckd = self.DataBatch_CKD
        ckd_set.data = self.DataBatch_ORG
        
        ckd_loader = DataLoader (ckd_set, \
                             batch_size=self.batch_size, \
                             shuffle=self.shuffle, \
                             num_workers=self.num_workers,
                             pin_memory=True)
        
        criterion_cls = nn.CrossEntropyLoss()
        coded_teacher_acc, _, _ = validate_ckd(ckd_loader, self.model, criterion_cls)
        print('coded teacher accuracy: ',coded_teacher_acc)
    
    def online_generation(self):
        self.CKD_operator()
        ckd_set = CIFAR100Dataset_simple(root=self.data_folder,
                                            download=False,
                                            train=self.train)
        
        ckd_set.data_ckd = self.DataBatch_CKD
        ckd_set.data = self.DataBatch_ORG
        
        ckd_loader_new = DataLoader (ckd_set, \
                                        batch_size=self.batch_size, \
                                        shuffle=self.shuffle, \
                                        num_workers=self.num_workers,
                                        pin_memory=True)
        
        return ckd_loader_new


    def groupCal(self, input, target, size, st, ed) :
        for idx in range(size):
            # self.generateCompress(input[idx], idx) # I want to parallalize it .. <no significant improvement>
            self.getCompressed(input[idx], idx, st, ed) 
            data_window = idx * (len(self.QF_range)+1)
            # Fill the tensor with the integer number N
            self.target_tmp.fill_(target[idx])
            self.target_tensor[data_window:data_window+(len(self.QF_range)+1)] = self.target_tmp
        
        data_tensor_window = size * (len(self.QF_range)+1)
        if torch.cuda.is_available():
            if self.data_tensor.device.type != 'cuda': self.data_tensor = self.data_tensor.cuda()
            if self.target_tensor.device.type != 'cuda': self.target_tensor  = self.target_tensor.cuda()

        output = self.model(self.data_tensor[:data_tensor_window].float())
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(self.target_tensor[:data_tensor_window].view(1, -1).expand_as(pred))
        correct_indexes = torch.nonzero(correct[0]).squeeze(1)

        ref_kl =  torch.zeros(correct_indexes.size(0), output.size(1), dtype=torch.float32).cuda()
        window = (len(self.QF_range) + 1)
        count_correct_per_image = torch.zeros(len(target), dtype=torch.int)
        
        # copy refrence proability for the original sample
        for idx in range(correct_indexes.size(0)):
            # Copy the original proability vector
            ref_index = (correct_indexes[idx].item() // window)
            count_correct_per_image[ref_index] +=1
            ref_index = (window) * ref_index
            ref_kl[idx] = output[ref_index] 

        # Truely classified samples only
        output_ = output[correct_indexes]
        kl_divs = F.kl_div(F.log_softmax(ref_kl, dim=1), F.softmax(output_, dim=1), reduction='none').sum(dim=1)

        # Build the batch with the selected images
        st_index = 0
        for idx in range(len(target)):
            if count_correct_per_image[idx] != 0:
                end_index = st_index + count_correct_per_image[idx].item()                
                tmp_kl = kl_divs[st_index:end_index] # moving window
                selected_img = st_index + torch.argmax(tmp_kl)
                self.DataBatch[idx] = self.data_tensor[correct_indexes[selected_img].item()].float()
                self.DataBatch_aug_parameters[idx] = self.aug_parameters[correct_indexes[selected_img].item()]
                org, _ = transform_apply(Image.fromarray(input[idx]), self.test_transform, \
                                                                 aug_parameters=self.aug_parameters[correct_indexes[selected_img].item()])
                self.DataBatch_orginal[idx] = org.float()
                st_index = end_index
            else:
                ref_index = idx * window
                self.DataBatch[idx] = self.data_tensor[ref_index].float()
                self.DataBatch_aug_parameters[idx] = self.aug_parameters[ref_index]
                org, _ = transform_apply(Image.fromarray(input[idx]), self.test_transform, aug_parameters=self.aug_parameters[ref_index])
                self.DataBatch_orginal[idx] = org.float()

        return self.DataBatch[:size], self.DataBatch_orginal[:size],  self.DataBatch_aug_parameters[:size]
            
    def getCompressed(self, input, index, st, ed) :
        # checkSimilar(Image.fromarray(input), self.Cifar_Compress[index+st][-1])
        input_aug, aug_parameters = self.transform(self.Cifar_Compress[index+st][-1], self.test_transform)
        aug_parameters.insert(0, -1)
        data_window = index * (len(self.QF_range)+1)
        
        self.data_tensor[data_window] = input_aug 
        self.aug_parameters[data_window] = aug_parameters 

        for idx, qf in enumerate(self.QF_range):
            image = self.Cifar_Compress[index+st][qf]
            image, aug_parameters = self.transform(image, self.test_transform)
            aug_parameters.insert(0, qf)
            self.data_tensor[data_window+idx+1] = image
            self.aug_parameters[data_window+idx+1] = aug_parameters


    
    def generateCompress(self, input, idx):
        try:
            input = Image.fromarray(input)
        except:
            pass 

        if input.mode == "RGB":
            pass
        elif input.mode == "RGBA":
            # Convert the RGBA image to RGB and alpha channel tensors separately
            input = input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        
        input_aug, aug_parameters = self.transform(input, self.test_transform)
        aug_parameters.insert(0, -1)

        data_window = idx * (len(self.QF_range)+1)
        self.data_tensor[data_window] = input_aug 
        self.aug_parameters[data_window] = aug_parameters 

        buffer = BytesIO()
        for idx, qf in enumerate(self.QF_range):
            input.save(buffer, 'JPEG', quality=qf, subsampling=0)
            image = Image.open(buffer).convert("RGB")
            image, aug_parameters = self.transform(image, self.test_transform)
            aug_parameters.insert(0, qf)
            self.data_tensor[data_window+idx+1] = image
            self.aug_parameters[data_window+idx+1] = aug_parameters
            buffer.seek(0)
            buffer.truncate()
    

    def groupCal_(self, input, target, size):
        # without CKD
        for idx in range(size):
            try:
                input[idx] = Image.fromarray(input[idx])
            except:
                pass 

            if input[idx].mode == "RGB":
                pass
            elif input[idx].mode == "RGBA":
                # Convert the RGBA image to RGB and alpha channel tensors separately
                input[idx] = input[idx].convert("RGB")
            else:
                raise ValueError(f"Unsupported image mode: {image.mode}")
            input[idx], aug_parameters = self.transform(input[idx], self.test_transform)
            self.DataBatch[idx] = input[idx].float()
        return self.DataBatch[:size] 

def validate_ckd(val_loader, model, criterion):
    """validation"""
    """ here we are generating before inference """
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    start_time = time.time() 
    # switch to evaluate mode
    with torch.no_grad():
        end = time.time()
        for idx, (data) in enumerate(val_loader):
            if  len(data) == 3:
                # Normal Teacher for training data
                input, target, index = data
            elif len(data) == 4:
                # Coded Teacher for training data
                orignal , input, target, index = data
            else:
                # Original Input for validation data
                input, target = data

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("CKD outter Elapsed time:", elapsed_time)
    return top1.avg, top5.avg, losses.avg