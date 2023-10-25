"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100_ckd import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_CKD
import sys

from io import BytesIO
from helper.loops_ckd import train_distill as train, validate, validate_ckd 
from helper.pretrain import init
from torch.utils.data import DataLoader
from dataset.ckd_selector import CKD_selector_parallel
from helper.util import AverageMeter, accuracy
import pickle
from torchvision import datasets, transforms
import copy
import multiprocessing
import numpy as np
from dataset.ckd_selector import CIFAR100Dataset_simple

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--ckd', type=str, default='1', help='trial id')
    parser.add_argument('--delta', type=int, default=5, help='trial id')

    opt = parser.parse_args()

    opt.model_t = get_teacher_name(opt.path_t)
    opt.ckd_model_t = get_CKD_path(opt.path_t)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def get_CKD_path(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1]
    return model_path.replace(segments, "ckd_<train/val>.npz")


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('==> done')
    return model


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

opt = parse_option()

if opt.dataset == 'cifar100':
    n_cls = 100
else:
    raise NotImplementedError(opt.dataset)
# model
model_t = load_teacher(opt.path_t, n_cls)

# dataloader
if opt.dataset == 'cifar100':
    if opt.ckd in ['ckd']:
        is_instance = False
        data = get_cifar100_dataloaders_CKD(batch_size=opt.batch_size,
                                                num_workers=opt.num_workers,
                                                is_instance=is_instance, model_t=model_t)
        if is_instance:
            train_loader, val_loader, n_data = data
        else:
            train_loader, val_loader = data

    else:
        if opt.distill in ['crd']:
            train_loader, val_loader = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers,
                                                                            k=opt.nce_k,
                                                                            mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
else:
    raise NotImplementedError(opt.dataset)

criterion_cls = nn.CrossEntropyLoss()
data = torch.randn(2, 3, 32, 32)
model_t.eval()


def main_CKD_TrainVal():
    if torch.cuda.is_available():
        model_t.cuda()
        cudnn.benchmark = True

    # validate coded teacher accuracy [working]
    if opt.ckd in ['ckd'] and False:
        print("==> CKD validation")
        
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('teacher accuracy: ', teacher_acc)

        train_flag = False
        batch_size=400
        ckd_selector = CKD_selector_parallel( dataset_size=len(val_loader.dataset), model=model_t.eval(), delta=opt.delta, train=train_flag, \
                                                batch_size=batch_size, num_workers=opt.num_workers,\
                                                mode="online") #"generate"
        ckd_loader = ckd_selector()
        teacher_acc, _, _ = validate_ckd(ckd_loader, model_t, criterion_cls, opt)
        print('coded teacher accuracy: ', teacher_acc)
    
    # Training
    if opt.ckd in ['ckd']:
        print("==> CKD Training")
        teacher_acc, _, _ = validate_ckd(train_loader, model_t, criterion_cls, opt)
        print('Teacher accuracy: ', teacher_acc)
        
        train_flag = True
        batch_size=400
        ckd_selector = CKD_selector_parallel( dataset_size=len(train_loader.dataset), model=model_t.eval(), delta=opt.delta, train=train_flag, \
                                                batch_size=batch_size, num_workers=opt.num_workers,\
                                                mode="online") #"generate"
        ckd_loader = ckd_selector()
        teacher_acc, _, _ = validate_ckd(ckd_loader, model_t, criterion_cls, opt)
        print('coded teacher accuracy: ', teacher_acc)

def genrate(process_id=0, lock=None):
    print("Process", multiprocessing.current_process().name)

    if torch.cuda.is_available():
        model_t.cuda()
        cudnn.benchmark = True

    train_flag = True
    ckd_batch_size=100
    ckd_selector = CKD_selector_parallel( dataset_size=len(train_loader.dataset), model=model_t.eval(), delta=opt.delta, train=train_flag, \
                                            ckd_batch_size=ckd_batch_size, num_workers=opt.num_workers,\
                                            mode="save_ckd", ckd_model_t_path= opt.ckd_model_t) #"save_ckd"
    ckd_selector.loadCompressedSet()
    ckd_selector.createContainers()
    for i in range(0,13):
        ckd_selector(process_id, lock)

def main_multiprocess():
    num_processes = 1
    lock = multiprocessing.Lock()
    processes = []

    # Create and start processes
    for i in range(num_processes):
        p = multiprocessing.Process(target=genrate, args=(i, lock))
        processes.append(p)
        p.start()

    # Join processes
    for p in processes:
        p.join()

    print("All processes completed.")

def generateCompress(input, QF_range):
    comp_img = {}

    if input.mode == "RGB":
        pass
    elif input.mode == "RGBA":
        # Convert the RGBA image to RGB and alpha channel tensors separately
        input = input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image mode: {input.mode}")
    
    buffer = BytesIO()
    comp_img[-1] = copy.copy(input)
    for idx, qf in enumerate(QF_range):
        org_copy = copy.copy(input)
        org_copy.save(buffer, 'JPEG', quality=qf, subsampling=0)
        image = Image.open(buffer).convert("RGB")
        comp_img[qf] = image
        buffer.seek(0)
        buffer.truncate()
    return comp_img

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths


def main_generateAllCompressed():
    data_folder = get_data_folder()
    start = time.time()
    delta = 5
    QF_range = range(0,100+delta, delta)

    ckd_set = datasets.CIFAR100(root=data_folder,
                                download=True,
                                train=True)
    
    
    train_loader = DataLoader(ckd_set,
                              batch_size=1,
                              collate_fn=custom_collate_fn, 
                              shuffle=False,
                              num_workers=1)
    
    batch_time = AverageMeter()
    end = time.time()
    DataBatch_CKD = []
    with torch.no_grad():
        for idx, (input ,target) in enumerate(train_loader):
            tmp = generateCompress(input[0], QF_range)
            # breakpoint()
            DataBatch_CKD.append(tmp)      
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % 100 == 0:
                print('CKD ==> Epoch: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} per iteration'
                        .format(idx, len(train_loader), batch_time=batch_time))
                sys.stdout.flush()
    
    print("CKD ==> Total Time: %.2f "%(time.time() - start))
    # Save the dataset using pickle
    
    # save_dir = data_folder + '/cifar-100-python/train_compressed_delta_5.npy'
    # print(save_dir)
    # np.save(save_dir, DataBatch_CKD)

    save_dir = data_folder+'/cifar-100-python/train_compressed_delta_5.pickle'
    with open(save_dir, 'wb') as f:
        pickle.dump(DataBatch_CKD, f)

if __name__ == '__main__':
    # main_CKD_TrainVal()
    genrate()
    # main_generateAllCompressed()
    # main_multiprocess()
