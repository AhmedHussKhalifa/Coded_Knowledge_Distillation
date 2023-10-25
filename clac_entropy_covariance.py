"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger.tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100_ckd import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_CKD

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops_ckd import train_distill as train, validate, validate_ckd 
from helper.pretrain import init

from helper.util import AverageMeter, accuracy

import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

from dataset.ckd_selector import CKD_selector_parallel

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=4000, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--ckd', type=str, default='', help='trial id')
    parser.add_argument('--delta', type=int, default=5, help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    if opt.distill == 'kd':
        opt.model_name = 'S_CKD:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_T:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.kd_T, opt.trial)
    elif opt.distill =='hint':
        opt.model_name = 'S_CKD:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_layer:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.hint_layer, opt.trial)
    else:
        opt.model_name = 'S_CKD:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def get_student_name(model_path):
    """parse student name"""
    if "S_CKD:" in model_path:
        start_index = model_path.find("S_CKD:") + len("S_CKD:")
    elif "S:" in model_path:
        start_index = model_path.find("S:") + len("S:")
    end_index = model_path.find("_T")
    return model_path[start_index:end_index]


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

def load_student(model_path, n_cls):
    print('==> loading student model')
    model_t = get_student_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('==> done')
    return model


def entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy




def renyi_entropy(output, alpha_range):
    prep = F.softmax(output, dim=1)
    renyi_output = torch.empty(len(alpha_range), output.size()[0], dtype=torch.float32)
    for idx, alpha in enumerate(alpha_range):
        if alpha != 0 and alpha != 1:
            renyi_output[idx] = (1 / (1-alpha) )* torch.log(torch.pow(prep, alpha).sum(dim=1))
        elif alpha == 0:
            renyi_output[idx] = torch.log(output.size()[1])
        elif alpha == 1:
            renyi_output[idx] = entropy(output)
      
    return renyi_output.transpose(0, 1)


def epoch_analysis(val_loader, model, criterion, opt, alpha_range):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    renyi_allset = torch.empty(len(val_loader.dataset), len(alpha_range), dtype=torch.float32)
    GT_prop_allset = torch.empty(len(val_loader.dataset), dtype=torch.float32)

    with torch.no_grad():
        end = time.time()
        for idx, (data) in enumerate(val_loader):

            if len(data) == 3:
                input, target, index = data
            else:
                input, target = data
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)

            st =  val_loader.batch_size * idx
            ed = st +  len(input)
            renyi_allset[st:ed] = renyi_entropy(output, alpha_range)
            
            tmp = F.softmax(output, dim=1)
            GT_prop_allset[st:ed]  = tmp[torch.arange(tmp.size(0)), target]
            # _, pred = output.topk(1, 1, True, True)
            # pred = pred.t()
            # correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg, renyi_allset, GT_prop_allset

def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.dataset == 'cifar100':
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)
    
    
    breakpoint()
    # model
    # save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

    ################################################################ 

    # model_dir = 'resnet8x4_resnet32x4/CKD/S_CKD:resnet8x4_T:resnet32x4_cifar100_kd_r:1_a:0.96_b:0.04_10/'
    # model_dir = 'resnet8x4_resnet32x4/KD/S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.9_a:0.1_b:0.0_16/'
    # model_dir = 'resnet8x4_resnet32x4/TALD/S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.1_a:0.9_b:0.0_lossald1_0.01/'



    # model_dir = 'MobileNetV2_ResNet50/CKD/S_CKD:MobileNetV2_T:ResNet50_cifar100_kd_r:1.0_a:0.6_b:0.0_T:11.0_16/' # 68.41
    # model_dir = 'MobileNetV2_ResNet50/CKD/S_CKD:MobileNetV2_T:ResNet50_cifar100_kd_r:1.0_a:0.74_b:0.0_T:9.0_15/' # 67.80999755859375


    # model_dir = 'MobileNetV2_vgg13/CKD/S_CKD:MobileNetV2_T:vgg13_cifar100_kd_r:1.0_a:0.8_b:0.0_T:18.0_15/'


    model_dir = 'resnet20_resnet56/CKD/S_CKD:resnet20_T:resnet56_cifar100_kd_r:1.0_a:0.35_b:0.0_T:12.0_16/'
    # model_dir = 'resnet20_resnet56/CKD/S_CKD:resnet20_T:resnet56_cifar100_kd_r:1.0_a:0.4_b:0.0_T:8.0_16/'

    ################################################################

    # model_dir = 'resnet8x4_resnet32x4/KD/S:resnet8x4_T:resnet32x4_cifar100_kd_r:0.9_a:0.1_b:0.0_999/'
    # model_dir = 'MobileNetV2_vgg13/KD/S:MobileNetV2_T:vgg13_cifar100_kd_r:0.9_a:0.1_b:0.0_999/'
    # model_dir = 'MobileNetV2_ResNet50/KD/S:MobileNetV2_T:ResNet50_cifar100_kd_r:0.9_a:0.1_b:0.0_999/'
    # model_dir = 'resnet20_resnet56/KD/S:resnet20_T:resnet56_cifar100_kd_r:0.9_a:0.1_b:0.0_999/'


    save_file = '/home/ahamsala/PROJECT_AH/model_KD/'
    if "MobileNetV2" in model_dir:
        student_dir = save_file + model_dir + 'MobileNetV2_best.pth'
    elif "resnet8x4" in model_dir:
        student_dir = save_file + model_dir + 'resnet8x4_best.pth'
    elif "resnet20" in model_dir:
        student_dir = save_file + model_dir + 'resnet20_best.pth'

    model_t = load_teacher(opt.path_t, n_cls)
    
    get_teacher_name(opt.path_t)
    model_s = load_student(student_dir, n_cls)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers,
                                                                            k=opt.nce_k,
                                                                            mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
    else:
        raise NotImplementedError(opt.dataset)
    
    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)
   
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate student accuracy
    criterion_cls = nn.CrossEntropyLoss()
    alpha_range = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])

    print(model_dir)
    test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
    print('Student validation accuracy: ', test_acc)

    student_acc, _, _, renyi_allset, GT_prop_allset = epoch_analysis(train_loader, model_s, criterion_cls, opt, alpha_range)
    print("Renyi Entropy")
    print(alpha_range)
    print(renyi_allset.mean(dim=0))
    print(renyi_allset.var(dim=0))

    print("Average Propability")
    print(GT_prop_allset.mean(dim=0))
    print(GT_prop_allset.var(dim=0))
    
    print('Student training accuracy: ', student_acc)
    # load the new dataset class with the dataloader 

    dir_path = "./experiments_Entropy_Calc/" + opt.model_s + "/"
    file_path = os.path.join(dir_path , opt.distill + ".txt") 
    # Check if the directory exists
    if not os.path.exists(dir_path):
        print("Directory does not exist. Creating directory...")
        os.makedirs(dir_path)
    exp_txt = open(file_path, 'a+')
    exp_txt.write(model_dir+ "\t" + str(student_acc.item()) + "\n") # Write some text
    exp_txt.close() # Close the file
    # renyi_dir = save_file + model_dir + ("S_CKD" if "S_CKD:" in model_dir else "KD")+ ".pt"
    # torch.save(renyi_allset, renyi_dir)



if __name__ == '__main__':
    main()
