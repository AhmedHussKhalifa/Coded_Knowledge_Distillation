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


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar100_ckd import get_cifar100_dataloaders_CKD
from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import validate
from helper.pretrain import init

import numpy as np
from helper.util import AverageMeter, accuracy
import torch.nn.functional as F

from dataset.ckd_selector import CKD_selector_parallel
import matplotlib.pyplot as plt
import seaborn as sns

def cal_correlation(x, coef=False):
    '''Calculate the correlation matrix for a pytorch tensor.
    Input shape: [n_sample, n_attr]
    Output shape: [n_attr, n_attr]
    Refer to: https://github.com/pytorch/pytorch/issues/1254
    '''
    # calculate covariance matrix
    y = x - x.mean(dim=0)
    c = y.t().mm(y) / (y.size(0) - 1)
    
    if coef:
        # normalize covariance matrix
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        c = torch.clamp(c, -1.0, 1.0)
    return c

def get_class_corr(loader, model):
    model.eval().cuda()
    logits = 0
    n_batch = len(loader)
    with torch.no_grad():
        for ix, data in enumerate(loader):
            input = data[0]
            # print('[%d/%d] -- forwarding' % (ix, n_batch))
            input = input.float().cuda()
            if type(logits) == int:
                logits = model(input) # [batch_size, n_class]
            else:
                logits = torch.cat([logits, model(input)], dim=0)
    # Use numpy:
    # logits -= logits.mean(dim=0)
    # logits = logits.data.cpu().numpy()
    # corr = np.corrcoef(logits, rowvar=False)

    # Use pytorch
    corr = cal_correlation(logits, coef=True)
    return corr

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=400, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
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

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4.0, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--ckd', type=str, default='', help='trial id')

    
    parser.add_argument('--K_samples', default=1000, type=int, help='feature dimension')
    
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

    if opt.ckd == 'ckd':
        opt.model_name = 'S_CKD:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_T:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.kd_T, opt.trial)    
        print(opt.model_name)
    elif opt.ckd == 'TALD':
        opt.model_name = 'S_TALD:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                    opt.gamma, opt.alpha, opt.beta, opt.trial)
    else:
        opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
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
    if "S_TALD:" in model_path:
        start_index = model_path.find("S_TALD:") + len("S_TALD:")
    elif "S:" in model_path:
        start_index = model_path.find("S:") + len("S:")
    end_index = model_path.find("_T:")
    return model_path[start_index:end_index]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model, model_t


def load_student(model_path, n_cls):
    print('==> loading student model')
    model_s = get_student_name(model_path)
    model_path = "./save/student_model/"+ model_path +"/" + model_s +"_best.pth"
    model = model_dict[model_s](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('==> done')
    return model, model_s

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

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.ckd in ['ckd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_CKD(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,
                                                                        shuffle=False)
            n_cls = 100
        else:
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                                num_workers=opt.num_workers,
                                                                                k=opt.nce_k,
                                                                                mode=opt.mode,
                                                                                shuffle=False)
            else:
                train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                            num_workers=opt.num_workers,
                                                                            is_instance=True,
                                                                            shuffle=False)
            n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    student_dir = opt.model_name

    model_t, t_name = load_teacher(opt.path_t, n_cls)
    model_s, s_name = load_student(student_dir, n_cls)

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
        model_s = model_s.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    # teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    # print('teacher accuracy: ', teacher_acc)

    # validate student accuracy
    student_acc, _, _ = validate(val_loader, model_s, criterion_cls, opt)
    print('student  accuracy: ', student_acc)


    # alpha_range = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    alpha_range = np.array([1])
    student_acc, _, _, renyi_allset, GT_prop_allset = epoch_analysis(train_loader, model_s, criterion_cls, opt, alpha_range)
    # print(alpha_range)
    print("Renyi Entropy : ", renyi_allset.mean(dim=0), renyi_allset.std(dim=0))

    print("Average Propability (mean/std) : ", GT_prop_allset.mean(dim=0), GT_prop_allset.std(dim=0))
    
    print('Student training accuracy: ', student_acc)


    dir_path = "./experiments/analysis/"
    if opt.ckd == "ckd":
        prefix = "c" + opt.distill
    elif opt.ckd == "TALD":
        prefix = "TALD+"+ opt.distill
    else:
        prefix = opt.distill

    file_path = os.path.join(dir_path , prefix+ "_E_P.txt") 

    # Check if the directory exists
    if not os.path.exists(dir_path):
        print("Directory does not exist. Creating directory...")
        os.makedirs(dir_path)

    exp_txt = open(file_path, 'a+') #
    exp_txt.write("\n" + opt.model_name +"\n") # Write some text
    line = "&\t"+ str(np.round(renyi_allset.mean(dim=0).item(),4)) + " ($\pm$ " + str(np.round(renyi_allset.std(dim=0).item(),4)) + ")\t&\t" + str(np.round(GT_prop_allset.mean(dim=0).item(),4)) + " ($\pm$ " + str(np.round(GT_prop_allset.std(dim=0).item(),4))  + ")\t" 
    exp_txt.write(line) # Write some text
    exp_txt.close() # Close the file

    exit(0)

    if opt.ckd in ['ckd']:
        opt.ckd_model_t = get_CKD_path(opt.path_t)
        ckd_selector = None
        train_flag = True
        dataset_size = len(train_loader.dataset)
        ckd_selector = CKD_selector_parallel(dataset_size, train=train_flag, batch_size=opt.batch_size, \
                                            num_workers=opt.num_workers, mode="loadOffline",\
                                            ckd_model_t_path= opt.ckd_model_t, \
                                            shuffle=False, \
                                            distill=opt.distill, k = opt.nce_k)


        # To check if we improve the training responses [can be commented]
        # ckd_loader = ckd_selector()
        # teacher_acc, _, _ = validate_ckd(ckd_loader, model_t, criterion_cls, opt)
        # print('coded teacher accuracy: ', teacher_acc)
        # del ckd_loader
        # train_loader = ckd_selector()
        
        # train_analysis(ckd_selector, model_t, model_s, opt, dir_path+ "/"+prefix)
        train_analysis_epochs(ckd_selector, model_t, model_s, opt, dir_path+ "/"+prefix)
    
    elif opt.ckd in ['TALD']:
        print("Different settings needed")
    else:
        # train_analysis(train_loader, model_t, model_s, opt, dir_path+ "/"+prefix)
        train_analysis_epochs(train_loader, model_t, model_s, opt, dir_path+ "/"+prefix)

    # corr = get_class_corr(train_loader, model_t).cpu().numpy()
    # print(prefix, corr.mean())


    # plt.imshow(corr, cmap='hot', interpolation='nearest')
    # plt.savefig(os.path.join(dir_path , prefix+"_matplot_corr.png"), dpi=900)

    # ax = sns.heatmap(corr, linewidth=0.5)
    # plt.savefig(os.path.join(dir_path , prefix+"_seaborn_corr.png"), dpi=900)

    # file_path = os.path.join(dir_path , prefix + "_corr.txt") 
    # exp_txt = open(file_path, 'a+') #
    # exp_txt.write(opt.model_name +"\t") # Write some text
    # line = str(corr.mean()) + '\n'
    # exp_txt.write(line) # Write some text
    # exp_txt.close() # Close the file

        
def get_CKD_path(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1]
    return model_path.replace(segments, "ckd_<train/val>.npz")



def train_analysis_epochs(ckd_selector, model_t, model_s, opt, data_path):
    """vanilla training"""
    model_t.train()
    model_s.train()
    num_epochs = 10
    num_rows = 50000  # Number of rows
    num_cols = 0 # Number of columns
    sum_output = np.zeros((100, num_cols))
    avg_loss = np.array([])
    K_samples = opt.K_samples


    if opt.ckd in ['ckd']:
        train_loader = ckd_selector()
    else:
        train_loader = ckd_selector
    
    ce_output = np.zeros((num_rows, num_cols))

    for epoch in range(1, num_epochs + 1):
    # for epoch in range(1, 2):
        cum_k_samples = 0
        ce_output_tmp  = np.array([])
        sum_output_tmp = np.zeros(100)
        print("Epoch : ", epoch)
        for idx, data in enumerate(train_loader):
            if opt.ckd in ['ckd']:
                input, input_ckd, target, index = data
                input_ckd = input_ckd.cuda()
            else: 
                input, target, index = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # ===================forward=====================
            if opt.ckd in ['ckd']:
                logit_t = model_t(input_ckd).detach()
            else:
                logit_t = model_t(input).detach()
                
            logit_s = model_s(input).detach()

            output_t = F.softmax(logit_t, dim=1)
            output_s = F.softmax(logit_s, dim=1)

            sum_output_tmp += np.sum(output_t.cpu().numpy() , axis=0)

            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_s, F.softmax(logit_t, dim=1)).cpu().numpy()
            
            # Only for the Teacher
            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_t, F.softmax(logit_t, dim=1)).cpu().numpy()
            # tmp = -torch.sum(output_t * F.log_softmax(logit_t, dim=1), dim=1).cpu().numpy()
            # tmp = -torch.sum(output_t * F.log_softmax(logit_s, dim=1), dim=1).cpu().numpy()

            '''
            Here we need to have student for the CKD
            '''    
            # tmp = -1 * (output_t * F.log_softmax(logit_s, dim=1)).cpu().numpy() 
            tmp = nn.CrossEntropyLoss(reduction='none')(logit_s, F.softmax(logit_t, dim=1)).cpu().numpy()           
            '''
            Here we assumed that student will preform as the teacher
            '''
            # tmp = -1 * (output_t * F.log_softmax(logit_t, dim=1)).cpu().numpy() 
            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_t, F.softmax(logit_t, dim=1)).cpu().numpy()   
            
            # Dot-Prodcut    
            ce_output_tmp = np.append(ce_output_tmp, tmp)  
 
            # # This part where we start 
            cum_k_samples += len(output_t)
            if cum_k_samples > K_samples:
                break


        avg_loss = np.append(avg_loss, np.mean(ce_output_tmp))  
        sum_output_tmp /= cum_k_samples
        ce_output_tmp = np.expand_dims(ce_output_tmp, axis=1)
        sum_output_tmp = np.expand_dims(sum_output_tmp, axis=1)

        sum_output = np.hstack([sum_output, sum_output_tmp])
        # ce_output = np.hstack([ce_output, ce_output_tmp])
        if opt.ckd in ['ckd']:
            ckd_selector.ckd_set.incermentEpoch()

    covariance_matrix_propability = np.cov(sum_output, rowvar=True)
    covariance_sum_propability = np.mean(covariance_matrix_propability)
    # var_propability = np.var(sum_output)
    # std_propability = np.std(np.var(sum_output, axis=1))

    std_propability = np.var(np.diagonal(np.cov(sum_output, rowvar=True)))

    var_loss = np.var(avg_loss)

    print("Covariance of sum of Proability [Over {} Sequence]:", covariance_sum_propability)
    print("Variance Probability [Over {} Sequence]:".format(num_epochs), std_propability)
    print("Variance Loss [Over {} Sequence]:".format(num_epochs), var_loss)


    file_path = data_path + "_corr_innerloop.txt"
    exp_txt = open(file_path, 'a+') #
    exp_txt.write(opt.model_name +"\t" + str(K_samples) +"\t") # Write some text
    line = str(covariance_sum_propability) + "\t" + str(std_propability) + "\t" + str(var_loss)  + '\n'
    exp_txt.write(line) # Write some text
    exp_txt.close() # Close the file





def train_analysis(ckd_selector, model_t, model_s, opt, data_path):
    """vanilla training"""
    model_t.train()
    model_s.train()
    num_epochs = 10
    num_rows = 50000  # Number of rows
    num_cols = 0 # Number of columns
    sum_output = np.zeros((100, num_cols))
    avg_loss = np.array([])
    K_samples = opt.K_samples

    if opt.ckd in ['ckd']:
        train_loader = ckd_selector()
    else:
        train_loader = ckd_selector
    
    ce_output = np.zeros((num_rows, num_cols))

    # for epoch in range(1, num_epochs + 1):
    for epoch in range(1, 5):
        cum_k_samples = 0
        ce_output_tmp  = np.zeros((0, 100))
        sum_output_tmp = np.zeros(100)
        print("Epoch : ", epoch)
        for idx, data in enumerate(train_loader):
            if opt.ckd in ['ckd']:
                input, input_ckd, target, index = data
                input_ckd = input_ckd.cuda()
            else: 
                input, target, index = data

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # ===================forward=====================
            if opt.ckd in ['ckd']:
                # logit_t = model_t(input_ckd).detach()
                logit_t = model_t(input).detach()
            else:
                logit_t = model_t(input).detach()
                
            logit_s = model_s(input).detach()

            output_t = F.softmax(logit_t, dim=1)
            output_s = F.softmax(logit_s, dim=1)

            sum_output_tmp += np.sum(output_t.cpu().numpy() , axis=0)

            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_s, F.softmax(logit_t, dim=1)).cpu().numpy()
            
            # Only for the Teacher
            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_t, F.softmax(logit_t, dim=1)).cpu().numpy()
            # tmp = -torch.sum(output_t * F.log_softmax(logit_t, dim=1), dim=1).cpu().numpy()
            # tmp = -torch.sum(output_t * F.log_softmax(logit_s, dim=1), dim=1).cpu().numpy()

            '''
            Here we need to have student for the CKD
            '''    
            tmp = -1 * (output_t * F.log_softmax(logit_s, dim=1)).cpu().numpy() 
            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_s, F.softmax(logit_t, dim=1)).cpu().numpy()           
            '''
            Here we assumed that student will preform as the teacher
            '''
            # tmp = -1 * (output_t * F.log_softmax(logit_t, dim=1)).cpu().numpy() 
            # tmp = nn.CrossEntropyLoss(reduction='none')(logit_t, F.softmax(logit_t, dim=1)).cpu().numpy()   

            ce_output_tmp = np.append(ce_output_tmp, tmp, axis=0) 
 
            # # This part where we start 
            # cum_k_samples += len(output_t)
            # if cum_k_samples > K_samples:
            #     break

        if opt.ckd in ['ckd']:
            ckd_selector.ckd_set.incermentEpoch()
        
        # breakpoint()
        ce_output = np.hstack([ce_output, ce_output_tmp.mean(axis=1).reshape([-1,1])])

    covariance_matrix_per_sample  = np.cov(ce_output, rowvar=True)
    average_covariance_per_sample  = np.mean(covariance_matrix_per_sample)

    print("Average Covariance per sample [Over {} Sequence]:".format(num_epochs), average_covariance_per_sample)


    file_path = data_path + "_corr_innerloop.txt"
    exp_txt = open(file_path, 'a+') #
    exp_txt.write(opt.model_name) # Write some text

    line = "\t" + str(average_covariance_per_sample) + '\n'
    exp_txt.write(line) # Write some text
    exp_txt.close() # Close the file



if __name__ == '__main__':
    main()