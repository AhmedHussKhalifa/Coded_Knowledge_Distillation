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
from PIL import Image
from helper.util import AverageMeter, accuracy
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

from multiprocessing import Pool
from multiprocessing import set_start_method

import torch.multiprocessing as mp
import time
from functools import partial



class CKD:
    def __init__(self, model, delta, train=False, test_transform=None):
        # set_start_method('spawn')
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        self.delta = delta
        self.QF_range = range(0,100+self.delta, self.delta)
        # self.QF_range = []

        self.num_channels, self.height, self.width = 3 , 32 , 32

        if train:
            self.transform = self.transform_new
        else:
            self.transform = self.transform_apply

        # Create a tensor with the desired shape
        self.data_tensor = torch.zeros(len(self.QF_range)+1, self.num_channels, self.height, self.width, dtype=torch.float32).cuda()
        self.target_tensor = torch.empty(len(self.QF_range)+1, dtype=torch.long)
        self.test_transform = test_transform
        
    
    def __call__(self, input, target):

        input = Image.fromarray(input)
        if input.mode == "RGB":
            pass
        elif input.mode == "RGBA":
            # Convert the RGBA image to RGB and alpha channel tensors separately
            input = input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        if torch.cuda.is_available():
            if self.data_tensor.device.type != 'cuda': self.data_tensor = self.data_tensor.cuda()
            if self.target_tensor.device.type != 'cuda': self.target_tensor  = self.target_tensor.cuda()

        input_aug, aug_parameters = self.transform(input, self.test_transform)
        self.target_tensor.fill_(target)
        self.data_tensor[0]=input_aug
        buffer = BytesIO()
        # Generate all possible compressed images
        for idx, qf in enumerate(self.QF_range):
            
            # input.save(buffer, "JPEG", quality=5)
            # input.save(buffer, 'JPEG', quality=qf, subsampling=0)
            input.save(buffer, 'JPEG', quality=qf, 
                                    # exif =  image.info['exif'],
                                    # optimize=False,
                                    # qtable=image.quantization,
                                    subsampling=0,
                                    # subsampling=JIP.get_sampling(input)
                                    )
            image = Image.open(buffer).convert("RGB")
            image, aug_parameters = self.transform(image, self.test_transform)
            # data_tensor = torch.cat((data_tensor, image.unsqueeze(0)), dim=0)
            
            self.data_tensor[idx+1] = image
            # self.data_aug_parameters.append(aug_parameters)
        
        # self.data_tensor = self.data_tensor.float()
        self.data_tensor = self.data_tensor.float() # this is a must to get a valid output
        output = self.model(self.data_tensor)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(self.target_tensor.view(1, -1).expand_as(pred))
        correct_indexes = torch.nonzero(correct[0]).squeeze(1)

        # return input_aug.cuda(), target

        # Return the original Image if none of the compressed images are correct
        if len(correct_indexes) == 0:
            return input_aug.cuda(), target

        if 0 not in correct_indexes:
            sample_indexes = torch.cat([torch.tensor([0]).cuda(), correct_indexes], dim=0)
        else:
            sample_indexes = correct_indexes
        
        # Convert the reference tensor to a probability distribution by applying the softmax function
        output = output[sample_indexes]
        output = F.softmax(output, dim=1)
        kl_values = self.kl_divergence(output[0], output)
        correct_probs, correct_index = torch.max(output, dim=1)
        correct_probs = correct_probs[correct_index == target]
        kl_values = kl_values[correct_index == target]

        if kl_values.numel() > 0:
            return self.data_tensor[correct_indexes[torch.argmax(kl_values)]], target
        else:
            return self.data_tensor[0], target
        

    def transform_new(self, image, test_transform):
        crop = transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation='BILINEAR')
        # crop = transforms.RandomCrop(32, padding=4)
        i, j, h, w  = crop.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333))
        image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)


        #  check this Ahmed ... it might work
        # crop = transforms.RandomCrop(32,  padding=4)
        # i, j, h, w  = crop.get_params(image, output_size=(32,32))
        # image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)
        rand_HorFlip = random.random()

        if rand_HorFlip > 0.5:
            image = TF.hflip(image)

        aug_parameters = [i, j, h, w, rand_HorFlip] 

        image = test_transform(image)

        return image, aug_parameters

    def transform_apply(self, image, test_transform, aug_parameters=None):
        if aug_parameters is not None:
            i, j, h, w, rand_HorFlip = aug_parameters
            image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)

            if rand_HorFlip > 0.5:
                image = TF.hflip(image)
        image = test_transform(image)
        return image, None

    def kl_divergence(self, reference_tensor, tensors):
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
    
    # def __del__(self):
    #     self.model.cpu()
    #     del self.model


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def transform_apply(image, test_transform, aug_parameters=None):
    if aug_parameters is not None:
        i, j, h, w, rand_HorFlip = aug_parameters
        image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)

        if rand_HorFlip > 0.5:
            image = TF.hflip(image)

    # Transform to tensor
    # image = TF.to_tensor(image)
    # norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # image = norm(image)
    image = test_transform(image)
    return image, None

def transform_new(image, test_transform):
    crop = transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation='BILINEAR')
    # crop = transforms.RandomCrop(32, padding=4)
    # image = transforms.ToPILImage()(image)
    i, j, h, w  = crop.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333))
    image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)


    #  check this Ahmed ... it might work
    # crop = transforms.RandomCrop(32,  padding=4)
    # i, j, h, w  = crop.get_params(image, output_size=(32,32))
    # image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)



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

class CKD_CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, model=None, delta=5):
        print("This the CKD dataloader")

        super().__init__(root=root, train=train, download=download,
                    transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.model = model
        self.delta = delta
        # self.QF_range = range(0,100+self.delta, self.delta)
        self.QF_range = []
        self.num_channels, self.height, self.width = 3 , 32 , 32
        if torch.cuda.is_available():
            model.cuda()

        if train:
            self.transform = transform_new
        else:
            self.transform = transform_apply

    
    def select_ckd(self, input, target):    
        with torch.no_grad():
            data_tensor = torch.empty(0, self.num_channels, self.height, self.width)    
            # Create a transform to convert the tensor to a PIL image
            # Convert the tensor to a PIL image
            data_aug_parameters = []
            
            input_aug, aug_parameters = transform_new(input)
            data_tensor = torch.cat((data_tensor, input_aug.unsqueeze(0)), dim=0)
            data_aug_parameters.append(aug_parameters)

            # Create a tensor with the desired shape
            target_tensor = torch.empty(len(self.QF_range)+1, dtype=torch.long)
            # Fill the tensor with the integer number N
            target_tensor.fill_(target)

            # Generate all possible compressed images
            for qf in self.QF_range:
                buffer = BytesIO()
                # input.save(buffer, "JPEG", quality=5)
                # input.save(buffer, 'JPEG', quality=qf, subsampling=0)
                input.save(buffer, 'JPEG', quality=qf, 
                                        # exif =  image.info['exif'],
                                        # optimize=False,
                                        # qtable=image.quantization,
                                        # subsampling=0,
                                        # subsampling=JIP.get_sampling(input)
                                        )
                image = Image.open(buffer).convert("RGB")
                image, aug_parameters = transform_new(image)
                data_tensor = torch.cat((data_tensor, image.unsqueeze(0)), dim=0)
                data_aug_parameters.append(aug_parameters)

            if torch.cuda.is_available():
                if data_tensor.device.type != 'cuda': data_tensor = data_tensor.cuda()
                if target_tensor.device.type != 'cuda': target_tensor = target_tensor.cuda()

            output = self.model(data_tensor)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target_tensor.view(1, -1).expand_as(pred))

            correct_indexes = torch.nonzero(correct[0]).squeeze(1)
            if 0 in correct_indexes or len(correct_indexes) == 0:
                # print("No correct samples generated")
                return data_tensor[0]

            if 0 not in correct_indexes:
                sample_indexes = torch.cat([torch.tensor([0]).cuda(), correct_indexes], dim=0)
            else:
                sample_indexes = correct_indexes
            
            # Convert the reference tensor to a probability distribution by applying the softmax function
            output = output[sample_indexes]
            output = F.softmax(output, dim=1)
            kl_values = kl_divergence(output[0], output)
            correct_probs, correct_index = torch.max(output, dim=1)
            correct_probs = correct_probs[correct_index == target]
            kl_values = kl_values[correct_index == target]
            max_index = torch.argmax(kl_values)
            if (len(kl_values) > 1): breakpoint()
        return data_tensor[torch.argmax(kl_values)]


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        # lossless compression
        # buffer = BytesIO()
        # img.save(buffer, format='WebP', lossless=True)
        # img = Image.open(buffer).convert("RGB")
        
        if self.transform is not None:
            img, _ = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100Dataset_2_INPUTS(datasets.CIFAR100):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        print("This the CIFAR100Dataset_2_INPUTS")
        super().__init__(root=root, train=train, download=download,
                transform=transform, target_transform=target_transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]
        
        if self.transform:
            sample, target = self.transform(sample, target)
        
        return sample, target


class CIFAR100Dataset_simple(datasets.CIFAR100):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        print("This the CIFAR100Dataset_simple")
        super().__init__(root=root, train=train, download=download,
                transform=transform, target_transform=target_transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]
        
        if self.transform:
            sample, target = self.transform(sample)
        
        return sample, target


class BatchProcessor:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.pool = mp.Pool(processes=num_workers, initializer=self._init_worker)

    def process_batch(self, batch, func):
        images, targets = batch
        processed_batch = self.pool.starmap(func, zip(images, targets))
        processed_batch = torch.stack(processed_batch, dim=0)
        return processed_batch

    def _init_worker(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def __del__(self):
        self.pool.close()
        self.pool.join()


class CKD_selector_parallel(object):
    def __init__(self, model=None, delta=5, train=False, batch_size=32):
        print("This the CKD dataloader: ")
        self.train = train
        self.model = model
        self.delta = delta
        self.QF_range = range(0,100+self.delta, self.delta)
        # self.QF_range = []
        self.num_channels, self.height, self.width = 3 , 32 , 32
        self.batch_size=batch_size
        
        if torch.cuda.is_available():
            model.cuda()

        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])

        if train:
            self.transform = transform_new
        else:
            self.transform = transform_apply

        # this to return it to the process
        self.DataBatch =  torch.zeros(self.batch_size, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.data_tensor = torch.zeros(self.batch_size * (len(self.QF_range)+1), self.num_channels, self.height, self.width, dtype=torch.float32)
        
        self.target_tensor = torch.empty(self.batch_size * (len(self.QF_range)+1), dtype=torch.long)
        self.target_tmp = torch.empty((len(self.QF_range)+1), dtype=torch.long)

    def groupCal(self, input, target, size):
        for idx in range(size):
            self.generateCompress(input[idx], target[idx], idx)
        
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
                st_index = end_index
            else:
                ref_index = idx * window
                self.DataBatch[idx] = self.data_tensor[ref_index].float()


        return self.DataBatch[:size]
            
    def generateCompress(self, input, target, idx):
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
        
        data_window = idx * (len(self.QF_range)+1)
        self.data_tensor[data_window] = input_aug 

        buffer = BytesIO()
        for idx, qf in enumerate(self.QF_range):
            input.save(buffer, 'JPEG', quality=qf, 
                                    subsampling=0,
                                    )
            image = Image.open(buffer).convert("RGB")
            image, aug_parameters = self.transform(image, self.test_transform)
            self.data_tensor[data_window+idx+1] = image
            buffer.seek(0)
            buffer.truncate()
        
        
        # Fill the tensor with the integer number N
        self.target_tmp.fill_(target)
        self.target_tensor[data_window:data_window+(len(self.QF_range)+1)] = self.target_tmp
    

class CKD_selector(object):
    def __init__(self, model=None, delta=5, train=False, batch_size=32):
        print("This the CKD dataloader: ")
        self.train = train
        self.model = model
        self.delta = delta
        self.QF_range = range(0,100+self.delta, self.delta)
        # self.QF_range = []
        self.num_channels, self.height, self.width = 3 , 32 , 32
        self.batch_size=batch_size
        
        if torch.cuda.is_available():
            model.cuda()

        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])

        if train:
            self.transform = transform_new
        else:
            self.transform = transform_apply

        # Create a tensor with the desired shape
        self.data_tensor = torch.zeros(len(self.QF_range)+1, self.num_channels, self.height, self.width, dtype=torch.float32)
        self.target_tensor = torch.empty(len(self.QF_range)+1, dtype=torch.long)
        self.DataBatch =  torch.zeros(self.batch_size, self.num_channels, self.height, self.width, dtype=torch.float32).cuda()

    def distillBatch(self, input, target, size):
        for idx in range(size):
            self.DataBatch[idx] = self.distill(input[idx], target[idx])
        return self.DataBatch[:size]


    def distill(self, input, target):
        # return self.test_transform(input)
        # self.data_aug_parameters = []
        
        # Create a transform to convert the tensor to a PIL image
        # Convert the tensor to a PIL image
        if input.mode == "RGB":
            pass
        elif input.mode == "RGBA":
            # Convert the RGBA image to RGB and alpha channel tensors separately
            input = input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        input_aug, aug_parameters = self.transform(input, self.test_transform)
        # self.data_aug_parameters.append(aug_parameters)

        # Fill the tensor with the integer number N
        self.target_tensor.fill_(target)

        if torch.cuda.is_available():
            if self.data_tensor.device.type != 'cuda': self.data_tensor = self.data_tensor.cuda()
            if self.target_tensor.device.type != 'cuda': self.target_tensor  = self.target_tensor.cuda()
        
        self.data_tensor[0] = input_aug
        
        # Generate all possible compressed images
        for idx, qf in enumerate(self.QF_range):
            buffer = BytesIO()
            # input.save(buffer, "JPEG", quality=5)
            # input.save(buffer, 'JPEG', quality=qf, subsampling=0)
            input.save(buffer, 'JPEG', quality=qf, 
                                    # exif =  image.info['exif'],
                                    # optimize=False,
                                    # qtable=image.quantization,
                                    # subsampling=0,
                                    # subsampling=JIP.get_sampling(input)
                                    )
            image = Image.open(buffer).convert("RGB")
            image, aug_parameters = self.transform(image, self.test_transform)
            # data_tensor = torch.cat((data_tensor, image.unsqueeze(0)), dim=0)
            self.data_tensor[idx+1] = image
            # self.data_aug_parameters.append(aug_parameters)
        
        self.data_tensor = self.data_tensor.float() # this is a must to get a valid output
        output = self.model(self.data_tensor)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(self.target_tensor.view(1, -1).expand_as(pred))
        correct_indexes = torch.nonzero(correct[0]).squeeze(1)

        # Return the original Image if none of the compressed images are correct
        if len(correct_indexes) == 0:
            return input_aug.cuda()

        if 0 not in correct_indexes:
            sample_indexes = torch.cat([torch.tensor([0]).cuda(), correct_indexes], dim=0)
        else:
            sample_indexes = correct_indexes
        
        # Convert the reference tensor to a probability distribution by applying the softmax function
        output = output[sample_indexes]
        output = F.softmax(output, dim=1)
        kl_values = kl_divergence(output[0], output)
        correct_probs, correct_index = torch.max(output, dim=1)
        correct_probs = correct_probs[correct_index == target]
        kl_values = kl_values[correct_index == target]

        # torch.argmax(correct_probs)
        # pred.eq(self.target_tensor.view(1, -1).expand_as(pred))
        # if len(correct_indexes) > 1 and correct[0][0] == False:
        #     breakpoint()

        # return input_aug.cuda()

        if kl_values.numel() > 0:
            return self.data_tensor[correct_indexes[torch.argmax(kl_values)]]
        else:
            return self.data_tensor[0]
'''
output = self.model(self.data_tensor[correct_indexes[torch.argmax(kl_values)]].unsqueeze(0))
_, pred = output.topk(1, 1, True, True)
pred = pred.t()
correct = pred.eq(self.target_tensor[0].view(1, -1).expand_as(pred))
'''