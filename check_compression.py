import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from io import BytesIO
import torch
import copy 


import torch
import torch.nn as nn

# Define the random model architecture
class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()
        self.fc1 = nn.Linear(3072, 10)
        self.fc2 = nn.Linear(10, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the random model
model = RandomModel()

# Generate random input data
batch_size =1
width = 32
height = 32
channels = 3
input_data = torch.randn(batch_size, channels, height, width)  # Input tensor of shape (batch_size, channels, height, width)


test_transform = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

def transform_apply(image, aug_parameters=None, compress=False):
    resize = transforms.Resize(size=(32,32))
    image = resize(image)

    qf, i, j, h, w, rand_HorFlip = list(aug_parameters)
    try:
        if (aug_parameters is not None) and (not np.isnan(aug_parameters).any()) and aug_parameters[-1] != -2:
            # print(list(aug_parameters))
            image = TF.pad(image, (4, 4, 4, 4))
            image  = TF.resized_crop(image, i, j, h, w, size=(32,32), interpolation=TF.InterpolationMode.BILINEAR)
            if rand_HorFlip > 0.5:
                image = TF.hflip(image)
            
    except:
        print("Error in transform_apply")
    
    # if compress:
    #     buffer = BytesIO()
    #     image.save(buffer, 'JPEG', quality=int(qf), subsampling=0)
    #     image_wo_transform = copy.copy(Image.open(buffer).convert("RGB"))  
    
    image_wo_transform = image
    image = test_transform(image)
    return image, image_wo_transform

def transform_new(image, quality):
    resize = transforms.Resize(size=(32,32))
    image = resize(image)
    
    rand_HorFlip = random.random()

    image_org = copy.copy(image)

    # ONLY CIFAR 100
    padding = 4
    output_size = (32, 32)
    image = TF.pad(image, (padding, padding, padding, padding))
    crop = transforms.RandomCrop(output_size[0])
    i, j, h, w  = crop.get_params(image, output_size)
    image = TF.resized_crop(image, i, j, h, w, output_size, interpolation=TF.InterpolationMode.BILINEAR)
    
  
    # resize = transforms.Resize(size=(128,128))
    # image = resize(image)
    # print(i,j,h,w)
    # image.save('./compressed_images/example0_%.2f.jpg'%(rand_HorFlip))


    # image = TF.pad(image_org, (4, 4, 4, 4))
    # image = TF.resized_crop(image, i, j, h, w, output_size)
    # resize = transforms.Resize(size=(128,128))
    # image = resize(image)
    # image.save('./compressed_images/example1_%.2f.jpg'%(rand_HorFlip))




    # image = transforms.RandomCrop(32,  padding=4)(image_org)
    # resize = transforms.Resize(size=(128,128))
    # image = resize(image)
    # image.save('./compressed_images/example2_%.2f.jpg'%(rand_HorFlip))
    # resize = transforms.Resize(size=(32,32))
    # image = resize(image)
    
    # Random horizontal flipping
    if rand_HorFlip > 0.5:
        image = TF.hflip(image)

    aug_parameters = [i, j, h, w, rand_HorFlip] 


    # image.save(buffer1, 'JPEG', quality=quality, subsampling=0)
    # image_wo_transform = Image.open(buffer1).convert("RGB")
    image_wo_transform = image
    image = test_transform(image)
    return image, aug_parameters, image_wo_transform

# Load the original image
img = Image.open('original_image.jpg')

# Convert the image to a numpy array
img_arr = np.array(img)
input = Image.fromarray(img_arr)

buffer1 = BytesIO()
buffer2 = BytesIO()
count = 0

# qf_range = range(100,-1,-1)
qf_range = range(0,101,20)
# Loop over different JPEG quality levels and compress the image
for quality in qf_range:
    buffer = BytesIO()
    input_org = copy.copy(input)
    input_org.save(buffer1, 'JPEG', quality=quality, subsampling=0)
    input_org = Image.open(buffer1).convert("RGB")

    # Compress the image using JPEG with the specified quality level
    input1 = copy.copy(input_org)
    input1, aug_parameters, input1_wo_transform = transform_new(input1, quality)      
    aug_parameters.insert(0, quality) 
    input2 = copy.copy(input_org)
    input2, input2_wo_transform = transform_apply(input2, aug_parameters=aug_parameters, compress=True)

    input1 = input1.view(batch_size, -1)  # Flatten the input1 tensor
    output1 = model(input1)

    input2 = input2.view(batch_size, -1)  # Flatten the input2 tensor
    output2 = model(input2)

    # print(np.sum(np.subtract(input1_wo_transform, input2_wo_transform)))

    # Check if the pixel values of the original and compressed images are the same
    # if torch.sum(torch.subtract(input1, input2)) < 1 :
    if torch.all(torch.eq(output1, output2)) and torch.sum(torch.subtract(input1, input2)) == 0 \
        and np.sum(np.subtract(input1_wo_transform, input2_wo_transform)) == 0 :
        # print(f"pixel values and Transform ==> QF {quality} has the same as the original image")
        count+=1
    else:
        print(f"pixel values and Transform ==> QF {quality} is different from the original image")

print("Counts : ",(100* count/len(qf_range)))
# Delete the original image and the original numpy array to free memory
img.close()
del img_arr