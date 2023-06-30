from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os,shutil
from models import vgg, resnet, inception
import xlwt
import pdb
import numpy as np

BATCH_SIZE = 10
# which file you want to record your train data file
val_data_file = './result/val_data_file.txt'

MODEL_NAME = 'inception'
INPUT_MODEL_SIZE = {'vgg': 224, 'resnet': 224, 'inception': 299}
INPUT_IMAGE_SIZE = {'vgg': 225, 'resnet': 225, 'inception': 299}
input_model_size = INPUT_MODEL_SIZE[MODEL_NAME]
input_image_size = INPUT_IMAGE_SIZE[MODEL_NAME]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create a file to record something
def create_record_file(record_file):
    if os.path.exists(record_file):
       os.remove(record_file)
       os.mknod(record_file)
    else:
       os.mknod(record_file)

import os
 
# new folder
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                    
        os.makedirs(path)             

def visualize_model(MODEL_NAME,model_path,img_path):
    if MODEL_NAME == 'resnet': 
       model = models.resnet50(pretrained=False)
       #--------------resnet: change the last layer ------------------------
       num_ftrs = model.fc.in_features
       model.fc = nn.Linear(num_ftrs, 4)
       #-------------------------------------------------------------------
    elif MODEL_NAME == 'vgg':
       model = vgg.vgg16(pretrained=False)
       #--------------vgg16: change the last layer ------------------------
       num_ftrs = model.classifier[6].in_features
       model.classifier[6] = nn.Linear(num_ftrs, 2)
       #-------------------------------------------------------------------
    else:
       model = inception.inception_v3(pretrained=False)
       #--------------inceptionV3: change the last layer ------------------
       num_ftrs = model.fc.in_features
       model.fc = nn.Linear(num_ftrs, 4)
    
    model = model.to(device) 
    #model.load_state_dict(torch.load(model_path),strict=False)
    model.load_state_dict(torch.load(model_path),strict=False)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize(input_image_size),
        transforms.CenterCrop(input_model_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = 'val'
    image_datasets = datasets.ImageFolder(os.path.join(img_path, x),data_transforms)
    
    #---------------record img name and true label--------------------
    list_name = []
    create_record_file(val_data_file)
    with open(val_data_file, 'a') as f1:
     for image_index, image_file in enumerate(image_datasets.imgs):
         _image_name, image_label = image_file
         image_name = _image_name.split('/')[-1]
         f1.write(str(image_index+1) + '\t' + image_name + '\t' + str(image_label) + '\n') 
         list_name.append(image_name)
    #-----------------end--------------
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=0)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('test',cell_overwrite_ok=True)
    sheet.write(0,0,'image name')
    sheet.write(0,1,'true label')
    sheet.write(0,2,'predict label')
    #sheet.write(0,3,'predict prob')

    index = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            index = index + 1
            print(index)
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_numpy = labels.data.cpu().numpy()
            outputs = model(inputs)
            # softmax
            m = nn.Softmax(dim=1)
            out_prob = m(outputs)
            #pdb.set_trace()
            out_prob_numpy = out_prob.data.cpu().numpy()
            # the label of max prob
            _, preds = torch.max(outputs, 1)
            pro_len = np.size(out_prob_numpy,0)
            for t in range(pro_len):
                num = t + (index - 1) * BATCH_SIZE
                sheet.write(num,0,list_name[num])
                sheet.write(num,1,str(labels_numpy[t]))
                sheet.write(num,2,str(preds[t].item()))
                #sheet.write(num,3,str(out_prob_numpy[t][1]))
                book.save(r'./test.xls')               

model_path = './save_models/trained_model_inception_50.pth'
img_path = '../data/OCTMNIST_1000_0.1'
visualize_model(MODEL_NAME,model_path,img_path)
