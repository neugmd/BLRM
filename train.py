# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
from models import vgg, resnet, inception
import random
import xlwt 
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# set the random seed
random.seed(1024)

# which data you want to train
data_dir = '../data/OCT_600/'
# which file you want to record your train data file
train_data_file = './result/train_data_file.txt'
# record the estimated label per epoch
data_iter_label = './result/data_iter_label.txt'
# record the detail of label in each batch
data_batch_label = './result/data_batch_label.txt'

# choose the model
MODEL_NAME = 'resnet'
INPUT_IMAGE_SIZE = {'vgg': 225, 'resnet': 299, 'inception': 299}
INPUT_MODEL_SIZE = {'vgg': 224, 'resnet': 299, 'inception': 299}
input_image_size = INPUT_IMAGE_SIZE[MODEL_NAME]
input_model_size = INPUT_MODEL_SIZE[MODEL_NAME]

#------------------------------------------------------------------------------------------
# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_image_size),
        transforms.CenterCrop(input_model_size),
        #transforms.RandomResizedCrop(input_model_size),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_image_size),
        transforms.CenterCrop(input_model_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
}
#------------------------------------------------------------------------------------------
# initiallize the global parameters of model
CLASS_NUM = 4
BATCH_SIZE = 64
EPOCH_NUM = 80
# create a file to record something
def create_record_file(record_file):
    if os.path.exists(record_file):
       os.remove(record_file)
       os.mknod(record_file)
    else:
       os.mknod(record_file)
#------------------------------------------------------------------------------------------
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# random shuffle the train data
random.shuffle(image_datasets['train'].imgs)
random.shuffle(image_datasets['val'].imgs)
#------------------------------------------------------------------------------------------
# LMM: initiallize the label estimation parameters 
train_data_likelyhood = np.zeros((CLASS_NUM, len(image_datasets['train'])), dtype=np.float32) 
train_data_pred_prior = 1/CLASS_NUM * np.ones((CLASS_NUM, len(image_datasets['train'])), dtype=np.float32)    
train_data_pred_posterior = np.zeros((CLASS_NUM, len(image_datasets['train'])), dtype=np.float32)  
train_data_estimate_label = np.zeros((len(image_datasets['train'])), dtype=np.int)    
START_EPOCH = 18
EPOCH_WINDOW = 5.0
alpha = 1.0 - 1.0/EPOCH_WINDOW
beta = 1.0 - 1.0/EPOCH_WINDOW
WEIGHT = 1.0
MIN_PROB = 0.25   # control the prob to change label
IS_LMM = False # if turn on the LMM
SAMPLE_SELECTION = False
NOISE_RATE = 0.1*1.25
EPOCH_K = 15
#------------------------------------------------------------------------------------------
# record the random data in the txt
create_record_file(train_data_file)
# record the data label per batch
create_record_file(data_batch_label)

with open(train_data_file, 'a') as f1:
     for image_index, image_file in enumerate(image_datasets['train'].imgs):
         _image_name, image_label = image_file
         image_name = _image_name.split('/')[-1]
         train_data_likelyhood[image_label, image_index] = 1.0          
         train_data_estimate_label[image_index] = image_label
         f1.write(str(image_index+1) + '\t' + image_name + '\t' + str(image_label) + '\n') 
#------------------------------------------------------------------------------------------

#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
#                                             shuffle=True, num_workers=4)
#              for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------- save models function -----------------------------
def save_models(model, model_name, epoch):
    torch.save(model.state_dict(), "./save_models/trained_model_{}_{}.pth".format(model_name, epoch))
    print("Checkpoint saved")

#------------------------- train function -----------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    global train_data_likelyhood
    global train_data_pred_prior
    global train_data_pred_posterior
    global train_data_estimate_label
    #------------------------------------------------

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('train_para',cell_overwrite_ok=True)
    sheet.write(0,0,'epoch')
    sheet.write(0,1,'Learning_rate')
    sheet.write(0,2,'Loss/train')
    sheet.write(0,3,'Acc/train')

    sheet_v = book.add_sheet('val_para',cell_overwrite_ok=True)
    sheet_v.write(0,0,'epoch')
    sheet_v.write(0,1,'Learning_rate')
    sheet_v.write(0,2,'Loss/val')
    sheet_v.write(0,3,'Acc/val')
    
    # Set tensorboardX
    writer = SummaryWriter(comment='./')  

    create_record_file(data_iter_label)
    global_count_train = 0
    global_count_val = 0
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        with open(data_iter_label, 'a') as f2:
             f2.writelines(str(epoch) + " input_label: ")
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # Iterate over data.
                batch_start = 0
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:
                if phase == 'train':
                    global_count_train = global_count_train + 1
                if phase == 'val':
                    global_count_val = global_count_val + 1
                #pdb.set_trace()
                inputs = inputs.to(device)
                labels = labels.to(device)
                #pdb.set_trace()
                #vutils.save_image(inputs[1], './data_unlabel_ori.jpg', normalize=False)
                labels_numpy = labels.data.cpu().numpy()
                #pdb.set_trace()
               
                batch_end = batch_start + labels_numpy.shape[0]
                batch_index = np.ceil(batch_end / BATCH_SIZE)
                if phase == 'train':
                    
                    batch_end = batch_start + labels_numpy.shape[0]
                    #---------- wether change the labels or not -----------
                    if IS_LMM == True:
                        if epoch > EPOCH_WINDOW + START_EPOCH:
                            #print('-----------------------------------------------')
                            #print(labels_numpy)
                            #print(train_data_estimate_label[batch_start:batch_end])
                            #print('-----------------------------------------------')
                            LMM_FLAG = "ON"
                            labels_input = torch.from_numpy(train_data_estimate_label[batch_start:batch_end]).cuda()
                        else:
                            labels_input = labels
                            LMM_FLAG = "OFF"
                    else:
                        labels_input = labels
                        LMM_FLAG = "OFF"
                else:
                    labels_input = labels
                    LMM_FLAG = "OFF"
                #------------------------------------------------------

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'val':
                        outputs_logits = model(inputs)
                    elif MODEL_NAME == 'inception':
                         outputs = model(inputs)
                         outputs_logits = outputs[0]
                    else:
                         outputs_logits = model(inputs)
                    
                    #print(outputs_logits.shape)
                    _, preds = torch.max(outputs_logits, 1)
                    #---------------select samples to compute the loss-------------
                    if SAMPLE_SELECTION == True:
                        #--------set the percentage of one batch
                        percentage = 1 - NOISE_RATE*min(epoch*epoch/EPOCH_K, 1) 
                        #percentage = 0.75
                        #print("Batch Start: " + str(batch_start) + " selected percentage: " + str(percentage))
                        #--------compute the loss of each sample in the batch------
                        outputs_logits_softmax = F.softmax(outputs_logits,dim =1)
                        labels_onehot = torch.zeros(labels_numpy.shape[0], CLASS_NUM).scatter_(1, labels.cpu().unsqueeze(-1), 1)
                        #--------calculate and sort the loss--------
                        cross_loss = labels_onehot*outputs_logits_softmax.cpu()
                        cross_loss_sum = torch.sum(cross_loss, dim=1)
                        sorted_sum, indices = torch.sort(cross_loss_sum)

                        #--------find the index of selected samples---------
                        selected_num = int(labels_numpy.shape[0] * percentage)
                        threshold = sorted_sum[selected_num-1].item()
                        
                        #pdb.set_trace()
                        index = (cross_loss_sum <= threshold).nonzero().squeeze(1)
                        temp_logits = torch.index_select(outputs_logits.cpu(), 0, index)
                        temp_logits = temp_logits.to(device)
                        selected_outputs_logits = temp_logits
                    else:
                        selected_outputs_logits = outputs_logits
                    #--------------------------------------------------------------       
                    if phase == 'train':
                        #----------------------------------------------------------------------------
                        #------ estimate the label --------------------------------------------------
                        #pdb.set_trace()
                        output_logits_numpy = outputs_logits.data.cpu().numpy()
                        preds_numpy = preds.data.cpu().numpy()
                        # labels_numpy = labels.data.cpu().numpy()
                        iter_pred = copy.deepcopy(output_logits_numpy)
                        iter_pred_input = iter_pred - np.max(iter_pred, axis=1).reshape(len(iter_pred),1)
                        iter_pred_exp = np.exp(iter_pred_input)  #nomalization to [0, 1]
                        iter_pred_sum = np.sum(iter_pred_exp, axis=1)   #make summation
                        # iter_pred_sum_tile = np.tile(iter_pred_sum, (iter_pred_exp.shape[1],1))
                       
                        iter_pred_prob = iter_pred_exp.T / iter_pred_sum.reshape(1, iter_pred_exp.shape[0])

                        #pdb.set_trace()
                       
                        train_data_likelyhood[:, batch_start:batch_end] = train_data_likelyhood[:, batch_start:batch_end] * WEIGHT
                        for i in range(iter_pred_prob.shape[1]):
                            train_data_likelyhood[int(preds_numpy[i]), batch_start + i] = 1.0/(1.0 - np.power(beta, epoch+1.0)) * \
                                                ((1.0 - beta) * 1.0 + beta * \
                                                    train_data_likelyhood[int(preds_numpy[i]), batch_start + i])
                        #pdb.set_trace()
                        
                        train_data_likelyhood_tmp = copy.deepcopy(train_data_likelyhood[:, batch_start:batch_end])
                        train_data_likelyhood_input = train_data_likelyhood_tmp - np.max(train_data_likelyhood_tmp, axis=0).reshape(1, train_data_likelyhood_tmp.shape[1])
                        train_data_likelyhood_exp = np.exp(train_data_likelyhood_input)
                        likelyhood_tmp_sum = np.sum(train_data_likelyhood_exp, 0)
                        train_data_likelyhood_prob = train_data_likelyhood_exp / likelyhood_tmp_sum
                        #pdb.set_trace()
                        for i in range(train_data_pred_prior.shape[0]):
                            train_data_pred_prior[i, batch_start:batch_end] = 1.0/(1.0 - np.power(alpha, epoch+1.0)) * \
                                                ((1.0-alpha) * iter_pred_prob[i,:] + 
                                                    alpha * train_data_pred_prior[i, batch_start:batch_end]) 
                            train_data_pred_posterior[i, batch_start:batch_end] = \
                                                train_data_pred_prior[i, batch_start:batch_end] * \
                                                train_data_likelyhood_prob[i, :]
                        # pdb.set_trace()
                                            
                        train_data_pred_posterior_tmp = copy.deepcopy(train_data_pred_posterior[:, batch_start:batch_end])
                        train_data_pred_posterior_input = train_data_pred_posterior_tmp - np.max(train_data_pred_posterior_tmp, axis=0).reshape(1, train_data_pred_posterior_tmp.shape[1])
                        train_data_pred_posterior_exp = np.exp(train_data_pred_posterior_input)
                        train_data_pred_posterior_sum = np.sum(train_data_pred_posterior_exp, 0)
                        train_data_pred_posterior_prob = train_data_pred_posterior_exp / train_data_pred_posterior_sum
                        
                        #pdb.set_trace()
                        
                        for i in range(train_data_pred_posterior_prob.shape[1]):
                            if train_data_pred_posterior_prob[labels_numpy[i], i] < MIN_PROB:
                                #print(str(train_data_pred_posterior_prob[labels_numpy[i], i]))
                                _estimate_label = np.argmax(train_data_pred_posterior_prob[:, i])
                                train_data_estimate_label[batch_start + i] = _estimate_label
                            else:
                                train_data_estimate_label[batch_start + i] = labels_numpy[i]

                        #pdb.set_trace()
                        flag_E_P = (train_data_estimate_label[batch_start:batch_end] == labels_numpy)
                        with open(data_batch_label, 'a') as f3:
                              f3.writelines("epoch: "+str(epoch+1)+" batch_index: "+str(batch_index+1) + " LMM: " + LMM_FLAG + " flag: " + str(flag_E_P) + "\n")
                              f3.write("Original: " + str(labels_numpy) + '\n')
                              f3.write("Estimate: " + str(train_data_estimate_label[batch_start:batch_end]) + '\n')
                              f3.write("EstiProb: " + str(train_data_pred_posterior_prob) + '\n')
        
                        with open(data_iter_label, 'a') as f2:
                            train_data_estimate_label_str = str(train_data_estimate_label[batch_start:batch_end])[1:-1].split('\n')
                            for list_element in train_data_estimate_label_str:
                                f2.writelines(list_element + ' ')
                            #f2.writelines(str(train_data_estimate_label[batch_start:batch_end])[1:-1]+' ')

                    #----------------------------------------------------------------------------------------------
                    #----------------------------------------------------------------------------------------------
                    if SAMPLE_SELECTION == True:
                        labels_temp = torch.index_select(labels_input.cpu(), 0, index)
                        labels_temp = labels_temp.to(device)
                        selected_labels_input = labels_temp
                    else:
                        selected_labels_input = labels_input

                    loss = criterion(selected_outputs_logits, selected_labels_input)
                    #print('loss: {:.4f}'.format(loss.item()))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if phase == 'train':
                    batch_start = batch_end

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # record loss and acc by each step
                step_loss = loss.item() * inputs.size(0)
                step_acc = torch.sum(preds == labels.data).item() / preds.shape[0]
                '''
                # Write the tensorboardX records
                if phase == 'train':
                    writer.add_scalar('train/loss', float(step_loss), global_count_train)
                    writer.add_scalar('train/acc', float(step_acc), global_count_train)
                if phase == 'val':
                    writer.add_scalar('val/loss', float(step_loss), global_count_val)
                    writer.add_scalar('val/acc', float(step_acc), global_count_val)
                '''
            if phase == 'train':
                scheduler.step()
                with open(data_iter_label, 'a') as f2:
                     f2.writelines('\n')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('Epoch:{}/{}--{} Loss: {:.4f} lr: {:.8f} Acc: {:.4f}'.format(epoch, num_epochs-1,
                phase, epoch_loss, optimizer.param_groups[0]['lr'], epoch_acc))
            #pdb.set_trace()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            #store variables in each epoch
            if phase == 'train':
                sheet.write(epoch+1,0,epoch)
                sheet.write(epoch+1,1,optimizer.param_groups[0]['lr'])
                sheet.write(epoch+1,2,epoch_loss)
                sheet.write(epoch+1,3,epoch_acc.item())
                book.save(r'./result/train_loss_val_score.xls')
            
            if phase == 'val':
                sheet_v.write(epoch+1,0,epoch)
                sheet_v.write(epoch+1,1,optimizer.param_groups[0]['lr'])
                sheet_v.write(epoch+1,2,epoch_loss)
                sheet_v.write(epoch+1,3,epoch_acc.item())
                book.save(r'./result/train_loss_val_score.xls')
            
            # Write the tensorboardX records
            if phase == 'train':
                writer.add_scalar('train/loss', float(epoch_loss), epoch)
                writer.add_scalar('train/acc', float(epoch_acc.item()), epoch)
            if phase == 'val':
                writer.add_scalar('val/loss', float(epoch_loss), epoch)
                writer.add_scalar('val/acc', float(epoch_acc.item()), epoch)

        #save models
        if (epoch>=50) and (epoch%5 == 0):
            save_models(model, MODEL_NAME, epoch)

    writer.close()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    pdb.set_trace()
    return model
#------------------------------------------------------------------------------
'''
Load a pretrained model and reset final fully connected layer.
'''
#model_conv = vgg.vgg16(pretrained=False)
#model_conv = inception.inception_v3(pretrained=False)
#model_conv = models.resnet18(pretrained=False)
#pthfile = "./pretrained_model/vgg16-397923af.pth" 
#model_conv.load_state_dict(torch.load(pthfile))
#print(model_conv)

#pdb.set_trace()
#for param in model_conv.parameters():
#    param.requires_grad = True

# Parameters of newly constructed modules have requires_grad=True by default
if MODEL_NAME == 'resnet': 
   model_conv = models.resnet18(pretrained=False)
   #--------------resnet: change the last layer ------------------------
   num_ftrs = model_conv.fc.in_features
   model_conv.fc = nn.Linear(num_ftrs, CLASS_NUM)
   #-------------------------------------------------------------------
elif MODEL_NAME == 'vgg':
   model_conv = models.vgg16(pretrained=False)
   #--------------vgg16: change the last layer ------------------------
   num_ftrs = model_conv.classifier[6].in_features
   model_conv.classifier[6] = nn.Linear(num_ftrs, CLASS_NUM)
   #-------------------------------------------------------------------
else:
   model_conv = models.inception_v3(pretrained=False)
   #--------------inceptionV3: change the last layer ------------------
   num_ftrs = model_conv.fc.in_features
   model_conv.fc = nn.Linear(num_ftrs, CLASS_NUM)
   #-------------------------------------------------------------------
print(model_conv)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
#optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.5)
#------------------------------------------------------------------------------

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=EPOCH_NUM)

