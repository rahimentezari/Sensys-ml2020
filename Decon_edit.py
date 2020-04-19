import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from models.train_model import train_model
from models.AlexNet3 import alexnet
from models.CNN_3 import CNN_3
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import shutil
import json
from os import listdir
from os.path import isfile, join
import csv, random
import numpy as np
import time
import glob
import pickle, pandas
from random import shuffle
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from auglib.augmentation import Augmentations, set_seeds
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName
from auglib.meters import AverageMeter
from auglib.test import test_with_augmentation
# ############################################################################################### create Dataset Folders
path = '/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/' \
             'DeconstructingLotteryTicket/ISIC2016/'

# with open(path + "Dataset/p.txt") as f:
#     content = f.readlines()
#     p_rand = random.sample(content, len(content))
#     p_rand_train = p_rand[0:14000]
#     p_rand_val = p_rand[14000:17000]
#     p_rand_test = p_rand[17000:20000]
#     print(len(p_rand_train), len(p_rand_val), len(p_rand_test))
#
# p_train = []
# for x in p_rand_train:
#     x = x.strip()
#     x = x[:-4]
#     p_train.append(x)
#
# df = pandas.DataFrame(data={"pix": p_train, "zlabel": "1"})
# df.to_csv(path + "Dataset/p_train.csv", sep=',',index=False, header=None)
#
# p_val = []
# for x in p_rand_val:
#     x = x.strip()
#     x = x[:-4]
#     p_val.append(x)
#
# df = pandas.DataFrame(data={"pix": p_val, "zlabel": "1"})
# df.to_csv(path + "Dataset/p_val.csv", sep=',',index=False, header=None)
#
# p_test = []
# for x in p_rand_test:
#     x = x.strip()
#     x = x[:-4]
#     p_test.append(x)
#
# df = pandas.DataFrame(data={"pix": p_test, "zlabel": "1"})
# df.to_csv(path + "Dataset/p_test.csv", sep=',',index=False, header=None)
#
# with open(path + "Dataset/n.txt") as f:
#     content = f.readlines()
#     n_rand = random.sample(content, len(content))
#     n_rand_train = n_rand[0:14000]
#     n_rand_val = n_rand[14000:17000]
#     n_rand_test = n_rand[17000:20000]
#     print(len(n_rand_train), len(n_rand_val), len(n_rand_test))
#
# n_train = []
# for x in n_rand_train:
#     x = x.strip()
#     x = x[:-4]
#     n_train.append(x)
#
# df = pandas.DataFrame(data={"pix": n_train, "zlabel": "0"})
# df.to_csv(path + "Dataset/n_train.csv", sep=',',index=False, header=None)
#
# n_val = []
# for x in n_rand_val:
#     x = x.strip()
#     x = x[:-4]
#     n_val.append(x)
#
# df = pandas.DataFrame(data={"pix": n_val, "zlabel": "0"})
# df.to_csv(path + "Dataset/n_val.csv", sep=',',index=False, header=None)
#
# n_test = []
# for x in n_rand_test:
#     x = x.strip()
#     x = x[:-4]
#     n_test.append(x)
#
# df = pandas.DataFrame(data={"pix": n_test, "zlabel": "0"})
# df.to_csv(path + "Dataset/n_test.csv", sep=',',index=False, header=None)

######################################## copy to respective folders
# with open(path + 'Dataset/label_val/label_val.csv') as f:
#     # line = line.rstrip()
#     lis = [line.rstrip().split(',') for line in f]        # create a list of lists
#     for x in enumerate(lis):              #print the list items
#         id = str(x[1][0]) + '.jpg'
#         if int(x[1][1]) == 0:
#             shutil.move(path + 'Dataset/Negative/' + id, path + 'Dataset/val/0')
#         else:
#             shutil.move(path + 'Dataset/Positive/' + id, path + 'Dataset/val/1')
#
# with open(path + 'Dataset/label_train/label_train.csv') as f:
#     # line = line.rstrip()
#     lis = [line.rstrip().split(',') for line in f]        # create a list of lists
#     for x in enumerate(lis):              #print the list items
#         id = str(x[1][0]) + '.jpg'
#         if int(x[1][1]) == 0:
#             shutil.move(path + 'Dataset/Negative/' + id, path + 'Dataset/train/0')
#         else:
#             shutil.move(path + 'Dataset/Positive/' + id, path + 'Dataset/train/1')

# with open(path + 'Dataset/label_test/label_test.csv') as f:
#     # line = line.rstrip()
#     lis = [line.rstrip().split(',') for line in f]        # create a list of lists
#     for x in enumerate(lis):              #print the list items
#         id = str(x[1][0]) + '.jpg'
#         if int(x[1][1]) == 0:
#             shutil.move(path + 'Dataset/Negative/' + id, path + 'Dataset/test/0')
#         else:
#             shutil.move(path + 'Dataset/Positive/' + id, path + 'Dataset/test/1')
#
#
# ######################################################################################## Compute statistics of Dataset
# data_dir = '/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/' \
#            'DeconstructingLotteryTicket/ConcreteCrack/Dataset/'
# num_workers = {
#     'train': 100
# }
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(128),
#         transforms.ToTensor(),
#     ])
#
# }
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x]) for x in ['train']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
#                                               shuffle=True, num_workers=num_workers[x])
#                for x in ['train']}
# def online_mean_and_sd(loader):
#     """Compute the mean and sd in an online fashion
#
#         Var[x] = E[X^2] - E^2[X]
#     """
#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)
#     i = 1
#     for data in dataloaders['train']:
#         # print(i)
#         # print(data[0].size())
#         # b, c, h, w = data.shape
#         b, c, h, w = data[0].size()
#         nb_pixels = b * h * w
#         sum_ = torch.sum(data[0], dim=[0, 2, 3])
#         sum_of_square = torch.sum(data[0] ** 2, dim=[0, 2, 3])
#         fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#         snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
#
#         cnt += nb_pixels
#         i = i+1
#     return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
#
#
# print(dataloaders)
# mean, std = online_mean_and_sd(dataloaders)
# print(mean, std)  # (tensor([0.6851, 0.6653, 0.6328]), tensor([0.1442, 0.1402, 0.1382]))


# ########################################################################################################## Data Loader
data_dir = '/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/' \
             'DeconstructingLotteryTicket/ISIC2016/Dataset'

num_workers = {
    'train' : 100,
    'val'   : 100,
    'test'  : 100
}
aug = {
    'hflip': True,  # Random Horizontal Flip
    'vflip': True,  # Random Vertical Flip
    'rotation': 90,  # Rotation (in degrees)
    'shear': 20,  # Shear (in degrees)
    'scale': (0.8, 1.2),  # Scale (tuple (min, max))
    'color_contrast': 0.3,  # Color Jitter: Contrast
    'color_saturation': 0.3,  # Color Jitter: Saturation
    'color_brightness': 0.3,  # Color Jitter: Brightness
    'color_hue': 0.1,  # Color Jitter: Hue
    'random_crop': True,  # Random Crops
    'random_erasing': False,  # Random Erasing
    'piecewise_affine': False,  # Piecewise Affine
    'tps': True,  # TPS Affine
    'autoaugment': False,  # AutoAugmentation
}
aug['size'] = 224
# aug['mean'] = [0.485, 0.456, 0.406]
# aug['std'] = [0.229, 0.224, 0.225]
aug['mean'] = [0.6851, 0.6653, 0.6328]
aug['std'] = [0.1442, 0.1402, 0.1382]
# 'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90,
# "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True}' \

augs = Augmentations(**aug)

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x]) for x in ['train', 'val', 'test']}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          augs.tf_augment) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=num_workers[x])
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}


# # ####################################################################################################### Initialization
prune_each_iteration = 0.50
index_dict4d = dict([("conv1", []), ("conv2", []), ("conv3", []), ("conv4", []), ("conv5", []),
                     ("lin1", []), ("lin2", []), ("lin3", [])])
index_dict1d = dict([("conv1", []), ("conv2", []), ("conv3", []), ("conv4", []), ("conv5", []),
                     ("lin1", []), ("lin2", []), ("lin3", [])])

# ############### Save ImageNet weights as Initialization
# model_ft = CNN_3(False)
model_ft = alexnet(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)


def count_params(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += param.numel()
    return zeros


number_params_org = count_params(model_ft)
print("number_params_org", number_params_org)
print("ORG Model", model_ft)

if not os.path.exists('./iterations/' + str(0) + '/saved/'):
    os.makedirs('./iterations/' + str(0) + '/saved/')
if not os.path.exists('./iterations/' + str(0) + '/pruned/'):
    os.makedirs('./iterations/' + str(0) + '/pruned/')
if not os.path.exists('./iterations/' + str(0) + '/pruned_rewind/'):
    os.makedirs('./iterations/' + str(0) + '/pruned_rewind/')
torch.save(model_ft, './iterations/' + str(0) + '/saved/model_0_epoch.pt')
torch.save(model_ft, './iterations/' + str(0) + '/pruned/model_pruned.pt')
torch.save(model_ft, './iterations/' + str(0) + '/pruned_rewind/model_prunedrewind.pt')


# ## Add up hook to gradient in order to avoid updating certain weights
def my_hook4d_conv1(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["conv1"])):
        a,b,c,d = index_dict4d["conv1"][i]
        grad_clone[a,b,c,d] = 0
    grad.detach()
    return grad_clone


def my_hook4d_conv2(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["conv2"])):
        a,b,c,d = index_dict4d["conv2"][i]
        grad_clone[a,b,c,d] = 0
    grad.detach()
    return grad_clone


def my_hook4d_conv3(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["conv3"])):
        a,b,c,d = index_dict4d["conv3"][i]
        grad_clone[a,b,c,d] = 0
    grad.detach()
    return grad_clone


def my_hook4d_conv4(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["conv4"])):
        a,b,c,d = index_dict4d["conv4"][i]
        grad_clone[a,b,c,d] = 0
    grad.detach()
    return grad_clone


def my_hook4d_conv5(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["conv5"])):
        a,b,c,d = index_dict4d["conv5"][i]
        grad_clone[a,b,c,d] = 0
    grad.detach()
    return grad_clone


def my_hook4d_lin1(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["lin1"])):
        a, b = index_dict4d["lin1"][i]
        grad_clone[a, b] = 0
    grad.detach()
    return grad_clone


def my_hook4d_lin2(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["lin2"])):
        a, b = index_dict4d["lin2"][i]
        grad_clone[a, b] = 0
    grad.detach()
    return grad_clone


def my_hook4d_lin3(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict4d["lin3"])):
        a, b = index_dict4d["lin3"][i]
        grad_clone[a, b] = 0
    grad.detach()
    return grad_clone


def my_hook1d_conv1(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["conv1"])):
        a = index_dict1d["conv1"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_conv2(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["conv2"])):
        a = index_dict1d["conv2"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_conv3(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["conv3"])):
        a = index_dict1d["conv3"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_conv4(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["conv4"])):
        a = index_dict1d["conv4"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_conv5(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["conv5"])):
        a = index_dict1d["conv5"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_lin1(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["lin1"])):
        a = index_dict1d["lin1"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_lin2(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["lin2"])):
        a = index_dict1d["lin2"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


def my_hook1d_lin3(grad):
    grad_clone = grad.clone()
    for i in range(len(index_dict1d["lin3"])):
        a = index_dict1d["lin3"][i]
        grad_clone[a] = 0
    grad.detach()
    return grad_clone


# ################################################################################################# Train to Convergence
def train_to_convergence(prune_each_iteration, class_weight, index_dict4d, index_dict1d, num_epochs, number_params_org):
    path_load = glob.glob('./iterations/' + str(prune_iteration-1) + '/pruned_rewind/model_prunedrewind.pt')[0]
    print("path_init_load", path_load)
    # model = alexnet(True)
    model = torch.load(path_load)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = nn.DataParallel(model)
    model_ft = model_ft.to(device)
    # Loss Function
    print(class_weight)
    # ############################## LOSS

    # criterion = torch.nn.BCEWithLogitsLoss(weight=class_weight, reduce=False) # returns an array not a scalar!
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)

    # ################################################################################ Count number of zeros
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "lin1", "lin2", "lin3"]
    all = 0
    # for x in layers:
    #     key = index_dict4d[x]
    #     print("index_dict4d", x, len(key))
    #     all = all + len(key)
    #     key = index_dict1d[x]
    #     print("index_dict1d", x, len(key))
    #     all = all + len(key)
    # print("index_dict4d1d_All", all)
    # # ######################################### Test if zero params remain zero! This checks for Conv1_bias
    # print("index_dict1d[conv1]", index_dict1d["conv1"])
    # if len(index_dict1d["conv1"]) != 0:
    #     first = index_dict1d["conv1"][0]
    #     last = index_dict1d["conv1"][-1]
    #     print("first", "last", first, last)
    # for child in model_ft.children():
    #     for layer in child.children():
    #         for param in layer.parameters():
    #             if param.data.size() == (64,):
    #                 print("hello")
    #                 for x in range(len(index_dict1d["conv1"])):
    #                     print("param.data[x]", index_dict1d["conv1"][x], param.data[index_dict1d["conv1"][x]])
    # ######################################################################################################
    hook_dict4d = dict([("hook1", my_hook4d_conv1), ("hook2", my_hook4d_conv2), ("hook3", my_hook4d_conv3),
                        ("hook4", my_hook4d_conv4), ("hook5", my_hook4d_conv5),
                        ("hook6", my_hook4d_lin1), ("hook7", my_hook4d_lin2), ("hook8", my_hook4d_lin3)])

    hook_dict1d = dict([("hook1", my_hook1d_conv1), ("hook2", my_hook1d_conv2), ("hook3", my_hook1d_conv3),
                        ("hook4", my_hook1d_conv4), ("hook5", my_hook1d_conv5),
                        ("hook6", my_hook1d_lin1), ("hook7", my_hook1d_lin2), ("hook8", my_hook1d_lin3)])
    # hook_dict = dict([("my_hook", my_hook)])

    # ############################################################################ Apply hooks
    conv_counter = 0
    lin_counter = 0
    child_counter = 0
    for child in model_ft.module.children():
        layer_counter = 0
        for layer in child.children():  # Going thru all layers of the network
            # print(layer)
            # torch.nn.utils.clip_grad_norm_(layer.parameters(), 1)
            if "Conv2d" in str(layer):  # check if it is a conv layer
                conv_counter += 1
                for param in layer.parameters():
                    if len(param.data.size()) == 4:
                        hook_name = "hook" + str(conv_counter)
                        param.register_hook(hook_dict4d[hook_name])  # zeroes the gradients over all mini-batches
                    else:
                        hook_name = "hook" + str(conv_counter)
                        param.register_hook(hook_dict1d[hook_name])
            elif "Linear" in str(layer):  # check if it is a linear layer
                lin_counter += 1
                for param in layer.parameters():
                    if len(param.data.size()) == 2:
                        hook_name = "hook" + str(5 + lin_counter)
                        param.register_hook(hook_dict4d[hook_name])
                    else:
                        hook_name = "hook" + str(5 + lin_counter)
                        param.register_hook(hook_dict1d[hook_name])
            else:
                for param in layer.parameters():
                    param.requires_grad = True  # freeze part of your model
            layer_counter += 1
        child_counter += 1
    # ############################################################################
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft,
                                               milestones=[10],
                                               gamma=0.1)
    total_params = sum(p.numel() for p in model_ft.parameters())
    trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    # print("Total Params - Trainable Params", total_params, trainable_params)
    # ###################################################################### Count Zeros weights
    def countZeroWeights(model):
        zeros = 0
        for param in model.parameters():
            if param is not None:
                zeros += param.numel() - param.nonzero().size(0)
        return zeros
    number_Zeros = countZeroWeights(model_ft)
    print("Zero Params_Before training", number_Zeros)

    # path_load = glob.glob('./iterations/' + str(prune_iteration - 1) + '/pruned/model_pruned.pt')[0]
    # model = torch.load(path_load)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_pruned = model.to(device)
    # def countZeroWeights2(model):
    #     zeros = 0
    #     for child in model.children():
    #         # print(child)
    #         for layer in child.children():
    #             # print("layer", layer)
    #             for param in layer.parameters():
    #                 # print("hi")
    #                 zeros += param.data.view(-1).size()[0] - len(torch.nonzero(param.data))
    #     return zeros
    # number_Zeros = countZeroWeights2(model_pruned)
    # print("Zero Params_Before training2", number_Zeros)

    # ############################################################################ Train with hooks
    # best, best_model = train_model(prune_each_iteration, model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft,
    #                                num_epochs=num_epochs)
    best, best_model, best_tpfpfn0, best_tpfpfn1, best_test_acc, best_test_auc = \
        train_model(prune_each_iteration, model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft,
                    num_epochs=num_epochs, scheduler=scheduler)
    # ############################################################################
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    # print("Total Params - Trainable Params", total_params, trainable_params)
    number_Zeros = countZeroWeights(best_model)
    print("Zero Params_After traning", number_Zeros)
    percent = 1 - float(number_Zeros)/number_params_org
    print("percent", percent)
    return percent, best, best_model,  best_tpfpfn0, best_tpfpfn1, best_test_acc, best_test_auc


# ####################################################################################### prune_and_rewindRemaining_to_k
def prune_and_rewindRemaining_to_k(prune_iteration, best, best_model, k):
    best_path = 'iterations/' + str(prune_iteration) + '/saved/model_' + str(best) + '_epoch.pt'
    # last_best = glob.glob('iterations/' + str(prune_iteration - 1) + '/saved/*.pt')[0]
    # init_path = last_best
    init_path = glob.glob('./iterations/' + str(prune_iteration - 1) + '/pruned_rewind/model_prunedrewind.pt')[0]

    model = torch.load(best_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_end = model.to(device)

    # model.load_state_dict(torch.load(init_path))
    model = torch.load(init_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_init = model.to(device)

    print("This step will last around 10 minutes!")
    # print("model_init", model_init)
    # print("model_end", model_end)

    # print(model_init)
    conv_counter = 0
    lin_counter = 0
    for child_end, child_init in zip(model_end.children(), model_init.children()):
        for layer_end, layer_init in zip(child_end.children(), child_init.children()):  # Going thru layers of the network
            # print(layer_init)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if "Conv2d" in str(layer_end):  # check if it is a conv layer
                # print("This is a Conv Layer")
                conv_counter += 1
                for param_end, param_init in zip(layer_end.parameters(), layer_init.parameters()):
                    diff_param = torch.zeros(param_end.data.size())
                    diff_param = diff_param.to(device)
                    # print(diff_param.shape)
                    if len(param_end.data.size()) == 4:
                        # Loop through all the entries
                        diff_param = abs(param_end.data) - abs(param_init.data)
                        index_name = "conv" + str(conv_counter)
                        for x in index_dict4d[index_name]:
                            diff_param[x] = 10e5
                        w_prune_this_iteration = \
                            int((param_end.data.view(-1).shape[0] - len(index_dict4d[index_name])) * prune_each_iteration)
                        idx = torch.topk(diff_param.view(-1), w_prune_this_iteration, largest=False, sorted=True, out=None)[1]
                        for id in range(w_prune_this_iteration):
                            org_idx = np.unravel_index(idx[id].item(), param_end.data.size())
                            param_end.data[org_idx] = 0
                            param_init.data[org_idx] = 0
                            index_dict4d[index_name].append(org_idx)
                    else:  # len(param_end.data.size()) == 1:
                        diff_param = abs(param_end.data) - abs(param_init.data)
                        for x in index_dict1d[index_name]:
                            diff_param[x] = 10e5
                        w_prune_this_iteration = \
                            int((param_end.data.view(-1).shape[0] - len(index_dict1d[index_name])) * prune_each_iteration)
                        idx = torch.topk(diff_param.view(-1), w_prune_this_iteration, largest=False, sorted=True, out=None)[1]
                        # diff_param.view(-1).scatter_(0, idx, 0)
                        for id in range(w_prune_this_iteration):
                            org_idx = np.unravel_index(idx[id].item(), param_end.data.size())
                            # if index_name == "conv1":
                            #     print(org_idx)
                            param_end.data[org_idx] = 0
                            param_init.data[org_idx] = 0
                            # param_init.data[org_idx].copy_(torch.zeros_like(param_init[org_idx]))
                            index_dict1d[index_name].append(org_idx)
                            # mask_dict1d[index_name][org_idx] = 0

            elif "Linear" in str(layer_end):
                lin_counter += 1
                for param_end, param_init in zip(layer_end.parameters(), layer_init.parameters()):
                    diff_param = torch.zeros(param_end.data.size())
                    diff_param = diff_param.to(device)
                    diff_param = abs(param_end.data) - abs(param_init.data)
                    if len(param_end.data.size()) == 2:
                        index_name = "lin" + str(lin_counter)
                        for x in index_dict4d[index_name]:
                            diff_param[x] = 10e5
                        w_prune_this_iteration = \
                            int((param_end.data.view(-1).shape[0] - len(index_dict4d[index_name])) * prune_each_iteration)
                        # Loop through all the entries
                        idx = torch.topk(diff_param.view(-1), w_prune_this_iteration, largest=False, sorted=True, out=None)[1]
                        # diff_param.view(-1).scatter_(0, idx, 0)
                        for id in range(w_prune_this_iteration):
                            org_idx = np.unravel_index(idx[id].item(), param_end.data.size())
                            param_end.data[org_idx] = 0
                            param_init.data[org_idx] = 0
                            index_name = "lin" + str(lin_counter)
                            # print(org_idx)
                            index_dict4d[index_name].append(org_idx)
                            # mask_dict4d[index_name][org_idx] = 0
                    else:  # len(param_end.data.size()) == 1:
                        diff_param = abs(param_end.data) - abs(param_init.data)
                        for x in index_dict1d[index_name]:
                            diff_param[x] = 10e5
                        w_prune_this_iteration = \
                            int((param_end.data.view(-1).shape[0] - len(index_dict1d[index_name])) * prune_each_iteration)
                        idx = torch.topk(diff_param.view(-1), w_prune_this_iteration, largest=False, sorted=True, out=None)[1]
                        # diff_param.view(-1).scatter_(0, idx, 0)
                        for id in range(w_prune_this_iteration):
                            org_idx = np.unravel_index(idx[id].item(), param_end.data.size())
                            param_end.data[org_idx] = 0
                            param_init.data[org_idx] = 0
                            index_dict1d[index_name].append(org_idx)
                            # mask_dict1d[index_name][org_idx] = 0



    # # ### Force to zero
    # layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "lin1", "lin2", "lin3"]
    # conv_counter = 0
    # for child in model_ft.children():
    #     # print(child)
    #     for layer in child.children():
    #         # print(layer)
    #         if "Conv2d" in str(layer):  # check if it is a conv layer
    #             conv_counter += 1
    #             index_name = "conv" + str(conv_counter)
    #             for param in layer.parameters():
    #                 # print("size", param.data.size())
    #                 if len(param.data.size()) == 4:
    #                     # print("len(index_dict4d[index_name])",len(index_dict4d[index_name]))
    #                     for x in range(len(index_dict4d[index_name])):
    #                         param.data[index_dict4d[index_name][x]] = 0
    #                 else:
    #                     # print("len(index_dict1d[index_name])", len(index_dict1d[index_name]))
    #                     for x in range(len(index_dict1d[index_name])):
    #                         param.data[index_dict1d[index_name][x]] = 0
    if not os.path.exists('./iterations/' + str(prune_iteration) + '/pruned/'):
        os.makedirs('./iterations/' + str(prune_iteration) + '/pruned/')
    if not os.path.exists('./iterations/' + str(prune_iteration) + '/pruned_rewind/'):
        os.makedirs('./iterations/' + str(prune_iteration) + '/pruned_rewind/')
    torch.save(model_end, './iterations/' + str(prune_iteration) + '/pruned/model_pruned.pt')
    torch.save(model_init, './iterations/' + str(prune_iteration) + '/pruned_rewind/model_prunedrewind.pt')

    return index_dict4d, index_dict1d


# ##############################################################################################################For loop
start_time = time.time()


k = 1
percent = 1.0
prune_iteration = 1
num_epochs = 500
class_weight = torch.FloatTensor([1, 1])
df = pd.DataFrame(columns=['iteration', 'Class_Number', 'CR', 'best_epoch', 'TP', 'FP', 'FN', 'Overall_Test_Precision',
                           'test_AUC'])
# for prune_iteration in range(1, 23):
while percent > 0.01:
    print("#################################################################iteration", prune_iteration)
    if prune_iteration > 1:
        num_epochs = 100
        class_weight = torch.FloatTensor([1, 1])
    print("Training Started!")
    percent, best, best_model, TPFPFN0_best, TPFPFN1_best, test_acc_best, test_auc_best\
        = train_to_convergence(prune_iteration, class_weight, index_dict4d, index_dict1d, num_epochs, number_params_org)
    # ######### Write to csv
    df.loc[((prune_iteration - 1) * 2) + 0 + 1] = prune_iteration, 0, percent, best,\
                                                  TPFPFN0_best[0].item(), TPFPFN0_best[1].item(), TPFPFN0_best[2].item(),\
                                                  test_acc_best, test_auc_best
    df.loc[((prune_iteration - 1) * 2) + 1 + 1] = prune_iteration, 1, percent, best,\
                                                  TPFPFN1_best[0].item(), TPFPFN1_best[1].item(), TPFPFN1_best[2].item(),\
                                                  test_acc_best, test_auc_best
    df.to_csv('stats.csv', sep='\t')
    # ######################
    print("Pruning Started!")
    index_dict4d, index_dict1d = prune_and_rewindRemaining_to_k(prune_iteration, best, best_model, k)

    prune_iteration = prune_iteration + 1


print("--- %s seconds ---" % (time.time() - start_time))




