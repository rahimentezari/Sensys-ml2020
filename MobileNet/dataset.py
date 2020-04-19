import os
import itertools
import numpy as np
from auglib.augmentation import Augmentations, set_seeds
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName
from auglib.meters import AverageMeter
from auglib.test import test_with_augmentation
import torchvision.datasets as datasets
import torch, os
import pickle


class Dataset(object):
    def __init__(self, datasource, path_data, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(42)
        if self.datasource == 'mnist':
            self.num_classes = 10
            # self.dataset = mnist.read_data(os.path.join(self.path_data, 'MNIST'))
        elif self.datasource == 'isic':
            self.num_classes = 2
            # self.dataset = mnist.read_data(os.path.join(self.path_data, 'MNIST'))

            data_dir = '/home/r/raent/Rahim/NetworkCompression/Single-ModeCompression/Code/' \
                       'DeconstructingLotteryTicket/ISIC2016/Dataset'

            num_workers = {
                'train': 100,
                'val': 100,
                'test': 100
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
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=900,
                                                          shuffle=True, num_workers=num_workers[x])
                           for x in ['train', 'val', 'test']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
            print(dataloaders)
            # with open('ISIC_dataset.pkl', 'wb') as handle:
            #     pickle.dump(dataloaders, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # with open('ISIC_dataset.pkl', 'rb') as handle:
            #     dataloaders = pickle.load(handle)
            phase = 'train'
            train_dict = {m: [] for m in ['input', 'label']}
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                train_dict = {'input': inputs.cpu().numpy().transpose((0, 3, 2, 1)), 'label': labels.cpu().numpy()}
                # train_dict['input'].append(np.asarray(inputs.cpu().numpy()))
                # train_dict['label'].append(np.asarray(labels.cpu().numpy()))
                # train_dict_tmp = {'input': inputs.cpu().numpy(), 'label': labels.cpu().numpy()}
                # train_dict['input'] = [train_dict['input'], train_dict_tmp['input']]
                # train_dict['label'] = [train_dict['label'], train_dict_tmp['label']]
            phase = 'test'
            test_dict = {k: [] for k in ['input', 'label']}
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                test_dict = {'input': inputs.cpu().numpy().transpose((0, 3, 2, 1)), 'label': labels.cpu().numpy()}
                # test_dict['input'].append(np.asarray(inputs.cpu().numpy()))
                # test_dict['label'].append(np.asarray(labels.cpu().numpy()))
                # test_dict_tmp = {'input': inputs.cpu().numpy(), 'label': labels.cpu().numpy()}
                # test_dict['input'] = [test_dict['input'], test_dict_tmp['input']]
                # test_dict['label'] = [test_dict['label'], test_dict_tmp['label']]

            self.dataset = {'train': train_dict, 'test':test_dict}

        elif self.datasource == 'cifar-10':
            print(cifar.read_data(os.path.join(self.path_data, 'cifar-10-batches-py')))
            self.num_classes = 10
            self.dataset = cifar.read_data(os.path.join(self.path_data, 'cifar-10-batches-py'))
        else:
            raise NotImplementedError
        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1),
            self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        print("self.num_example", self.num_example)
        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}
