import io, os, re, ftplib, posixpath
import os.path as osp

import numpy as np
from PIL import Image

import torch, torchvision
from torch import utils
import torchvision.transforms as T

class Imagenet64(utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.train = kwargs['train']
        self.ftp = None
        if 'url' in kwargs.keys():
            self.url = kwargs['url']
            assert 'ftp' == self.url[0:3]
            self.use_ftp = True
            addr_re = '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'
            self.addr = re.findall(addr_re, self.url)[0] 
            self.ftp = ftplib.FTP(self.addr)
            self.__ftp_login__()


            self.list_func = self.ftp.nlst
            self.join_func = posixpath.join 
            self.root_dir = re.split(addr_re, self.url)[-1]
        else: 
            self.join_func = osp.join
            self.list_func = lambda path: [self.join_func(path, d) for d in os.listdir()]
            self.root_dir = kwargs['root_dir']

        self.image_dir = self.join_func(self.root_dir, kwargs['image_dir'])
        self.label_dir = self.join_func(self.root_dir, kwargs['label_dir'])

        self.images = self.list_func(self.image_dir)
        self.labels = self.list_func(self.label_dir)

        if 'transforms' in kwargs.keys():
            self.transform = T.compose([*kwargs['transforms'],T.ToTensor()])
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __getitem__(self, idx):
        # TODO: handle splitting of training / validation data 
        image_file = self.images[idx]
        label_file = self.labels[idx]
        if self.ftp:
            with io.BytesIO() as f:
                self.ftp.retrbinary('RETR ' + image_file, f.write)
                img = Image.open(f, mode='r', formats=['PNG']).convert('RGB')
                X = self.transform(img) 
            
            with io.BytesIO() as f:
                self.ftp.retrbinary('RETR ' + label_file, f.write)
                y = int(f.getvalue())
        else:
            # TODO: implement handling of local files
            pass

        return X, y
        
    def __len__(self):
        return len(self.images)

    def __ftp_login__(self):
        self.ftp.login(user='anonymous')

if __name__=='__main__':

    args = {'url':'ftp://192.168.1.58/data/Imagenet64',
        'image_dir':'images','label_dir':'labels',
        'train':True}

    data = Imagenet64(**args)
    x, y = data[0]
    print(x)
    print(x.shape)