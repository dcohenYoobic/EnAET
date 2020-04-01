# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
from ops.os_operation import mkdir
import os
from torchvision.datasets.utils import download_url, check_integrity
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np
import scipy.io as sio
import glob
from PIL import Image
import json
class SANOFI(object):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root):
        self.root = root
        self.final_path = os.path.join(self.root, 'sanofi')
        mkdir(self.final_path)
        self.train_path = os.path.join(self.final_path, 'trainset')
        self.test_path = os.path.join(self.final_path, 'testset')
        mkdir(self.train_path)
        mkdir(self.test_path)
        if os.path.getsize(self.train_path) < 10000:
            self.Process_Dataset(self.train_path,'train')
        if os.path.getsize(self.test_path) < 10000:
            self.Process_Dataset(self.test_path,'test')

    def Process_Dataset(self, train_path, split):
        mapping_labels = {"box":-1}
        img_paths = glob.glob("catalog_{}_new/*/*.jpg".format(split))
        for i,img_path in enumerate(img_paths):
            img = Image.open(img_path)
            dir_path = os.path.dirname(img_path)
            if dir_path not in mapping_labels.keys():
                mapping_labels[dir_path] = max(mapping_labels.values()) + 1
            tmp_train_path=os.path.join(train_path,'trainset'+str(i)+'.npy')
            tmp_aim_path = os.path.join(train_path, 'aimset' + str(i) + '.npy')
            np.save(tmp_train_path,img)
            np.save(tmp_aim_path,np.int64(mapping_labels[dir_path]))
        with open("classes.json", 'w') as f:
            json.dump(mapping_labels, f)
        
            

