import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
import glob
import numpy as np
from itertools import chain
warnings.filterwarnings('ignore')
from .ModelNetDataLoader import pc_normalize, farthest_point_sample

def load_points_from_category(paths):
    return [np.loadtxt(x, delimiter=',').astype(float) for x in paths]


class MAMLModelNetDataset(Dataset):
    def __init__(self, root, categories:list[str], args):
        self.root = root
        self.npoints = args.num_point
        self.categories_name = categories
        cat_directories = [os.path.join(root, x) for x in categories]
        cat_directories_files = [glob.glob(x + '/*.txt') for x in cat_directories]
        self.points_path = list(chain(*cat_directories_files))

        labels = [np.repeat(i, len(f)) for (i, f) in enumerate(cat_directories_files)]
        self.labels = list(chain(*labels))
        pass

    def __len__(self):
        return len(self.points_path)
    
    def __getitem__(self, index):
        label = self.labels[index]
        cat_points = load_points_from_category([self.points_path[index]])[0]
        cat_points = cat_points[0:self.npoints, 0:3]
        return cat_points, label
    pass

def expand_to_1000(files: list[str]):
    """
    將一個類別的資料個數擴展至 1000 個，方便處理
    """
    ll = len(files)
    x = 1000 // ll
    repeat_choice = np.repeat(range(ll), x)
    qq = [files[i] for i in repeat_choice]

    y = 1000 - (ll * x)
    random_choice = np.random.choice(ll, y)
    rr = [files[i] for i in random_choice]

    flatten_list = list(chain([*qq, *rr]))
    np.random.shuffle(flatten_list)
    return flatten_list

class MAMLModelNetDataset_KShot(Dataset):
    """
    n_way: 10
    k_shot: 5
    
    每次取資料包含 n_way 種類別各有 k_shot 個資料
    """
    def __init__(self, root, categories:list[str], args):
        self.root = root
        self.npoints = args.num_point
        # n_way
        self.num_category = 10
        self.k_shot = 5
        self.categories_name = categories
        cat_directories = [os.path.join(root, x) for x in categories]
        cat_directories_files = [glob.glob(x + '/*.txt') for x in cat_directories]
        self.cat_dir_files_1000 = [expand_to_1000(x) for x in cat_directories_files]
        pass

    def __len__(self):
        return 200
    
    def __getitem__(self, index):
        start = index * 5
        end = (index + 1) * 5
        cat_points = [load_points_from_category(x[start:end]) for x in self.cat_dir_files_1000]
        cat_points = np.array(cat_points)
        cat_points = cat_points.reshape([50, 10000, 6])
        cat_points = cat_points[:, 0:self.npoints, :]
        labels = np.arange(10).repeat(5)
        return cat_points, labels
    pass


def split_four_categoris(data_root:str, args):
    """
    將 ModelNet40 分成 4 組，每組 10 種類別

    - 取其中三個用於 Task 訓練
    - 剩餘一個：
        - 取其 train set 做 fine tune
        - 取其 test set 輸出模型評估指標
    """
    file_paths = os.path.join(data_root, 'modelnet40_shape_names.txt')
    datasets = []
    with open(file_paths, 'r') as f:
        cat = [line.rstrip() for line in f]
        cat.sort()
        ii = [x * 10 for x in range(4)]
        cat = [cat[x: x + 10] for x in ii]
        datasets = [MAMLModelNetDataset(data_root, x, args) for x in cat]
    return datasets
