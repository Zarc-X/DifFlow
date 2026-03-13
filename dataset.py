import os
import json
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GenADDataset(Dataset):
    """
    通用异常检测数据集加载器
    支持 MVTec AD, VisA, Real-IAD.
    通过 dataset_type 参数区分不同的加载解析逻辑。
    """
    def __init__(self, root_dir, category, dataset_type='mvtec', is_train=True, transform=None, gt_transform=None, meta_file=None):
        self.root_dir = root_dir
        self.category = category
        self.dataset_type = dataset_type.lower()
        self.is_train = is_train
        self.meta_file = meta_file
        
        # 默认数据预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # 扩散模型常用[-1, 1]归一化
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
        else:
            self.transform = transform
            
        if gt_transform is None:
            self.gt_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.gt_transform = gt_transform

        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = [] # 0: good, 1: anomaly

        categories_to_load = []
        if isinstance(self.category, list):
            categories_to_load = self.category
        elif self.category == 'all':
            categories_to_load = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        else:
            categories_to_load = [self.category]

        if self.dataset_type == 'mvtec':
            phase = 'train' if self.is_train else 'test'
            for cat in categories_to_load:
                category_dir = os.path.join(self.root_dir, cat, phase)
                
                if not os.path.exists(category_dir):
                    print(f"Warning: Dataset path not found, skipping: {category_dir}")
                    continue
                    
                defect_types = os.listdir(category_dir)
                for defect in defect_types:
                    defect_dir = os.path.join(category_dir, defect)
                    if not os.path.isdir(defect_dir): continue
                    
                    img_files = glob(os.path.join(defect_dir, "*.png")) + \
                                glob(os.path.join(defect_dir, "*.jpg")) + \
                                glob(os.path.join(defect_dir, "*.bmp"))
                                
                    for img_file in img_files:
                        img_paths.append(img_file)
                        if defect == 'good':
                            labels.append(0)
                            gt_paths.append(None)
                        else:
                            labels.append(1)
                            gt_dir = os.path.join(self.root_dir, cat, 'ground_truth', defect)
                            gt_file = os.path.join(gt_dir, os.path.basename(img_file).rsplit('.', 1)[0] + '_mask.png')
                            gt_paths.append(gt_file)

        elif self.dataset_type == 'visa':
            # VisA 的结构类似 mvtec，但是掩码没有 _mask 后缀，且有的根目录多了 1cls 的层级
            phase = 'train' if self.is_train else 'test'
            for cat in categories_to_load:
                category_dir = os.path.join(self.root_dir, cat, phase)
                
                if not os.path.exists(category_dir):
                    print(f"Warning: Dataset path not found, skipping: {category_dir}")
                    continue
                    
                defect_types = os.listdir(category_dir)
                for defect in defect_types:
                    defect_dir = os.path.join(category_dir, defect)
                    if not os.path.isdir(defect_dir): continue
                    
                    img_files = glob(os.path.join(defect_dir, "*.JPG")) + \
                                glob(os.path.join(defect_dir, "*.png"))
                    
                    for img_file in img_files:
                        img_paths.append(img_file)
                        if defect == 'good':
                            labels.append(0)
                            gt_paths.append(None)
                        else:
                            labels.append(1)
                            gt_dir = os.path.join(self.root_dir, cat, 'ground_truth', defect)
                            # VisA 的 gt 就是同名 png
                            gt_file = os.path.join(gt_dir, os.path.basename(img_file).rsplit('.', 1)[0] + '.png')
                            gt_paths.append(gt_file)
            
            with open(self.meta_file, 'r') as f:
                for line in f:
                    meta = json.loads(line)
                    # 区分 train/test 的标志可能不在 json 中直接体现，如果只是过滤 train 可以考虑文件名或者预先分开的 meta_file
                    img_paths.append(meta['filename'])
                    labels.append(meta['label'])
                    if meta['label'] == 1 and 'maskname' in meta:
                        gt_paths.append(meta['maskname'])
                    else:
                        gt_paths.append(None)
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        return img_paths, gt_paths, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # 测试阶段返回标签和GT掩码
        if not self.is_train:
            gt_path = self.gt_paths[idx]
            if gt_path is not None and os.path.exists(gt_path):
                gt = Image.open(gt_path).convert('L')
                gt = self.gt_transform(gt)
            else:
                gt = torch.zeros([1, img.shape[1], img.shape[2]]) # 无掩码给全0
            return img, label, gt, img_path
            
        return img, label
