from torchvision import datasets, transforms
import torch
import os
from PIL import Image

def is_valid_folder(folder_name):
    """检查文件夹是否为有效类别文件夹（非隐藏文件夹）"""
    return not folder_name.startswith('.')

def is_valid_file(file_name):
    """检查文件是否为有效图像文件"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    return (not file_name.startswith('.')) and (os.path.splitext(file_name)[1].lower() in valid_extensions)

def find_classes(dir):
    """自定义类别发现函数，过滤隐藏文件夹"""
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and is_valid_folder(d)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    """自定义数据集构建函数，只包含有效图像文件"""
    images = []
    dir = os.path.expanduser(dir)
    
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
            
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    
    return images

class RobustImageFolder(datasets.VisionDataset):
    """完全自定义的图像数据集加载器，确保只加载有效文件和文件夹"""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        classes, class_to_idx = find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 images in subfolders of: {self.root}\n"
                               "Supported image extensions are: " + ",".join(valid_extensions))
        
        self.loader = datasets.folder.default_loader
        self.extensions = datasets.folder.IMG_EXTENSIONS
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        
        # 打印加载信息
        print(f"从 {self.root} 加载数据集:")
        print(f"  找到 {len(self.classes)} 个类别: {self.classes}")
        print(f"  共 {len(self.samples)} 个有效样本")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data = RobustImageFolder(
        root=os.path.join(root_path, dir),
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        **kwargs
    )
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data = RobustImageFolder(
        root=os.path.join(root_path, dir),
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=False,
        **kwargs
    )
    return test_loader