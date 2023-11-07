import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import os


class QrDataset(Dataset):
    def __init__(self, root_path, txt_path, qr_tensor, soft_label, neg_num=None, transform=None):
        f = open(txt_path, 'r')
        al = f.readlines()
        imgs_path = []
        imgs_label = []
        for item in al:
            img, label = item.split(' ')
            imgs_path.append(os.path.join(root_path,img))
            imgs_label.append(int(label.strip()))
        f.close()
        self.imgs_path = imgs_path
        self.qr_label = qr_tensor
        self.soft_label = soft_label
        self.imgs_label = torch.tensor(imgs_label)
        self.neg_num = neg_num
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        img = Image.open(self.imgs_path[item])
        label = self.imgs_label[item]
        qr = self.qr_label[label]
        sf = self.soft_label[label]
        if self.neg_num is not None:
            i1 = 1 - sf
            prob = F.normalize(i1, dim=0)
            idx = prob.multinomial(num_samples=self.neg_num, replacement=False)
            neg_sample = self.soft_label[idx]

            return self.transform(img), neg_sample, label, sf
        else:

            return self.transform(img), qr, label, sf


if __name__ == '__main__':
    from torchvision import transforms
    qr_tensor = torch.load('rqr_label_456.tensor')
    soft_label = torch.load('rsoft_label_456.tensor')
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(300),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    fish_data = QrDataset(r'D:\Datasets\SelWildFish', 'val.txt',  qr_tensor, soft_label, 6, data_transform['val'])
    print(len(fish_data))