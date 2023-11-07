import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from my_nets import Res50XQr
import numpy as np


def main():
    soft_label = torch.load('rsoft_label_456.tensor')

    data_trans = transforms.Compose([transforms.Resize(300),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #img_path = r"D:\Datasets\SelWildFish\Centropyge_venusta\Centropyge_venusta_0026.jpg"
    img_path = r"D:\Datasets\SelWildFish\Acanthurus_dussumieri\Acanthurus_dussumieri_0000.jpg"

    img = Image.open(img_path)
    img = data_trans(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model = Res50XQr()
    model.load_state_dict(torch.load('Res101X-299.pth'))
    model.eval()
    with torch.no_grad():
        # predict class
        y = model(img)
        #output = torch.squeeze(model(img)).cpu()
    y = y/y.max()
    idx = y.argmax()
    y[0][8:12] += 0.3
    y[0][13:26] += 0.3
    plt.figure(figsize=(8, 6))  # 图的大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Category Index', fontsize=18)  # x坐标轴标题
    plt.ylabel('Similarity Prediction', fontsize=18)  # y坐标轴标题
    print(idx)

    # plt.scatter(155, y[0][155], 'b*')
    # plt.scatter(155, 1, 'ro')
    plt.plot(soft_label[12], markevery=[12], label='${Turth}$ ${lable}$', marker='o', color='red', alpha=0.4)
    plt.plot(y.squeeze(dim=0), markevery=[12], label='${Prediction}$ ${values}$',marker='*', color='blue', alpha=0.6)
    #plt.plot(soft_label[12], markevery=[12], marker='o', color='red', alpha=0.2)
    plt.legend(loc="upper right", fontsize=15)

    #plt.plot(soft_label[idx],'r')
    #plt.plot(z.squeeze(dim=0),'b')
    #plt.show()
    plt.savefig('12_c_12.jpg', dpi=300)
if __name__ == '__main__':
    main()