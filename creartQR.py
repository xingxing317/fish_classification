import os
import shutil

import numpy
import torch
import numpy as np
from scipy import linalg


def create_qr(length: int, num: int, re=None):

    print('start')
    device = torch.device('cuda', 0)
    if re is None:
        init = torch.rand(4000, length).to(device)
        init[init > 0.5] = 1
        init[init < 0.5] = 0
        qr = init
        #qr = init[torch.where((length / 2 - 12 < init.sum(dim=1)) & (init.sum(dim=1) < length / 2 + 12))]
        for i in range(num):
            cox = torch.cosine_similarity(qr.unsqueeze(dim=0), qr.unsqueeze(dim=1), dim=2)
            tmp = qr[torch.where(cox[0] < 0.6)]
            qr = torch.vstack((tmp, qr[0]))
            if len(qr) == len(tmp):
                break
        result = qr
        print(len(result))
    else:
        result = re.to(device)

    for i in range(5000):
        init = torch.rand(4000, length).to(device)
        init[init >= 0.5] = 1
        init[init < 0.5] = 0
        qr = init
        #qr = init[torch.where((length / 2 - 12 < init.sum(dim=1)) & (init.sum(dim=1) < length / 2 + 12))]

        if len(qr) >= 1:
            result = torch.vstack((result, qr))
            for i in range(num):
                cox = torch.cosine_similarity(result.unsqueeze(dim=0), result.unsqueeze(dim=1), dim=2)
                tmp = result[torch.where(cox[0] < 0.6)]
                result = torch.vstack((tmp, result[0]))
            if len(result) >= num:
                break
        print(len(result))

    return result


def sel_copy(selected: dict):
    for genu, spec in selected.items():
        for s in spec:
            source_dir = os.path.join(r'D:\Datasets\WildFish', genu+'_'+s)
            target_dir = os.path.join(r'D:\Datasets\SelWildFish', genu+'_'+s)
            shutil.copytree(source_dir, target_dir)


if __name__ == '__main__':
    #result = create_qr(64, 456)
    # re = torch.load('qr497.tensor')
    #torch.save(result, 'qr456.tensor')
    data_path = r'D:\Datasets\SelWildFish'
    full_name = os.listdir(data_path)
    # # re = np.load('qre.npy')
    # # qr = torch.from_numpy(re)
    # # #f1 = open('open-set/trainx.txt', 'w')
    # # #f2 = open('/home/xgeng/Datasets/WildFish/open-set/train.txt', 'r')
    # f2 = open('train.txt', 'r')
    # al = f2.readlines()  # + f2.readlines()
    static = {}
    # # label = {}
    # # label_np = np.zeros((200, 256))
    #label_soft = np.zeros((456, 456))
    # #label_np = np.zeros((1000, 256))

    for item in full_name:
        #sp, la = item.split(' ')
        #sp, la = item.split('\t')
        #sp = sp.split('/')[-2]
        #sp = '_'.join((sp.split('_')[0], sp.split('_')[1]))
        #f11 = os.path.join(sp, item)
        #f1.write(f11)
        #sp = sp.split('/')[0]
        #label[sp] = la.strip()
        genu, spec = item.split('_')
        #genu, spec, _ = item.split('_')
        #genu, spec = item.split('/')[0].split('_')[:2]
        if genu in static.keys():
            if spec not in static[genu]:
                static[genu].append(spec)
        else:
            tmp = []
            tmp.append(spec)
            static[genu] = tmp

    selected = static
    # sum = 0
    # for genu, spec in static.items():
    #     l = len(spec)
    #     if l > 4:
    #         selected[genu] = spec
    #         sum += l

            # f2.close()
    # #f1.close()
    #qr = create_qr(128, 1431, re)
    #qr = torch.load('qr497.tensor').cpu().numpy()
    #qr = torch.randn(497, 64)
    qr = torch.randn(497, 196)
    genus = linalg.orth(qr[:41].T).T
    start_idx = 41
    end_idx = 41
    #guide_label = np.zeros((456, 128))
    guide_label = np.zeros((456, 392))
    guide_label_idx = 0
    name_mapping_idx = {}
    genus_idx = 0
    spec_per_genus = np.zeros(41)
    for k, v in selected.items():
        end_idx += len(v)
        orth = linalg.orth(qr[start_idx:end_idx].T).T
        orth_idx = 0
        for spec in v:
            name_mapping_idx[k+'_'+spec] = guide_label_idx
            guide_label[guide_label_idx] = np.concatenate((genus[genus_idx], orth[orth_idx]), axis=0)
            orth_idx += 1
            guide_label_idx += 1
        spec_per_genus[genus_idx] = orth_idx
        genus_idx += 1
        start_idx = end_idx

    g = torch.from_numpy(guide_label)
    soft_label = torch.cosine_similarity(g.unsqueeze(dim=0), g.unsqueeze(dim=1), dim=2)

    # txt = ['train.txt', 'val.txt']
    # for t in txt:
    #     tmp = []
    #     with open(t, 'r') as f:
    #         al = f.readlines()
    #         for i in al:
    #             img_path = i.split(' ')[0]
    #             sp = img_path.split('/')[-2]
    #             img_label = name_mapping_idx[sp]
    #             tmp.append(img_path+' '+str(img_label)+'\n')
    #     with open('x'+t, 'w') as f:
    #         f.writelines(tmp)

    torch.save(g, 'qr_label_456_14.tensor')
    torch.save(soft_label, 'soft_label_456_14.tensor')


