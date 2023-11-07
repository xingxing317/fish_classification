import os
import torch
import torch.distributed as dist
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
import numpy as np
from utils import train_one_epoch, evaluate, cleanup
from my_datasets import QrDataset
from my_nets import Res50XQr


def start(args):
    batch_size = args['batch_size']
    nw = 16
    qr_tensor = torch.load(args['qr'])
    soft_label = torch.load(args['sf'])
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device('cuda', local_rank)

    # if local_rank == 0:
    #     tb_writer = SummaryWriter()
    #     if os.path.exists("./weights") is False:
    #         os.makedirs("./weights")
    device = torch.device('cuda:1')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(300),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = QrDataset(root_path=args['root_path'],
                              txt_path='train.txt',
                              qr_tensor=qr_tensor,
                              soft_label=soft_label,
                              neg_num=36,
                              transform=data_transform["train"])

    val_dataset = QrDataset(root_path=args['root_path'],
                            txt_path='val.txt',
                            qr_tensor=qr_tensor,
                            soft_label=soft_label,
                            neg_num=36,
                            transform=data_transform["val"])
    model = Res50XQr(device=device)

    if os.path.exists(args['weights']):
        weights_dict = torch.load(args['weights'])
        model.load_state_dict(weights_dict)
        #model.load_state_dict({k.replace('module.', ''): v for k, v in weights_dict.items()})
        print('Using' + args['weights'] + ' to initial !!!')
    model.cuda(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.00001)
    lf = lambda x: ((1 + math.cos(x * math.pi / args['epochs'])) / 2) * (1 - args['lrf']) + args['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # model = torch.nn.parallel.DistributedDataParallel(model)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=nw)

    scaler = torch.cuda.amp.GradScaler()
    

    # loss, nce_loss, mse_loss, train_genus_acc, spec_acc, test_gen_acc, test_spec_acc
    train_log = torch.zeros(7, args['epochs'])

    for epoch in range(args['epochs']):
        # train_sampler.set_epoch(epoch=epoch)
        mean_loss, num_train = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        scheduler.step()
        sum_num, sum_num_genus = evaluate(model=model, data_loader=val_loader, device=device)
        acc = sum_num / len(val_dataset)
        acc_genus = sum_num_genus / len(val_dataset)
        acc_train = num_train[0] / len(train_dataset)
        acc_train_genus = num_train[1] / len(train_dataset)

        train_log[:3, epoch] = mean_loss
        train_log[3, epoch] = acc_train_genus
        train_log[4, epoch] = acc_train
        train_log[5, epoch] = acc_genus
        train_log[6, epoch] = acc

        print("[epoch {}]  accuracy: train {}  genus_train {} test {} genus_test {}".format(epoch,
        np.round(acc_train.cpu().item(), 5), np.round(acc_train_genus.cpu().item(), 5),
        np.round(acc, 5), np.round(acc_genus, 5)))

        tags = ["loss_summary", "loss_nce", "loss_mse", "train_acc_genus", "train_acc_species",
                "val_acc_genus", "val_acc_species"]
        tb_writer.add_scalars('loss', {tags[0]: mean_loss[0].item(),
                                       tags[1]: mean_loss[1].item(),
                                       tags[2]: mean_loss[2].item()}, epoch)
        tb_writer.add_scalars('train_acc', {tags[3]: acc_train_genus, tags[4]: acc_train}, epoch)
        tb_writer.add_scalars('val_acc', {tags[5]: torch.tensor(acc_genus), tags[6]: torch.tensor(acc)}, epoch)

        torch.save(model.state_dict(), "./weights/Res50X-{}.pth".format(epoch))
    torch.save(train_log, 'Res50X_train_log.tensor')
    #cleanup()


if __name__ == '__main__':
    arg = {'epochs': 300,
           'lr': 0.01,
           'lrf': 0.01,
           'root_path': '../../SelWildFish',
           'qr': 'rqr_label_456.tensor',
           'sf': 'rsoft_label_456.tensor',
           'batch_size': 192,
           'weights': ''}
    start(args=arg)
