import json
import numpy as np
import random
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
import sys
from nce_loss import InfoNCE


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    nce_loss = InfoNCE(negative_mode='paired')
    mse_loss = torch.nn.MSELoss()

    sum_num = torch.zeros(2).to(device)  # species, genus
    mean_loss = torch.zeros(3).to(device)  # loss, nce_loss, mse_loss
    # if dist.get_rank() == 0:
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            pred = model(data[0].to(device))
            _, pred_train = torch.max(pred, dim=1)
            sum_num[0] += torch.eq(pred_train, data[2].to(device)).sum()
            target = data[3].to(device)
            sum_num[1] += torch.ge(target[torch.arange(len(target)), pred_train.to(device)], 0.49).sum()
            #target = torch.nn.functional.one_hot(data[2], num_classes=685)

            nce = nce_loss(pred, target.to(device), data[1].to(device))
            mse = 8 * mse_loss(pred*0.1, target.to(device))
            loss = 9 * nce + 1 * mse
            scaler.scale(loss).backward()
        #loss = reduce_value(loss, average=True)
        mean_loss[0] = (mean_loss[0] * step + loss.detach()) / (step + 1)  # update mean losses
        mean_loss[1] = (mean_loss[1] * step + nce.detach()) / (step + 1)
        mean_loss[2] = (mean_loss[2] * step + mse.detach()) / (step + 1)
        # if dist.get_rank() == 0:
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss[0].item(), 4))

        # if not torch.isfinite(loss):
        #     print('WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        optimizer.zero_grad()

    return mean_loss, sum_num



@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    sum_num_genus = torch.zeros(1).to(device)
    # 在进程0中打印验证进度
    #if is_main_process():
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            pred = model(data[0].to(device))
            _, pred = torch.max(pred, dim=1)
            sum_num += torch.eq(pred, data[2].to(device)).sum()
            target = data[3].to(device)
            sum_num_genus += torch.ge(target[torch.arange(len(target)), pred.to(device)], 0.49).sum()
    # if device != torch.device("cpu"):
    #     torch.cuda.synchronize(device)

    # sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(), sum_num_genus.item()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."

        intersection = (predict * target).sum(-1).sum()
        union = (predict + target).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score
