from __future__ import print_function, absolute_import
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
import os.path as osp
import pandas as pd
from trainers import Trainer
from datasets import ActionData
from utils.logging import Logger
from torch.backends import cudnn
from utils.meters import AverageMeter
from torch.utils.data import DataLoader
from utils.lr_scheduler import WarmupMultiStepLR
from models import CognitiveProcessor, PercepProcessor, MHP, LipschitzGraph

def set_seed(seed):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def train(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    set_seed(args.seed)

    num_frames = args.num_frames    # 50
    stride = args.stride            # 25
    edge_dim = args.edge_dim        # 8
    num_neighbor = args.neighbors   # 6

    Cog = CognitiveProcessor(input_dim=64, convert_type=args.convert_type, num_features=num_frames,
                             n_channels=edge_dim, k=num_neighbor)
    Per = PercepProcessor(only_fuse=True)
    Mot = LipschitzGraph(edge_channel=edge_dim, n_layers=args.layers, act_type=args.act,
                         num_features=num_frames, norm=args.norm, get_logdets=args.get_logdets)
    model = MHP(p=Per, c=Cog, m=Mot, no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern)
    model = model.cuda()

    train_path = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, delimiter=',')
    train_path = train_path.drop(0)
    speaker_path = [path for path in list(train_path.values[:, 1])] + [path for path in list(train_path.values[:, 2])]
    listener_path = [path for path in list(train_path.values[:, 2])] + [path for path in list(train_path.values[:, 1])]

    train_neighbour_path = os.path.join(args.data_dir, 'neighbour_emotion_train.npy')
    train_neighbour = np.load(train_neighbour_path)

    neighbors = {
        'speaker_path': speaker_path,
        'listener_path': listener_path,
        'neighbors': train_neighbour
    }

    dataset = ActionData(root=args.data_dir, data_type='train',  neighbors=None,
                         neighbor_pattern=args.neighbor_pattern, num_frames=num_frames, stride=stride)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    trainer = Trainer(model=model, neighbors=neighbors, loss_name=args.loss_name,
                      no_inverse=args.no_inverse, neighbor_pattern=args.neighbor_pattern,
                      num_frames=num_frames, stride=stride, loss_mid=args.loss_mid,
                      cal_logdets=args.get_logdets)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, gamma=args.gamma, warmup_factor=args.warmup_factor,
                                     milestones=args.milestones, warmup_iters=args.warmup_step)

    for epoch in range(100):
        lr_scheduler.step(epoch)
        print('Epoch [{}] LR [{:.6f}]'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        trainer.train(epoch=epoch, dataloader=dataloader, optimizer=optimizer, train_iters=args.train_iters)
        if epoch > 0 and epoch % 5 == 0:
            torch.save(model, osp.join(args.logs_dir, "mhp-epoch{0}-seed{1}.pth").format(epoch, args.seed))

def test():
    sys.stdout = Logger(osp.join(args.logs_dir, 'test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    set_seed(args.seed)

    num_frames = args.num_frames
    stride = args.stride

    test_batch = len([i for i in range(0, 750 - num_frames + 1, stride)])

    model_pth = args.model_pth

    save_base = os.path.join(args.data_dir, 'outputs', model_pth.split('/')[-2])
    if not os.path.isdir(save_base):
        os.makedirs(save_base)

    testset = ActionData(root=args.data_dir, data_type='test', neighbors=None,
                         neighbor_pattern=args.neighbor_pattern, num_frames=num_frames, stride=stride)
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=False, )

    model = torch.load(model_pth)
    model = model.cuda()

    val_path = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), header=None, delimiter=',')
    val_path = val_path.drop(0)
    speaker_path = [path for path in list(val_path.values[:, 1])] + [path for path in list(val_path.values[:, 2])]
    listener_path = [path for path in list(val_path.values[:, 2])] + [path for path in list(val_path.values[:, 1])]

    val_neighbour_path = os.path.join(args.data_dir, 'neighbour_emotion_test.npy')
    val_neighbour = np.load(val_neighbour_path)

    neighbors = {
        'speaker_path': speaker_path,
        'listener_path': listener_path,
        'neighbors': val_neighbour
    }

    trainer = Trainer(model=model, neighbors=neighbors, neighbor_pattern=args.neighbor_pattern,
                      no_inverse=args.no_inverse)

    trainer.threshold = 0.06
    trainer.test(testloader, modify=args.modify, save_base=save_base)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Actiton Generation")
    # pattern
    parser.add_argument('--test', action='store_true')
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    # model
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--no-inverse', action='store_true')
    parser.add_argument('--convert-type', type=str, default='indirect')
    parser.add_argument('--edge-dim', type=int, default=8)
    parser.add_argument('--neighbors', type=int, default=6)
    # optimizer
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 15])
    parser.add_argument('--warmup-factor', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--loss-name', type=str, default='MSE')
    parser.add_argument('--train-iters', type=int, default=100)
    parser.add_argument('--get-logdets', action='store_true')
    parser.add_argument('--loss-mid', action='store_true')
    parser.add_argument('--neighbor-pattern', type=str, default='nearest', choices=['nearest', 'pair', 'all'])
    parser.add_argument('--num-frames', type=int, default=50)
    parser.add_argument('--stride', type=int, default=25)
    # testing configs
    parser.add_argument('--modify', action='store_true')
    parser.add_argument('--model-pth', type=str, metavar='PATH', default=' ')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../data/react_clean'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    if args.test:
        test()
    else:
        train(args)