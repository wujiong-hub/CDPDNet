import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, glob
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model.CDPDNet.cdpdnet import CDPDNet
from model.CDPDNet.cdpdnet import DinoEncoder
from model.CDPDNet.initialization import InitWeights_He
from dataset.dataloader import get_loader_trainsamples

torch.multiprocessing.set_sharing_strategy('file_system')


tasknames = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '07': 5, '08': 6, '09': 7, '10': 8, '12': 9, '13': 10} 

def setup_distributed(args):
    """Initialize distributed training"""
    if args.dist:
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.global_rank
        )
        
        # Log distributed training information
        print(f"[Rank {args.global_rank}/{args.world_size}] Initialized process group")
        print(f"[Rank {args.global_rank}] Local Rank: {args.local_rank}")
        print(f"[Rank {args.global_rank}] Node Rank: {args.node_rank}")
        print(f"[Rank {args.global_rank}] Master: {args.master_addr}:{args.master_port}")


def train(args, train_loader, model, dino, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        dino_features = dino(x)
        taskid = []
        for i in range(len(name)):
            taskid.append(int(tasknames[name[i][:2]]))
        logit_map = model(x, dino_features, torch.tensor(taskid, dtype=torch.long, device=args.device))
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item() 
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        print(name, image.shape)
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = F.sigmoid(pred)
        
        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    if args.local_rank == 0:
        with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                print(content)
                f.write(content)
                f.write('\n')
            content = 'Average | '
            for i in range(NUM_CLASS):
                content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][organ-1] / ave_organ_dice[1][organ-1])
            print(content)
            f.write(content)
            f.write('\n')
            
            


def process(args):
    rank = 0
    # Calculate ranks for distributed training
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.node_rank = int(os.environ.get("SLURM_NODEID", 0))
    args.global_rank = args.node_rank * args.gpus_per_node + args.local_rank
    
    # Setup distributed training
    if args.dist:
        setup_distributed(args)
    
    # Set device 
    args.device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(args.device)

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    base_num_features = 32
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    net_conv_kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d 
    model = CDPDNet([args.roi_x, args.roi_y, args.roi_z], 11, [1, 2, 4], base_num_features, NUM_CLASS,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    dino = DinoEncoder(img_size=(args.roi_x, args.roi_y, args.roi_z))

    
    if args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding)
        task_embedding = torch.load(args.task_embedding)
        model.organ_embedding.data = word_embedding.float()
        model.task_embedding.data = task_embedding.float()
        print('load word and task embedding')
    

    model.to(args.device)
    dino.to(args.device)
    model.train()
    dino.eval()
    if args.dist:
        model = DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

        dino = DistributedDataParallel(dino, device_ids=[args.device],output_device=args.local_rank)
        

    # criterion and optimizer
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    

    if args.resume:
        #checkpoint = torch.load(args.resume, map_location="cuda:0")
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, dino, optimizer, loss_seg_DICE, loss_seg_CE)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', np.array(scheduler.get_lr()), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir('out/' + args.log_name):
                os.mkdir('out/' + args.log_name)
            torch.save(checkpoint, 'out/' + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int, default= int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='checkpoints', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default=None, help='out/checkpoints/epoch.pth')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    parser.add_argument('--task_embedding', default='./pretrained_weights/txt_task_encoding.pth', 
                        help='The path of task embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=50, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['dataset']) # 'PAOT', 'felix'

    ### please check this argment carefully
    parser.add_argument('--data_root_path', default='../CLIP-Deep-Driven/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for dataset index for training')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')

    # for distributed training
    parser.add_argument('--world_size', type=int, default=int(os.environ.get("WORLD_SIZE", 1)),
                       help='Total number of processes to run')
    parser.add_argument('--master_addr', type=str, default=os.environ.get("MASTER_ADDR", "localhost"),
                       help='Master node address')
    parser.add_argument('--master_port', type=str, default=os.environ.get("MASTER_PORT", "29500"),
                       help='Master node port')
    parser.add_argument('--gpus_per_node', type=int, default=1,
                       help='Number of GPUs per node')
    
    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()
