import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
from monai.inferers import sliding_window_inference
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key, compute_HD95
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ

from model.CDPDNet.cdpdnet import CDPDNet
from model.CDPDNet.clipdino import DinoEncoder
from model.CDPDNet.initialization import InitWeights_He

torch.multiprocessing.set_sharing_strategy('file_system')

tasknames = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '07': 5, '08': 6, '09': 7, '10': 8, '12': 9, '13': 10}

class CombinedModel(nn.Module):
    def __init__(self, uniseg, dino):
        super().__init__()
        self.uniseg = uniseg
        self.dino = dino

    def forward(self, x, task_id):
        dino_features = self.dino(x)
        outputs = self.uniseg(x, dino_features, task_id)
        return outputs

def validation(model, ValLoader, val_transforms, args):
    save_dir = 'out/' + args.log_name + f'/test_{args.epoch}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir+'/predict', exist_ok=True)
    model.eval()
    dice_list, hd95_list = {}, {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        hd95_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]

        taskid = []
        for i in range(len(name)):
            taskid.append(int(tasknames[name[i][:2]]))

        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 2, lambda x: model(x, torch.tensor(taskid).cuda()), overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
            pred_hard_post = torch.tensor(pred_hard_post)
            
            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
                    dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                    hd95_organ = compute_HD95(pred[b,organ-1,:,:,:].cpu().data.numpy(), label[b,organ-1,:,:,:].numpy())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    hd95_list[template_key][0][organ-1] += hd95_organ
                    hd95_list[template_key][1][organ-1] += 1

                    content += '%s: %.4f, %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item(), hd95_organ)
                    print('%s: dice %.4f, hd95 %.4f, recall %.4f, precision %.4f.'%(ORGAN_NAME[organ-1], dice_organ.item(), hd95_organ, recall.item(), precision.item()))
            print(content)
        
        if args.store_result:
            pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            label_store = (label.numpy()).astype(np.uint8)
            pred_store = (pred_hard_post.numpy()).astype(np.uint8)
            np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1], 
                            pred=pred_store, label=label_store)
            
            ### testing phase for this function
            one_channel_label_v1, one_channel_label_v2 = merge_label(pred_hard_post, name)
            batch['one_channel_label_v1'] = one_channel_label_v1.cpu()
            batch['one_channel_label_v2'] = one_channel_label_v2.cpu()

            _, split_label = merge_label(batch["post_label"], name)
            batch['split_label'] = split_label.cpu()
            visualize_label(batch, save_dir + '/output/' + name[0].split('/')[0] , val_transforms)
            
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    ave_organ_hd95 = np.zeros((2, NUM_CLASS))

    with open('out/'+args.log_name+f'/test_{args.epoch}.txt', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                hd95 = hd95_list[key][0][organ-1] / hd95_list[key][1][organ-1]
                content += '%s: %.4f, %.4f, '%(ORGAN_NAME[organ-1], dice, hd95)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                ave_organ_hd95[0][organ-1] += hd95_list[key][0][organ-1]
                ave_organ_hd95[1][organ-1] += hd95_list[key][1][organ-1]
            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i], ave_organ_hd95[0][i] / ave_organ_hd95[1][i])
        print(content)
        f.write(content)
        f.write('\n')
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        print(np.mean(ave_organ_hd95[0] / ave_organ_hd95[1]))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_hd95[0] / ave_organ_hd95[1])))
        f.write('\n')
        


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='inference', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='./out/checkpoints/cdpd.pth', help='The path resume from checkpoint') 
    
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=["dataset"])
    parser.add_argument('--data_root_path', default='../datasetpath/', help='data root path') # please asign the root path of dataset when you run this script
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path') #it is fixed if you using the default testing dataset
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
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
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    parser.add_argument('--taskid', default=0, type=int)
    args = parser.parse_args()

    # prepare the CDPDNet
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
    cdpdnet = CDPDNet([96,96,96], 11, [1, 2, 4], base_num_features, NUM_CLASS,
                                len(net_num_pool_op_kernel_sizes),
                                conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    dino = DinoEncoder(img_size=(args.roi_x, args.roi_y, args.roi_z))
    
    #Load pre-trained weights
    store_dict = cdpdnet.state_dict()
    checkpoint = torch.load(args.resume, map_location="cuda:0")
    load_dict = checkpoint['net']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    cdpdnet.load_state_dict(store_dict)
    print('Use pretrained weights')

    cdpdnet.cuda()

    dino.eval()
    dino.cuda()
    model = CombinedModel(cdpdnet, dino)

    
    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader(args)

    validation(model, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
