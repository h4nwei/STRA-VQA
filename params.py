import argparse
from pathlib import Path




parser = argparse.ArgumentParser(description='UGC VQA challenge (ICMEw2021)')

# Hardware specifications
parser.add_argument('--seed', type=int, default=19950801, help='random seed')
parser.add_argument('--n_threads', type=int, default=32, help='number of threads for data loading')

# Data specifications

parser.add_argument('--dataset_name', type=str, default='ETRI', help='dataset root directory')
parser.add_argument('--data_root', type=Path, default='./etri/data/VSFA_resnet50_iqapretrain_ms/', help='feature root directory')
parser.add_argument('--data_info_path', type=Path, default='./etri/code_baseline/data/E-LIVE_new/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='MCL'', help='dataset root directory')
# parser.add_argument('--frame_root', type=Path, default='/data/zhw/vqa/MCL-V/frames/', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='./etri/data/MCL/VSFA_resnet50_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='./etri/code_baseline/data/MCL/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='AVT', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='./etri/data/AVT/VSFA_resnet50_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--data_info_path', type=Path, default='./etri/code_baseline/data/AVT/', help='dataset directory')
# parser.add_argument('--dataset_name', type=str, default='LIVE_HFR_ViT_Test_VFIPS', help='dataset root directory')
# parser.add_argument('--data_root', type=Path, default='./etri/data/LIVEHFR/VSFA_resnet50_iqapretrain_ms/', help='dataset root directory')
# parser.add_argument('--frame_root', type=Path, default='/data/zhw/vqa/LIVE-HFR/frames/', help='frame root directory')
# parser.add_argument('--data_info_path', type=Path, default='./etri/code_baseline/data/LIVEHFR_final/', help='dataset directory')
parser.add_argument('--max_len', type=int, default=600, help='dataset directory')
# parser.add_argument('--feat_dim', type=int, default=1024, help='dataset directory')
# parser.add_argument('--data_root', type=Path, default='../../icme_data/resnet50feat', help='dataset directory')
# parser.add_argument('--train_file', type=Path, default='./ugcset_mos.json', help='train file path, video_name-mos')
parser.add_argument("--clip_per_video", type=int, default=1, help='number of clip per video for training')
parser.add_argument("--frame_per_clip", type=int, default=12, help='number of frame per clip') #60
parser.add_argument("--patch_size", type=int, default=256, help='patch size of per clip')
parser.add_argument("--temporal_stride", type=int, default=1, help='number of interval of the clip')
parser.add_argument("--clip_len", type=int, default=24, help='number of interval of the clip')#24
parser.add_argument("--frame_interval", type=int, default=12, help='number of interval of the clip')#12
parser.add_argument("--num_clips", type=int, default=1, help='number of interval of the clip')
parser.add_argument("--spatial_stride", type=int, default=400, help='number of stride of each cropped patch')
# Model specifications
modelparsers = parser.add_subparsers(dest='model', help='model arch name')





# Option for ViT method
stra_vqa_cmd = modelparsers.add_parser('STRA-VQA', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='VIT method')
stra_vqa_cmd.add_argument('--model_name', type=str, default='STRA-VQA', help='name of the model')
stra_vqa_cmd.add_argument('--d_feat', type=int, default=4096, help='input image size for ViT') ##default 448
stra_vqa_cmd.add_argument('--depth', type=int, default=5, help='number of transformer blocks') ## 5
stra_vqa_cmd.add_argument('--att_head', type=int, default=6, help='number of heads in multi-head attention layer') #default 16
stra_vqa_cmd.add_argument('--mlp_dim', type=int, default=128, help='dimension of the MLP (FeedForward) layer')
stra_vqa_cmd.add_argument('--dim_head', type=int, default=64, help='dimension of Q K V')
stra_vqa_cmd.add_argument('--output_channel', type=int, default=1, help='Output channel number')
stra_vqa_cmd.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
stra_vqa_cmd.add_argument('--pool', type=str, default='reg', help='output result')
stra_vqa_cmd.add_argument('--emb_dropout', type=float, default=0.1, help='embedding dropout rate')


# Training specifications
parser.add_argument('--test_every', type=int, default=1, help='do test per every N epochs')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')#default 128
parser.add_argument('--test_batch', type=int, default=4, help='input batch size for test')

# Testing specifications
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--pre_train_path', type=str, default=None, help='where saved trained checkpoints')
parser.add_argument('--predict_res', type=str, default=None, help='where to save predicted results')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')#1e-4 defatule
parser.add_argument('--decay', type=str, default='20-40-60-80-100', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMW'),
                    help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1+1*Rank', help='loss function weights and types') #1*norm-in-norm

# Log specifications
parser.add_argument('--log_root', type=Path, default='./logs/', help='directory for saving model weights and log file')
parser.add_argument('--ckpt_root', type=Path, default='./ckpts/', help='dataset root directory')
parser.add_argument('--save_weights', type=int, default=1000, help='how many epochs to wait before saving model weights')
parser.add_argument('--save_scatter', type=int, default=1000, help='how many epochs to wait before saving scatter plot')


args = parser.parse_args(['stra-vqa'])

