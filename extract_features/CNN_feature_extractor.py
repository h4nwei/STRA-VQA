
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import extract
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import argparse
from pathlib import Path


class VideoDataset(Dataset):
    """
    Test or Validation
    """
    def __init__(self, video_names, video_folder):
        self.video_folder = video_folder
        self.video_names = video_names
        self.data_list = self._make_dataset()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _make_dataset(self):
        data_list = []
        for idx in range(len(self.video_names)):
            video_name = self.video_names[idx]
            frame_names = list(os.walk(os.path.join(self.video_folder, str(video_name))))[0]
            frame_paths = list(map(lambda x: os.path.join(frame_names[0], x),
                                              sorted(frame_names[2], key=lambda x: (x.split('.')[0]))))
            data_list.append((str(video_name), frame_paths))
        return data_list
        
    def __getitem__(self, index):
        video_name, frame_paths = self.data_list[index]
        
        len_video = len(frame_paths)
        frames = []
        for i in range(len_video):
            print(f'\t{video_name} image idx {i}')
            img = Image.open(frame_paths[i])
            img = self.transform(img)
            frames.append(img)
        transformed_data = torch.zeros([len_video, *frames[0].shape])
        for i in range(len_video):
            transformed_data[i] = frames[i]
        return video_name, transformed_data

    def __len__(self):
        return len(self.video_names)


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        # self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.features = models.resnet50(pretrained=True)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        for p in self.features.parameters():
            p.requires_grad = False
        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.h4 = None
        # self.feats = []
        # self.means = []
        # self.stds = []
        # self.s4_feats = []
        

    def forward(self, x):
        # features@: 7->res5c
        h0 = self.conv1(x)
        h0 = self.maxpool(self.relu(self.bn1(h0)))
        # h0 = self.relu(self.bn1(h0))

        h1 = self.layer1(h0)
        # print('shape of h1', h1.shape)
        diff_h1 = self.feat_diff(h1, self.h1) # feature difference
        mu_std_1 = globale_mean_std_pool2d(h1)  
        # print('shape of mu_std_1', mu_std_1.shape)
        self.h1 = h1
        # print('self.h1[0][0][0][1]', self.h1[0][0][0][1])
        t_h1 = torch.max(torch.std(diff_h1, dim=(2, 3)), 1)[0]
        
        # print('temporal of h1', t_h1)
        h1_st_var = local_mean_std_pool2d(diff_h1)
        # print('shape of h1_st_var', h1_st_var.shape)
        var_mu_std_1 = globale_mean_std_pool2d(h1_st_var)
        diff_mu_std_1 = globale_mean_std_pool2d(diff_h1) #
        # print('shape of diff_mu_std_1', diff_mu_std_1.shape)
            
        h2 = self.layer2(h1)
        diff_h2 = self.feat_diff(h2, self.h2)
        mu_std_2 = globale_mean_std_pool2d(h2)
        self.h2 = h2
        t_h2 = torch.max(torch.std(diff_h2, dim=(2, 3)), 1)[0]
        h2_st_var = local_mean_std_pool2d(diff_h2)
        var_mu_std_2 = globale_mean_std_pool2d(h2_st_var)
        diff_mu_std_2 = globale_mean_std_pool2d(diff_h2)
              
        h3 = self.layer3(h2)
        diff_h3 = self.feat_diff(h3, self.h3)
        mu_std_3 = globale_mean_std_pool2d(h3)
        self.h3 = h3
        t_h3 = torch.max(torch.std(diff_h3, dim=(2, 3)), 1)[0]
        h3_st_var = local_mean_std_pool2d(diff_h3)
        var_mu_std_3= globale_mean_std_pool2d(h3_st_var)
        diff_mu_std_3 = globale_mean_std_pool2d(diff_h3)
            
        h4 = self.layer4(h3)
        diff_h4 = self.feat_diff(h4, self.h4)
        mu_std_4 = globale_mean_std_pool2d(h4)
        self.h4 = h4
        t_h4 = torch.max(torch.std(diff_h4, dim=(2, 3)), 1)[0]
        h4_st_var = local_mean_std_pool2d(diff_h4)
        var_mu_std_4 = globale_mean_std_pool2d(h4_st_var)
        diff_mu_std_4 = globale_mean_std_pool2d(diff_h4)

        result_1 = {'h1_t': t_h1, 'diff_mu_std_1': diff_mu_std_1, 'mu_std_1': mu_std_1, 'var_mu_std_1': var_mu_std_1}
        result_2 = {'h2_t': t_h2, 'diff_mu_std_2': diff_mu_std_2, 'mu_std_2': mu_std_2, 'var_mu_std_2': var_mu_std_2}
        result_3 = {'h3_t': t_h3, 'diff_mu_std_3': diff_mu_std_3, 'mu_std_3': mu_std_3, 'var_mu_std_3': var_mu_std_3}
        result_4 = {'h4_t': t_h4, 'diff_mu_std_4': diff_mu_std_4, 'mu_std_4': mu_std_4, 'var_mu_std_4': var_mu_std_4}

        return result_1, result_2, result_3, result_4
    def feat_diff(self, feat_data, pre_feat_data):
        if pre_feat_data is None:
            return torch.zeros_like(feat_data)
        else:
            b_present = feat_data.shape[0]
            b_previous = pre_feat_data.shape[0]
            pre_feat = pre_feat_data[b_previous-b_present:,:]
            return (feat_data - pre_feat) 

def local_mean_std_pool2d(x):
    mean_x = nn.functional.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
    mean_x_square = nn.functional.avg_pool2d(x * x, kernel_size=3, padding=1, stride=1)
    var_x = mean_x_square - mean_x * mean_x 
    # std_x = torch.sqrt(var_x + 1e-8)
    return var_x / (mean_x + 1e-8) #spatial motion variation,

def globale_mean_std_pool2d(x):
    mean = nn.functional.adaptive_avg_pool2d(x, 1)
    std = global_std_pool2d(x)
    result = torch.cat([mean, std], 1)
    return result

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


class BCNN(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, input_dim=512):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x

class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        if config.std_modeling:
            outdim = 2
        else:
            outdim = 1
        if config.representation == 'BCNN':
            assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, outdim)
        else:
            # self.fc = nn.Linear(512, outdim)
            self.fc = nn.Linear(2048, outdim)

        if self.config.fc:
            # Freeze all previous layers.
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, x):
        """Forward pass of the network.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.config.representation == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)
            x = x.view(x.size()[0], x.size()[1])

        x = self.fc(x)

        if self.config.std_modeling:
            mean = x[:, 0]
            t = x[:, 1]
            var = nn.functional.softplus(t)
            return mean, var
        else:
            return x


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--get_scores', type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--backbone", type=str, default='resnet50')
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--network", type=str, default="basecnn") #basecnn or dbcnn
    parser.add_argument("--representation", type=str, default="NOTBCNN")

    parser.add_argument("--ranking", type=bool, default=True)  # True for learning-to-rank False for regular regression
    parser.add_argument("--fidelity", type=bool, default=True)  # True for fidelity loss False for regular ranknet with CE loss
    parser.add_argument("--std_modeling", type=bool,
                        default=True)  # True for modeling std False for not
    parser.add_argument("--std_loss", type=bool, default=True)
    parser.add_argument("--margin", type=float, default=0.025)

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./IQA_database/")
    parser.add_argument("--bid_set", type=str, default="./IQA_database/BID/")
    parser.add_argument("--clive_set", type=str, default="./IQA_database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./IQA_database/koniq-10k/")
    parser.add_argument("--spaq_set", type=str, default="./IQA_database/SPAQ/")

    parser.add_argument("--eval_bid", type=bool, default=True)
    parser.add_argument("--eval_clive", type=bool, default=True)
    parser.add_argument("--eval_koniq10k", type=bool, default=True)
    parser.add_argument("--eval_spaq", type=bool, default=True)

    parser.add_argument("--split_modeling", type=bool, default=False)

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str, default='train.txt') # train.txt | train_synthetic.txt | train_authentic.txt | train_sub2.txt | train_score.txt

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument("--database", type=str, default='KoNViD-1k')
    parser.add_argument("--frame_batch_size", type=int, default=64)
    parser.add_argument("--ith", type=int, default=0, help='start frame id')
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['C'], help='C K L N Y Q')
    parser.add_argument('--model_path', default='models/model_C', type=str, help='model path (default: models/model)')
    parser.add_argument('--video_path', default='data/test.mp4', type=str, help='video path (default: data/test.mp4)')

    return parser.parse_args()


def make_spatial_model():
    config = parse_config()
    model = BaseCNN(config)
    model = torch.nn.DataParallel(model).cuda()
    ckpt = './DataParallel-00008.pt'
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    return model

class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50'):
        super(CNNModel, self).__init__()
        if model == 'SpatialExtractor':
            print("use SpatialExtractor")
            # from SpatialExtractor.get_spatialextractor_model import make_spatial_model
            model = make_spatial_model()
            # self.features = nn.Sequential(*list(model.module.backbone.children())[:-2])
            self.features = model.module.backbone
            self.conv1 = self.features.conv1
            self.bn1 = self.features.bn1
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self.features.layer1
            self.layer2 = self.features.layer2
            self.layer3 = self.features.layer3
            self.layer4 = self.features.layer4
            for p in self.features.parameters():
                p.requires_grad = False
            self.h1 = None
            self.h2 = None
            self.h3 = None
            self.h4 = None
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

    def forward(self, x):
        # features@: 7->res5c
        h0 = self.conv1(x)
        h0 = self.maxpool(self.relu(self.bn1(h0)))
        # h0 = self.relu(self.bn1(h0))

        h1 = self.layer1(h0)
        # print('shape of h1', h1.shape)
        diff_h1 = self.feat_diff(h1, self.h1) # feature difference
        mu_std_1 = globale_mean_std_pool2d(h1)  
        # print('shape of mu_std_1', mu_std_1.shape)
        self.h1 = h1
        # print('self.h1[0][0][0][1]', self.h1[0][0][0][1])
        t_h1 = torch.max(torch.std(diff_h1, dim=(2, 3)), 1)[0]
        
        # print('temporal of h1', t_h1)
        h1_st_var = local_mean_std_pool2d(diff_h1)
        # print('shape of h1_st_var', h1_st_var.shape)
        var_mu_std_1 = globale_mean_std_pool2d(h1_st_var)
        diff_mu_std_1 = globale_mean_std_pool2d(diff_h1) #
        # print('shape of diff_mu_std_1', diff_mu_std_1.shape)
            
        h2 = self.layer2(h1)
        diff_h2 = self.feat_diff(h2, self.h2)
        mu_std_2 = globale_mean_std_pool2d(h2)
        self.h2 = h2
        t_h2 = torch.max(torch.std(diff_h2, dim=(2, 3)), 1)[0]
        h2_st_var = local_mean_std_pool2d(diff_h2)
        var_mu_std_2 = globale_mean_std_pool2d(h2_st_var)
        diff_mu_std_2 = globale_mean_std_pool2d(diff_h2)
              
        h3 = self.layer3(h2)
        diff_h3 = self.feat_diff(h3, self.h3)
        mu_std_3 = globale_mean_std_pool2d(h3)
        self.h3 = h3
        t_h3 = torch.max(torch.std(diff_h3, dim=(2, 3)), 1)[0]
        h3_st_var = local_mean_std_pool2d(diff_h3)
        var_mu_std_3= globale_mean_std_pool2d(h3_st_var)
        diff_mu_std_3 = globale_mean_std_pool2d(diff_h3)
            
        h4 = self.layer4(h3)
        diff_h4 = self.feat_diff(h4, self.h4)
        mu_std_4 = globale_mean_std_pool2d(h4)
        self.h4 = h4
        t_h4 = torch.max(torch.std(diff_h4, dim=(2, 3)), 1)[0]
        h4_st_var = local_mean_std_pool2d(diff_h4)
        var_mu_std_4 = globale_mean_std_pool2d(h4_st_var)
        diff_mu_std_4 = globale_mean_std_pool2d(diff_h4)

        result_1 = {'h1_t': t_h1, 'diff_mu_std_1': diff_mu_std_1, 'mu_std_1': mu_std_1, 'var_mu_std_1': var_mu_std_1}
        result_2 = {'h2_t': t_h2, 'diff_mu_std_2': diff_mu_std_2, 'mu_std_2': mu_std_2, 'var_mu_std_2': var_mu_std_2}
        result_3 = {'h3_t': t_h3, 'diff_mu_std_3': diff_mu_std_3, 'mu_std_3': mu_std_3, 'var_mu_std_3': var_mu_std_3}
        result_4 = {'h4_t': t_h4, 'diff_mu_std_4': diff_mu_std_4, 'mu_std_4': mu_std_4, 'var_mu_std_4': var_mu_std_4}

        return result_1, result_2, result_3, result_4
    def feat_diff(self, feat_data, pre_feat_data):
        if pre_feat_data is None:
            return torch.zeros_like(feat_data)
        else:
            b_present = feat_data.shape[0]
            b_previous = pre_feat_data.shape[0]
            pre_feat = pre_feat_data[b_previous-b_present:,:]
            return (feat_data - pre_feat) 



def get_features(video_data, frame_batch_size=16, device='cuda'):
    """feature extraction"""
    # extractor = torch.nn.DataParallel(ResNet50()).to(device)
    # torch.nn.DataParallel(model).cuda()
    extractor = CNNModel(model='SpatialExtractor')

    video_length = video_data.shape[0]
    # print('video length', video_length)
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    r1_t = torch.Tensor().to(device)
    r2_t = torch.Tensor().to(device)
    r3_t = torch.Tensor().to(device)
    r4_t = torch.Tensor().to(device)

    r1_diff_mu_std = torch.Tensor().to(device)
    r2_diff_mu_std = torch.Tensor().to(device)
    r3_diff_mu_std = torch.Tensor().to(device)
    r4_diff_mu_std = torch.Tensor().to(device)

    r1_mu_std = torch.Tensor().to(device)
    r2_mu_std = torch.Tensor().to(device)
    r3_mu_std = torch.Tensor().to(device)
    r4_mu_std = torch.Tensor().to(device)

    var1_mu_std = torch.Tensor().to(device)
    var2_mu_std = torch.Tensor().to(device)
    var3_mu_std = torch.Tensor().to(device)
    var4_mu_std = torch.Tensor().to(device)
    # output2 = torch.Tensor().to(device)
    extractor.eval()
    result = []
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            r1, r2, r3, r4 = extractor(batch)
            r1_t = torch.cat((r1_t, r1['h1_t']), 0)
            r2_t = torch.cat((r2_t, r2['h2_t']), 0)
            r3_t = torch.cat((r3_t, r3['h3_t']), 0)
            r4_t = torch.cat((r4_t, r4['h4_t']), 0)
            r1_diff_mu_std = torch.cat((r1_diff_mu_std, r1['diff_mu_std_1']), 0)
            r2_diff_mu_std = torch.cat((r2_diff_mu_std, r2['diff_mu_std_2']), 0)
            r3_diff_mu_std = torch.cat((r3_diff_mu_std, r3['diff_mu_std_3']), 0)
            r4_diff_mu_std = torch.cat((r4_diff_mu_std, r4['diff_mu_std_4']), 0)
            r1_mu_std = torch.cat((r1_mu_std, r1['mu_std_1']), 0)
            r2_mu_std = torch.cat((r2_mu_std, r2['mu_std_2']), 0)
            r3_mu_std = torch.cat((r3_mu_std, r3['mu_std_3']), 0)
            r4_mu_std = torch.cat((r4_mu_std, r4['mu_std_4']), 0)
            var1_mu_std = torch.cat((var1_mu_std, r1['var_mu_std_1']), 0)
            var2_mu_std = torch.cat((var2_mu_std, r2['var_mu_std_2']), 0)
            var3_mu_std = torch.cat((var3_mu_std, r3['var_mu_std_3']), 0)
            var4_mu_std = torch.cat((var4_mu_std, r4['var_mu_std_4']), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size
        last_batch = video_data[frame_start:video_length].to(device)
        r1, r2, r3, r4  = extractor(last_batch)
        r1_t = torch.cat((r1_t, r1['h1_t']), 0)
        r2_t = torch.cat((r2_t, r2['h2_t']), 0)
        r3_t = torch.cat((r3_t, r3['h3_t']), 0)
        r4_t = torch.cat((r4_t, r4['h4_t']), 0)
        r1_diff_mu_std = torch.cat((r1_diff_mu_std, r1['diff_mu_std_1']), 0)
        r2_diff_mu_std = torch.cat((r2_diff_mu_std, r2['diff_mu_std_2']), 0)
        r3_diff_mu_std = torch.cat((r3_diff_mu_std, r3['diff_mu_std_3']), 0)
        r4_diff_mu_std = torch.cat((r4_diff_mu_std, r4['diff_mu_std_4']), 0)
        r1_mu_std = torch.cat((r1_mu_std, r1['mu_std_1']), 0)
        r2_mu_std = torch.cat((r2_mu_std, r2['mu_std_2']), 0)
        r3_mu_std = torch.cat((r3_mu_std, r3['mu_std_3']), 0)
        r4_mu_std = torch.cat((r4_mu_std, r4['mu_std_4']), 0)
        var1_mu_std = torch.cat((var1_mu_std, r1['var_mu_std_1']), 0)
        var2_mu_std = torch.cat((var2_mu_std, r2['var_mu_std_2']), 0)
        var3_mu_std = torch.cat((var3_mu_std, r3['var_mu_std_3']), 0)
        var4_mu_std = torch.cat((var4_mu_std, r4['var_mu_std_4']), 0)
        
        print('r1_t', r1_t[10].item())
        # t = torch.cat((r1_t, r2_t, r3_t, r4_t), 0) ## [450, 4]; a,b,c,d=torch.split(torch.cat((r1_t, r2_t, r3_t, r4_t), 0), [450,450,450,450], 0)
        t = r4_t.squeeze()

        # diff_mu_std = torch.cat((r1_diff_mu_std, r2_diff_mu_std, r3_diff_mu_std, r4_diff_mu_std), 1) #a,b,c,d=torch.split(diff_mu_std, [512,1024,2048,4096], 1)
        diff_mu_std = r4_diff_mu_std.squeeze()
        
        # mu_std = torch.cat((r1_mu_std, r2_mu_std, r3_mu_std, r4_mu_std), 1)
        mu_std = r4_mu_std.squeeze()

        # var_mu_std = torch.cat((var1_mu_std, var2_mu_std, var3_mu_std, var4_mu_std), 1)
        var_mu_std = var4_mu_std.squeeze()
        # result.append(res)
    
    return t, diff_mu_std, mu_std, var_mu_std


def main(dataset, features_folder):
    '''
    '''
    
    for i in range(len(dataset)):
        video_name, current_data = dataset[i]
        print('Video {}: video name {}: length {}'.format(i, video_name, len(current_data)))
        feature_path = os.path.join(features_folder, str(video_name)) + '.npy'
        if os.path.isfile(feature_path):
            continue

        feat_ti, feat_diff_mu_std, feat_mu_std, feat_var_mu_std  = get_features(current_data, frame_batch_size=1)
        feat_save = torch.cat((feat_diff_mu_std, feat_mu_std), 1)
        print(video_name, ': ', feat_diff_mu_std.shape, feat_mu_std.shape, feat_save.shape)
        np.save(feature_path, feat_save.to('cpu').numpy())
        # np.savez(feature_path, ti= feat_ti.to('cpu').numpy(), diff_mu_std=feat_diff_mu_std.to('cpu').numpy(), mu_std=feat_mu_std.to('cpu').numpy(), 
        # var_mu_std = feat_var_mu_std.to('cpu').numpy())

if __name__ == "__main__":
    # import fire
    # fire.Fire()
    #Path need to be set
    features_folder = r'./AVT/VSFA_resnet50_iqapretrain_ms'
    video_root = r'./AVT/segments_frames'
    csv_file = Path("./AVT/video_names.csv")
    video_info = pd.read_csv(csv_file, header=0)
    video_names_ = video_info.iloc[:, 1].tolist()
    video_names = [name.split('.')[0] for name in video_names_]
 

    if not os.path.exists(features_folder):
        os.makedirs(features_folder)

    mp.set_start_method('spawn')
    num_processes = 1
    processes = []
    nvideo_per_node = int(len(video_names) / num_processes)
    # nvideo_per_node = 1
    for rank in range(num_processes):
        video_names_ = video_names[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        dataset = VideoDataset(video_names_, video_root)
        p = mp.Process(target=main, args=(dataset, features_folder))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
