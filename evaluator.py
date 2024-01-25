from pathlib import Path
import torch

from utility import AverageMeter, IQAPerformance


class Evaluator:
    def __init__(self, args, model, dataset_name, loader):
        self.args = args
        self.log_dir = Path(args.log_dir)
        self.loader_test = loader
        self.model = torch.nn.DataParallel(model).cuda()
        self.max_len = args.max_len
        self.dataset_name = dataset_name
 
    def predict(self):
        self.model.eval()
        pred_list = []
        mos_list = []
        perf = IQAPerformance(self.log_dir)
        with torch.no_grad():
            for bi, (x, info, length, y, scale, dis_name,ref_name) in enumerate(self.loader_test, start=1):
                # dis_name = dis_nameref_name
                #ref_name[0].split('_')[2]=='60hz' and dis_name[0].split('_')[2]!='30hz':
                # if ref_name[0].split('_')[2]=='120hz' and dis_name[0].split('_')[2]=='60hz' or ref_name[0].split('_')[2]=='60hz' and dis_name[0].split('_')[2]=='30hz':
                #     print(dis_name)
                #     continue
                ref, dis = x
                y = y.cuda(non_blocking=True)
                ref = ref.cuda(non_blocking=True)
                dis = dis.cuda(non_blocking=True)
                videos = [ref, dis]
                resolution, framerate, birate = info
                resolution = resolution.to(torch.float32).cuda(non_blocking=True)
                framerate = framerate.to(torch.float32).cuda(non_blocking=True)
                birate = birate.to(torch.float32).cuda(non_blocking=True)
                video_info = [resolution, framerate, birate]
                length = length.cuda(non_blocking=True)
                scale = scale.cuda(non_blocking=True)
                y_pred = self.model(videos, video_info, length)
                y_pred = y_pred.squeeze() * scale
                y = y.squeeze() * scale   

                perf.update(y_pred, y)
                pred_list.extend([p.item() for p in y_pred])
                mos_list.extend([s.item() for s in y])
        corr = perf.compute(is_plot=True, fig_name=f'validation_{self.dataset_name}.png')
        name = self.dataset_name+'_Testing.csv'
        with open(self.log_dir / name, 'w') as f:
            for mos, pred in zip(mos_list, pred_list):
                f.write(f"{float(mos):.3f}, {float(pred):.3f}\n")
        print(f"{self.dataset_name} Testing Result:\nSRCC {corr['srcc']:.4f} | KRCC {corr['krcc']:.4f} | PLCC {corr['plcc']:.4f} | RMSE {corr['rmse']:.4f}")
        return corr
    
        # corr = perf.compute(is_plot=True, fig_name=f'validation.png')
        # with open(self.log_dir / 'Testing.csv', 'w') as f:
        #     for mos, pred in zip(mos_list, pred_list):
        #         f.write(f"{float(mos):.3f}, {float(pred):.3f}\n")
        # print(f"Testing Result:\nSRCC {corr['srcc']:.4f} | KRCC {corr['krcc']:.4f} | PLCC {corr['plcc']:.4f} | RMSE {corr['rmse']:.4f}")
        # return corr

    def prepare(self, x, y):
        if type(x) in [list, tuple]:
            x = [x_.cuda(non_blocking=True) for x_ in x]
        else:
            x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        return [x, y]


class Evaluator_E2E:
    def __init__(self, args, model, loader):
        self.args = args
        self.log_dir = Path(args.log_dir)
        self.loader_test = loader
        self.model = torch.nn.DataParallel(model).cuda()
        self.max_len = args.max_len
 
    def predict(self):
        self.model.eval()
        pred_list = []
        mos_list = []
        perf = IQAPerformance(self.log_dir)
        with torch.no_grad():
            for bi, (x, info, length, y, scale, dis_name,ref_name) in enumerate(self.loader_test, start=1):
                ref, dis = x
                b, n, m, c, h, w = ref.shape
                y = y.to(torch.float32).cuda(non_blocking=True)
                ref = ref.to(torch.float32).cuda(non_blocking=True)
                dis = dis.to(torch.float32).cuda(non_blocking=True)
                # videos = [ref, dis]
                resolution, framerate, birate = info
                resolution = resolution.to(torch.float32).cuda(non_blocking=True)
                framerate = framerate.to(torch.float32).cuda(non_blocking=True)
                birate = birate.to(torch.float32).cuda(non_blocking=True)
                video_info = [resolution, framerate, birate]
                length = length.cuda(non_blocking=True)
                scale = scale.cuda(non_blocking=True)
                clip_scores = []
                for i in range(n):
                    ref_clip = ref[:, i, :, :, :, :]
                    dis_clip = dis[:, i, :, :, :, :]
                    # clips = [ref_clip, dis_clip]
                    # y_clip_pred = self.model(clips, video_info, length)
                    # clips = [ref_clip, dis_clip]
                    y_clip_pred = self.model(ref_clip, dis_clip)
                    clip_scores.append(y_clip_pred)
               
                # y_pred = y_pred.squeeze()
                y_pred = torch.stack(clip_scores, 0)
                y_pred = y_pred.mean(0)
                
                y_pred = y_pred.squeeze() * scale
                y = y.squeeze() * scale   

                perf.update(y_pred, y)
                pred_list.extend([p.item() for p in y_pred])
                mos_list.extend([s.item() for s in y])
        
        corr = perf.compute(is_plot=True, fig_name=f'validation.png')
        with open(self.log_dir / 'Testing.csv', 'w') as f:
            for mos, pred in zip(mos_list, pred_list):
                f.write(f"{float(mos):.3f}, {float(pred):.3f}\n")
        print(f"Testing Result:\nSRCC {corr['srcc']:.4f} | KRCC {corr['krcc']:.4f} | PLCC {corr['plcc']:.4f} | RMSE {corr['rmse']:.4f}")
        return corr

    def prepare(self, x, y):
        if type(x) in [list, tuple]:
            x = [x_.cuda(non_blocking=True) for x_ in x]
        else:
            x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        return [x, y]