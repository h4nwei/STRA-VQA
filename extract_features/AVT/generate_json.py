import pandas as pd
import os
from PIL import Image
import json
from pathlib import Path

def main():
    # database_name = '/home/zhw/vqa/code/3D-DISITS/feature/data/CSIQ/CSIQ.json'

    split_file_path = r'/home/zhw/vqa/code/etri/code_baseline/data/'
    split_idx_file_path_list = ["/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/splits/train_val_test_split_1.xlsx",
                                "/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/splits/train_val_test_split_2.xlsx",
                                "/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/splits/train_val_test_split_3.xlsx",
                                "/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/splits/train_val_test_split_4.xlsx",
                                "/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/splits/train_val_test_split_5.xlsx",                        
    ]
    database_name = 'AVT'

    csv_file = Path('/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/video_names.csv')
    database_info = pd.read_csv('/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/metadata.csv')
    video_info = pd.read_csv(csv_file)

    video_info = pd.read_csv(csv_file, header=0)
    video_names_ref = video_info.iloc[:, 0].tolist()
    video_names_dis = video_info.iloc[:, 1].tolist()
    video_moses = video_info.iloc[:, 2].tolist()
    video_stds = video_info.iloc[:, 3].tolist()

    # contentIndex = database_info.iloc[:, 1].tolist()
    # video_source_resolutions = database_info.iloc[:, 3].tolist()
    video_source_process_resolutions = database_info.iloc[:, 3].tolist()
    # video_source_framerate = database_info.iloc[:, 3].tolist()
    video_processedFrameRateRatio = database_info.iloc[:, 5].tolist()
    video_bitrate = database_info.iloc[:, 6].tolist()
    video_info = {'video_name_ref': video_names_ref,
    'video_name_dis': video_names_dis,
    'video_source_process_resolutions': video_source_process_resolutions,
    'video_mos': video_moses,
    'video_stds': video_stds,
    'video_processedFrameRateRatio': video_processedFrameRateRatio,
    'video_birate':video_bitrate
    }
    for csv_file in split_idx_file_path_list:
        # train_val_test_split_0.xlsx
        

        filename = split_file_path + database_name + '_' + csv_file.split('/')[-1].split('.')[0] + '.json'
        
        print(filename)
        # video_info = pd.read_csv(csv_file)

        # video_info = pd.read_csv(csv_file, header=0)
        video_info = pd.read_excel(csv_file)
        idx_all = video_info.iloc[:, 0].values

        video_names = video_info.iloc[:, 1].tolist()
        # video_category = video_info.iloc[:, 2].tolist()
        video_mos = video_info.iloc[:, 2].tolist()

        split_status = video_info['status'].values
        train_idx = idx_all[split_status == 'train']
        # validation_idx = idx_all[split_status == 'validation']
        test_idx = idx_all[split_status == 'test']
        data = {'video_name_ref': video_names_ref,
    'video_name_dis': video_names_dis,
    'video_source_process_resolutions': video_source_process_resolutions,
    'video_mos': video_moses,
    'video_stds': video_stds,
    'video_processedFrameRateRatio': video_processedFrameRateRatio,
    'video_birate':video_bitrate,
            'video_name_list': video_names,
        'video_label_list': video_mos,
        'train_idx': train_idx.tolist(),
        # 'val_idx': validation_idx.tolist(),
        'test_idx': test_idx.tolist()
        }
        
       
        with open(filename, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()