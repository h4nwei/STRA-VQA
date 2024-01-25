import pandas as pd
import os
from PIL import Image
import json
from pathlib import Path

def main():
    database_name = '/home/zhw/vqa/code/etri/code_baseline/feature/data/MCL/AVT_fr.json'
    csv_file = Path('/home/zhw/vqa/code/etri/code_baseline/feature/data/MCL/MCL.csv')
    # database_info = pd.read_csv('/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/metadata.csv')


    video_info = pd.read_csv(csv_file, header=0)
    video_names_ref = video_info.iloc[:, 0].tolist()
    video_names_dis = video_info.iloc[:, 1].tolist()
    video_moses = video_info.iloc[:, 2].tolist()
    video_stds = video_info.iloc[:, 3].tolist()


    video_source_process_resolutions = video_info.iloc[:, 5].tolist()
    video_processedFrameRateRatio = video_info.iloc[:, 6].tolist()
    video_bitrate = video_info.iloc[:, 6].tolist()
    video_info = {'video_name_ref': video_names_ref,
    'video_name_dis': video_names_dis,
    'video_source_process_resolutions': video_source_process_resolutions,
    'video_mos': video_moses,
    'video_stds': video_stds,
    'video_processedFrameRateRatio': video_processedFrameRateRatio,
    'video_birate':video_bitrate
    }
    

    print(video_info)
    with open(database_name, 'w') as f:
        json.dump(video_info, f)
        
        

if __name__ == '__main__':
    main()