import pandas as pd
import os
from PIL import Image
import json
from pathlib import Path

def main():
    database_name = '/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/ETRI-LIVE_fr.json'
    csv_file = Path('/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/ETRI-LIVE.csv')
    database_info = pd.read_csv('/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/ETRI-LIVE_videoInfo.csv')
    video_info = pd.read_csv(csv_file)

    video_info = pd.read_csv(csv_file, header=0)
    video_names_ref = video_info.iloc[:, 0].tolist()
    video_names_dis = video_info.iloc[:, 1].tolist()
    video_moses = video_info.iloc[:, 2].tolist()
    video_stds = video_info.iloc[:, 3].tolist()
    contentIndex = database_info.iloc[:, 1].tolist()
    video_source_resolutions = database_info.iloc[:, 2].tolist()
    video_source_process_resolutions = database_info.iloc[:, 4].tolist()
    video_source_framerate = database_info.iloc[:, 3].tolist()
    video_processedFrameRateRatio = database_info.iloc[:, 5].tolist()
    video_bitrate = database_info.iloc[:, 6].tolist()
    video_info = {'video_name_ref': video_names_ref,
    'video_name_dis': video_names_dis,
    'video_source_resolutions': video_source_resolutions,
    'contentIndex': contentIndex,
    'video_source_process_resolutions': video_source_process_resolutions,
    'video_mos': video_moses,
    'video_stds': video_stds,
    'video_source_framerate': video_source_framerate,
    'video_processedFrameRateRatio': video_processedFrameRateRatio,
    'video_birate':video_bitrate
    }
    

    # height = 2160
    # width = 3840

    # database_info = {}
    
    # for i in range(len(video_names)):
    #     video_name = str(video_names[i])
    #     label = video_moses[i]
 
    #     # print(video_name)
    #     # print(height, width)
    #     video_info = {'video_name': video_name, 'format': format,
    #                   'distorted_type': distorted_type,
    #                   'height': height,
    #                   'width': width,
    #                   'label': label}
    #     database_info[i] = video_info
    print(video_info)
    with open(database_name, 'w') as f:
        json.dump(video_info, f)
        
        

if __name__ == '__main__':
    main()