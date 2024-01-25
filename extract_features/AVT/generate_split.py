import pandas as pd
import numpy as np
import openpyxl
import random
def generate_split(idx):
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    split_file = '/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/train_val_test_split_{}.xlsx'.format(idx)
    print(split_file)
    seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    test_seed = random.sample(seed, 2)
    train_seed = [seed.remove(s) for s in test_seed]
    database_info_file_path = '/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/ETRI-LIVE_MOS.csv'
    df_info = pd.read_csv(database_info_file_path, header=0)
    video_names = df_info.iloc[:, 0].tolist()
    video_moses = df_info.iloc[:, 1].tolist()
    

    file_names = np.array([str(name) + '.yuv' for name in video_names])
    num_videos = len(file_names)
    num_train_videos = int(train_ratio * num_videos)
    num_val_videos = int(val_ratio * num_videos)
    num_test_videos = num_videos - num_train_videos - num_val_videos
    status = np.array(['train'] * num_train_videos + ['validation'] * num_val_videos + ['test'] * num_test_videos)
    np.random.shuffle(status)
    
    split_info = np.array([file_names, status]).T
    df_split_info = pd.DataFrame(split_info, columns=['video_name', 'status'])
    # print(df_split_info.head())
    df_split_info.to_excel(split_file)

def main():
    for idx in range(1, 10):
        generate_split(idx)
    
    
if __name__ == '__main__':
    main()