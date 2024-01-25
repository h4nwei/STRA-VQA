import pandas as pd
import numpy as np
import openpyxl
import random
def generate_split(idx):
    train_ratio = 0.8
    # val_ratio = 0.2
    test_ratio = 0.2
    split_file = '/home/zhw/vqa/code/etri/code_baseline/feature/data/LIVEHFR/train_val_test_split_{}.xlsx'.format(idx)
    print(split_file)
    # seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # test_seed = random.sample(seed, 2)
    # train_seed = [seed.remove(s) for s in test_seed]
    database_info_file_path = '/home/zhw/vqa/code/etri/code_baseline/feature/data/LIVEHFR/LIVE_HFR.csv'
    df_info = pd.read_csv(database_info_file_path, header=0)
    video_names_ref = df_info.iloc[:, 0].tolist()
    video_names_dis = df_info.iloc[:, 1].tolist()
    content_idx = df_info.iloc[:, 7].tolist()
    content_idx_new = list(set(content_idx))
    np.random.shuffle(content_idx_new)
    
    

    dis_names = np.array([str(name)  for name in video_names_dis])
    ref_names = np.array([str(name)  for name in video_names_ref])
    num_videos = len(content_idx_new)

    num_train_videos = int(train_ratio * num_videos)
    num_test_videos = int(test_ratio * num_videos)
    
    status = [ 'train' if v in content_idx_new[:num_train_videos]  else 'test'  for k, v in enumerate(content_idx)]

    # num_test_videos = num_videos - num_train_videos
    # train_status = np.array(['train'] * num_train_videos)
    # status = np.array(['train'] * num_train_videos + ['test'] * num_test_videos)
    # np.random.shuffle(status)
    
    split_info = np.array([ref_names, dis_names, status]).T
    df_split_info = pd.DataFrame(split_info, columns=['ref_video_name', 'dis_video_name', 'status'])
    # print(df_split_info.head())
    df_split_info.to_excel(split_file)

def main():
    for idx in range(0, 10):
        generate_split(idx)
    
    
if __name__ == '__main__':
    main()