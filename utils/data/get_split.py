import json
import numpy as np


def tmp_split():
    import pandas as pd
    import os
    split_file_path = r'/home/zhw/vqa/code/etri/code_baseline/data/E-LIVE/'
    split_idx_file_path_list = ["/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/FR_splits/train_val_test_split_1.xlsx",
    "/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/FR_splits/train_val_test_split_2.xlsx",
    "/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/FR_splits/train_val_test_split_3.xlsx",
    "/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/FR_splits/train_val_test_split_4.xlsx",
    "/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/FR_splits/train_val_test_split_5.xlsx",
    # "/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/train_val_test_split_6.xlsx",
    # "/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/train_val_test_split_7.xlsx",
    # "/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/train_val_test_split_8.xlsx",
    # "/home/zhw/vqa/code/vqatransformer/feature/data/E-LIVE/train_val_test_split_9.xlsx",
    ]
    database_info = r'/home/zhw/vqa/code/etri/code_baseline/feature/data/E-LIVE/ETRI-LIVE_fr.json'
    database_name = 'ETRI-LIVE'
    for split_idx_file_path in split_idx_file_path_list:
        # train_val_test_split_0.xlsx
        filename = split_file_path + database_name + '_' + split_idx_file_path.split('/')[-1].split('.')[0] + '.json'
        
        print(filename)
        # split_idx_file_path = os.path.join(split_file_path, split_idx_file_path)
        split_info = pd.read_excel(split_idx_file_path)
        idx_all = split_info.iloc[:, 0].values
        split_status = split_info['status'].values
        train_idx = idx_all[split_status == 'train']
        validation_idx = idx_all[split_status == 'validation']
        test_idx = idx_all[split_status == 'test']
        
        with open(database_info, 'r') as f:
            data = json.load(f)
            data['train_idx'] = train_idx.tolist()
            data['val_idx'] = validation_idx.tolist()
            data['test_idx'] = test_idx.tolist()
        with open(filename, 'w') as f:
            json.dump(data, f)
        

def get_split(**kwargs):
    database_info = kwargs['database_info']
    database_name = kwargs['database_name']
    with open(database_info, 'r') as f:
        data = json.load(f)
    num_videos = data['num_videos']
    print(data)
    print(num_videos)
    
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    num_train_videos = int(train_ratio * num_videos)
    num_val_videos = int(val_ratio * num_videos)
    num_test_videos = num_videos - num_train_videos - num_val_videos
    status = np.array(['train'] * num_train_videos + ['val'] * num_val_videos + ['test'] * num_test_videos)
    np.random.shuffle(status)
    train_idx = np.argwhere(status=='train').squeeze()
    val_idx = np.argwhere(status=='val').squeeze()
    test_idx = np.argwhere(status=='test').squeeze()
    
    data['train_idx'] = train_idx.tolist()
    data['val_idx'] = val_idx.tolist()
    data['test_idx'] = test_idx.tolist()
    
    return data
    

def main():
    database_info = r'C:\freetime\code\BasicQA\data\KoNViD\KoNViD.json'
    database_name = 'KoNViD'
    info = {'database_info': database_info,
            'database_name': database_name}
    data = get_split(**info)
    print(data['train_idx'])
    print(data['val_idx'])
    print(data['test_idx'])
   
if __name__ == '__main__':
    # main()
    tmp_split()