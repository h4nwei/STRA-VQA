from email import contentmanager
import os
import h5py
import skvideo.io
from PIL import Image
import gc
import pandas as pd

def get_frames():
    # info_path = 'VSFA_LIVE-Qualcomminfo.mat'
    # video_folder = '/media/kedeleshier/f1f792db-19f4-429e-8701-a57f0a6f4187/LZQ/VQA_datasets/LIVE-Qualcomm/Video'
    # frame_folder = '/media/kedeleshier/f1f792db-19f4-429e-8701-a57f0a6f4187/LZQ/VQA_frames/LIVEQualcomm'
    # Info = h5py.File(info_path, 'r')
    # video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
    #                range(len(Info['video_names'][0, :]))]
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    video_folder = '/data/zhw/vqa/MCL-V/reference_YUV_sequences/'
    frame_folder = '/data/zhw/vqa/MCL-V/frames/'
    csv_file = "/home/zhw/vqa/code/etri/code_baseline/feature/data/MCL/MCL.csv"
    video_info = pd.read_csv(csv_file, header=0)
    video_names = video_info.iloc[:, 0].tolist()
    width = video_info.iloc[:, 4].tolist()
    height = video_info.iloc[:, 5].tolist()
    format = 'YUV420'
    # width = 1920
    # height = 1080
    video_names = set(video_names)
    # pbar= tqdm(video_names)
    num_videos = len(video_names)
    print(video_names)
    for idx, video_name in enumerate(video_names):
        # pbar.set_description("Processing [%s] " % video_name)
        print(video_name)
        print(f"Processing [{idx+1}/{num_videos} {video_name}] [height:{height[idx]}--width:{width[idx]}]")
        # video_name: Artifacts/0723_ManUnderTree_GS5_03_20150723_130016.yuv
        # save_folder: D:\datasets\VQAFrames\LIVEQualcomm\0723_ManUnderTree_GS5_03_20150723_130016\
        save_folder = os.path.join(frame_folder, video_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        else:
            continue
        # video_data (ndarray): shape[lenght, H, W, C]
        if format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(video_folder, video_name.strip()+'.yuv'), height[idx], width[idx], inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(video_folder, video_name.strip()+'.mp4'))
        # video_data = skvideo.io.vread(os.path.join(video_folder, video_name.strip()+'.yuv'), height, width, inputdict={'-pix_fmt':'yuvj420p'})
        video_length = video_data.shape[0]
        
        for frame_idx in range(video_length):
            img_name = 'output-{:04d}.jpg'.format(frame_idx+1)
            save_name = os.path.join(save_folder, img_name) 
            if os.path.isfile(save_name):
                continue
            frame = video_data[frame_idx]
            img = Image.fromarray(frame)
            # print(save_name)
            img.save(save_name)
        del video_data
        gc.collect()

def main():
    get_frames()

if __name__ == '__main__':
    main()