import imp
import os
import skvideo.io
from PIL import Image
from tqdm import tqdm
import pandas as pd

def get_frames():
    # info_path = 'VSFA_LIVE-Qualcomminfo.mat'
    video_folder = '/data/zhw/vqa/LIVE-HFR/videos/'
    mp4_folder = '/data/zhw/vqa/LIVE-HFR/videos_120fps'
    frame_folder = '/data/zhw/vqa/LIVE-HFR/frames'
    csv_file = "/home/zhw/vqa/code/etri/code_baseline/feature/data/LIVEHFR/LIVE_HFR.csv"
    video_info = pd.read_csv(csv_file, header=0)
    video_names = video_info.iloc[:, 1].tolist()
    # Info = h5py.File(info_path, 'r')
    # video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
    #                range(len(Info['video_names'][0, :]))]
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    # width = int(Info['width'][0])
    # height = int(Info['height'][0])

    for folder_name in tqdm(os.listdir(video_folder)):
        # video_name: 2999049224.mp4
        # save_folder: D:\datasets\VQAFrames\LIVEQualcomm\0723_ManUnderTree_GS5_03_20150723_130016\
        folder_path = os.path.join(video_folder, folder_name)
        for video_name in os.listdir(folder_path):
            save_folder = os.path.join(frame_folder, video_name.split('.')[0])
            print("video_name {}".format(video_name))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            video_path = os.path.join(folder_path, video_name)
            mp4_path = os.path.join(mp4_folder, video_name)
            #change the framerate to 120
            os.system('ffmpeg -i {} -r 120 {}'.format(video_path, mp4_path))
            #extract frames
            os.system('ffmpeg -i {} -q:v 1 {}/output-%04d.jpg'.format(mp4_path, save_folder))
        # video_length = video_data.shape[0]

        # for frame_idx in range(video_length):
        #     frame = video_data[frame_idx]
        #     img = Image.fromarray(frame)
        #     save_name = os.path.join(save_folder, "{0:04d}".format(frame_idx)) + '.png'
        #     # print(save_name)
        #     img.save(save_name)


def main():
    get_frames()


if __name__ == '__main__':
    main()