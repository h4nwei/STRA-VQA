import imp
import os
import skvideo.io
from PIL import Image
from tqdm import tqdm
import pandas as pd

def get_frames():
    # info_path = 'VSFA_LIVE-Qualcomminfo.mat'
    video_folder = '/data/zhw/vqa/AVT/segments_ref_frate/'
    frame_folder = '/data/zhw/vqa/AVT/segments_frames'
    csv_file = "/home/zhw/vqa/code/etri/code_baseline/feature/data/AVT/video_names.csv"
    video_info = pd.read_csv(csv_file, header=0)
    video_names = video_info.iloc[:, 1].tolist()
    # Info = h5py.File(info_path, 'r')
    # video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
    #                range(len(Info['video_names'][0, :]))]
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    # width = int(Info['width'][0])
    # height = int(Info['height'][0])

    for video_name in tqdm(video_names):
        # video_name: 2999049224.mp4
        # save_folder: D:\datasets\VQAFrames\LIVEQualcomm\0723_ManUnderTree_GS5_03_20150723_130016\
        save_folder = os.path.join(frame_folder, video_name.split('.')[0])
        print("video_name {}".format(video_name))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # video_data (ndarray): shape[lenght, H, W, C] "{0:04d}".format(frame_idx)) + '.png
        # video_data = skvideo.io.vread(os.path.join(video_folder, video_name))
        os.system('ffmpeg -i {} {}/output-%04d.jpg'.format(os.path.join(video_folder, video_name), save_folder))
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