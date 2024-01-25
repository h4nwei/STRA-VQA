from PIL import Image
import os
from multiprocessing import Process
from pathlib import Path
import subprocess
# Specify the directory you want to use
def extract_frame(video_names, dir_path, save_path):
    for  foldername in video_names:
        # print("Exisiting {} video".format(foldername))
        print("Processing {} video".format(foldername))
        video_path = os.path.join(dir_path, foldername)
        save_frame_path = os.path.join(save_path + foldername)
        if not os.path.exists(save_frame_path):
            # print("Processing {} video".format(foldername))
            os.makedirs(save_frame_path)

        for i, filename in enumerate(os.listdir(video_path)):
            if filename.endswith('.png'):
                img = Image.open(os.path.join(video_path, filename))
                rgb_img = img.convert('RGB')
                img_name = 'output-{:04d}.jpg'.format(i+1)
                rgb_img.save(os.path.join(save_frame_path, img_name), 'JPEG')

if __name__ == '__main__':
    dir_path = '/data/zhw/vqa/AVT/src_videos_frames'
    save_path = '/data/zhw/vqa/AVT/segments_frames/'

    num_processes = 5
    processes = []
    video_names = os.listdir(dir_path)
    nvideo_per_node = int(len(video_names) / num_processes)
    for rank in range(num_processes):
        names_ = video_names[rank*nvideo_per_node:(rank+1)*nvideo_per_node]
        p = Process(target=extract_frame, args=(names_, dir_path, save_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()