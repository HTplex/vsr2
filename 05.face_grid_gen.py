# data explore
from glob import glob
from os.path import join, basename
# sequence_len,478,3
from os.path import basename
import numpy as np
import json
from pprint import pprint
import cv2
from tqdm import tqdm
from data_utils import *


data_root = "/data1/agent_h/data/datasets/zhiyun_mouth_clips_4s_30fps_08162024/"
data_ids = sorted([basename(x[:-5]) for x in glob(join(data_root,"*.json"))])

for data_id in tqdm(data_ids):
    mouth_grid_save_path = join(data_root,data_id+"_grid_simplev0.npz")
    mouth_frames_batch = np.load(join(data_root,data_id+"_mouth.npz"))['arr_0']
    with open(join(data_root,data_id+".json"), 'r') as fp:
        text_batch = json.load(fp)
    mouth_grid_batch = []
    for video_idx in range(len(mouth_frames_batch)):
        mouth_frames = mouth_frames_batch[video_idx]
        text = text_batch[video_idx]
        mouth_grid = gen_mouth_grid(mouth_frames,
                                    output_config="336/28",
                                    method="simple")
        mouth_grid_batch.append(mouth_grid)
    mouth_grid_batch = np.array(mouth_grid_batch)
    np.savez_compressed(mouth_grid_save_path,mouth_grid_batch)
