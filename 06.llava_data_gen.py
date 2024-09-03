# data explore
from glob import glob
from os.path import join, basename
# sequence_len,478,3
from os.path import basename, dirname
import numpy as np
import json
from pprint import pprint
import cv2
from tqdm import tqdm
from data_utils import *
import os


data_root = "/data1/agent_h/data/datasets/zhiyun_mouth_clips_4s_30fps_08162024/"
export_root = "/data1/agent_h/data/datasets/zhiyun_mouth_clips_4s_30fps_08162024_llava_direct/"
"""
llava_direct: directly use mouth grid as input, ask one prompt, answer is vsr result.
"""

data_ids = sorted([basename(x[:-5]) for x in glob(join(data_root,"*.json"))])
img_save_folder = join(export_root,"images")
os.makedirs(img_save_folder,exist_ok = True)
label_save_path = join(export_root,"label.json")

labels = []


for data_id in tqdm(data_ids):
    mouth_grid_save_path = join(data_root,data_id+"_grid_simplev0.npz")
    mouth_grid_batch = np.load(mouth_grid_save_path)['arr_0']
    with open(join(data_root,data_id+".json"), 'r') as fp:
        text_batch = json.load(fp)
    for idx,text_data in enumerate(text_batch):
        mouth_grid = mouth_grid_batch[idx]
        label = {
            "id": text_data['id'],
            "image": "{}_{}.png".format(text_data['id'],idx),
            
            "conversations": [
                {
                    "from": "human",
                    "value": "对此图片进行唇语识别\n<image>"
                },
                {
                    "from": "gpt",
                    "value": text_data['text']
                }
            ]
        }
        labels.append(label)
        img_save_path = join(img_save_folder,label["image"])
        cv2.imwrite(img_save_path,mouth_grid)
with open(label_save_path, 'w') as fp:
    json.dump(labels, fp, sort_keys=True, indent=4, ensure_ascii=False)
