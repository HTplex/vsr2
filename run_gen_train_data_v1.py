from glob import glob
from os.path import join


video_root = "/data/agent_h/datasets_chunyu/clip_videos_v3/"
video_paths = sorted(glob(join(video_root,"**/*.mp4"),recursive=True))
whisper_root = "/data/agent_h/datasets_chunyu/whisper_result_v3/"
whisper_paths = sorted(glob(join(whisper_root,"*")))
mediapipe_root = "/data/agent_h/datasets_chunyu/vsr_landmark_result_v3/"
mediapipe_paths = sorted(glob(join(mediapipe_root,"*")))

# 1. form whiper and face into frames

print(video_paths[0])
print(whisper_paths[0])
print(mediapipe_paths[0])


# clean whisper
from pprint import pprint
import json
from datagen_ops import *
import gzip
import pickle
import numpy as np
import mediapipe as mp
from mediapipe_util import *
import cv2
from os.path import basename
import os
import uuid
from tqdm import tqdm

text_save_batch = []
face_save_batch = []
save_batch_counter = 0
total_batch_counter = 0

for video_idx in tqdm(range(len(video_paths))):
    whisper_json_path = whisper_paths[video_idx]
    with open(whisper_json_path, 'r') as fp:
        whisper_result = json.load(fp)


    save_path = "/data/agent_h/face_text_windows_v3/"
    os.makedirs(save_path, exist_ok=True)

    text_windows = get_whisper_text_windows(whisper_result, clip_len=15, window_len=3)
    # pprint(text_windows)

        
    face_windows = get_mediapipe_windows(mediapipe_paths[video_idx], clip_len=15, window_len=3)
    # numpy_window_normalized = normalize_face_window(numpy_window[3:,:,:])
    for key,text in text_windows.items():
        if key not in face_windows:
            continue
        numpy_window = numpyify_face_windows(face_windows[key],sequence_len=15*25)
        base_name = basename(whisper_json_path)[:-12]+"{}_{}".format(key.split("/")[0],key.split("/")[1])
        text_save_batch.append({'id':base_name, 'text':text})
        face_save_batch.append(numpy_window)
        # print(base_name)
        save_batch_counter += 1
        if save_batch_counter % 100 == 0:

            save_id = uuid.uuid4()
            with open(join(save_path,"{}.json".format(save_id)), 'w') as fp:
                json.dump(text_save_batch, fp, sort_keys=True, indent=4, ensure_ascii=False)
            np.savez_compressed(join(save_path,"{}.npz".format(save_id)),np.array(face_save_batch,dtype=np.float32))
            text_save_batch = []
            face_save_batch = []
            total_batch_counter += 1
    