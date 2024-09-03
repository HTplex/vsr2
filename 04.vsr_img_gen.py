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
from multiprocessing import Pool
from os.path import exists

def job(data_id):
    # meta
    fps = 30
    frame_len = 4*fps
    mouth_h,mouth_w = 36,36
    face_h,face_w = 100,80
    data_root = "/data1/agent_h/data/datasets/zhiyun_mouth_clips_4s_30fps_08162024/"
    all_video_paths = glob("/data1/agent_h/data/datasets/zhiyun_part/**/*.mp4", recursive=True)
    face_save_path = join(data_root,data_id+"_face.npz")
    mouth_save_path = join(data_root,data_id+"_mouth.npz")
    auto_continue = True
    if auto_continue:
        if exists(face_save_path) and exists(mouth_save_path):
            # check npz is not interrupted
            try:
                np.load(face_save_path)
                np.load(mouth_save_path)
            except:
                print('data_id',data_id,'corrupted, reprocessing')
                pass
            else:
                return
    # init video path index
    indexed_video_paths = {}
    for video_path in all_video_paths:
        indexed_video_paths[basename(video_path)[:-4]]=video_path
    # load data
    landmarks_batch = np.load(join(data_root,data_id+".npz"))['arr_0']
    with open(join(data_root,data_id+".json"), 'r') as fp:
        text_batch = json.load(fp)
    
    # get face/mouth images for each sample, save as npz 100*575*100*80*3
    face_frames_batch = []
    mouth_frames_batch = []
    for idx,landmarks in tqdm(enumerate(landmarks_batch),total=len(landmarks_batch)):
        video_path = indexed_video_paths["_".join(text_batch[idx]['id'].split("_")[:-2])]
        begin_frame_no,end_frame_no = int(text_batch[idx]['id'].split("_")[-2])*fps,int(text_batch[idx]['id'].split("_")[-1])*fps
        frame_list = get_video_frames(video_path,begin_frame_no,end_frame_no)
        # frame sometimes is not full, landmarks is not full as well, are padded 0
        mouth_frames = np.zeros((frame_len,mouth_h,mouth_w,3),dtype=np.uint8)
        face_frames =  np.zeros((frame_len,face_h,face_w,3),dtype=np.uint8)
        if frame_list:
            h,w,c = frame_list[0].shape
            x1, y1, z1, x2, y2, z2 = get_3d_points_bbox(landmarks)
            mx1,my1,mx2,my2 = get_mouth_square(landmarks,h,w)

            for frame_idx,frame in enumerate(frame_list):
                # init frame
                face_frame = np.zeros((face_h,face_w,3),dtype=np.uint8)
                mouth_frame = np.zeros((mouth_h,mouth_w,3),dtype=np.uint8)     
                # face add if valid, add black if not        
                face_frame_raw = frame[int(y1[frame_idx]*h):int(y2[frame_idx]*h),
                                    int(x1[frame_idx]*w):int(x2[frame_idx]*w)]
                if (face_frame_raw.shape[0] > 20) and (face_frame_raw.shape[1] > 20):
                    face_frame = cv2.resize(face_frame_raw,(face_w,face_h))
                # mouth same thing
                mouth_frame_raw = frame[int(my1[frame_idx]):int(my2[frame_idx]),
                                    int(mx1[frame_idx]):int(mx2[frame_idx])]
                if mouth_frame_raw.shape[0] > 5 and mouth_frame_raw.shape[1] > 5:
                    mouth_frame = cv2.resize(mouth_frame_raw,(mouth_w,mouth_h))
                # add
                face_frames[frame_idx] = face_frame
                mouth_frames[frame_idx] = mouth_frame
        face_frames_batch.append(face_frames)
        mouth_frames_batch.append(mouth_frames)
    # print([frame.shape for frame in face_frames_batch])
    # print([frame.shape for frame in mouth_frames_batch])
    face_frames_batch = np.array(face_frames_batch)
    mouth_frames_batch = np.array(mouth_frames_batch)
    # print(face_frames_batch.shape,mouth_frames_batch.shape)

    np.savez_compressed(face_save_path,face_frames_batch)
    np.savez_compressed(mouth_save_path,mouth_frames_batch)


if __name__ == "__main__":
    p = Pool(64)
    data_root = "/data1/agent_h/data/datasets/zhiyun_mouth_clips_4s_30fps_08162024/"
    data_ids = sorted([basename(x[:-5]) for x in glob(join(data_root,"*.json"))])
    for _ in tqdm(p.imap_unordered(job,data_ids),total=len(data_ids)):
        pass
    # for data_id in data_ids:
    #     job(data_id)

