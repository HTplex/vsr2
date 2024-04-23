import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from glob import glob
import os
from os.path import join, basename
import json
import pickle  
import cv2
from mediapipe_util import *
import gzip


model_path = './data/face_landmarker.task'


all_video_paths = glob("/data/agent_h/datasets_chunyu/clip_videos_v3/**/*.mp4", recursive=True)
print(len(all_video_paths))


def video_face_landmark_worker(video_path):
    output_root = "/data/agent_h/vsr_landmark_result_v3/"
    os.makedirs(output_root,exist_ok=True)
    save_path = join(output_root,basename(video_path)[:-4]+".FaceLandmarkerResultListzip")
    if os.path.exists(save_path):
        return
    
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
                                           running_mode=VisionRunningMode.VIDEO
                                          )
    detector = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    face_landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        detection_result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))
        face_landmarks_list.append(detection_result)

    with gzip.open(save_path, 'wb') as fp:
        pickle.dump(face_landmarks_list, fp)
        
    # else:
    #     with open(save_path, 'wb') as fp:
    #         pickle.dump(face_landmarks_list,fp)  
    fp.close()


# sp
# for video_path in tqdm(all_video_paths[:100]):
#     video_face_landmark_worker(video_path)


# mp

from multiprocessing import Pool
pool = Pool(128)
for _ in tqdm(pool.imap_unordered(video_face_landmark_worker, all_video_paths[:]),
              total=len(all_video_paths[:])):
    pass

        
# debug

# annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
# show_img_np(annotated_image)