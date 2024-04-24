import itertools
from mediapipe_util import *
import cv2
import gzip
import pickle
import numpy as np
import mediapipe as mp

def get_whisper_text_windows(whisper_result, clip_len=15, window_len=3):
    text_windows = {} #  dict of {'0/15':'习近平总书记', '3/18':'总书记123', ...}
    # get words list
    all_words = [x['words'] for x in whisper_result['segments']]
    flattened_words = list(itertools.chain.from_iterable(all_words))
    # flatten
    """
   [{'end': 1.36,'probability': 0.801658421754837,'start': 0.8799999999999997,'word': '习'},
    {'end': 1.5, 'probability': 0.9996213912963867, 'start': 1.36, 'word': '近'},
    {'end': 1.7, 'probability': 0.9999738931655884, 'start': 1.5, 'word': '平'},
    {'end': 1.92, 'probability': 0.9996397495269775, 'start': 1.7, 'word': '总'},
    {'end': 2.16, 'probability': 0.9998128414154053, 'start': 1.92, 'word': '书'},
    {'end': 2.36, 'probability': 0.9997205138206482, 'start': 2.16, 'word': '记'},
    ...
   ]
    """
    # pprint(flattened_words)
    
    # print(len(flattened_words))
    # fill text windows
    start_time = 0
    end_time = clip_len
    while start_time < max([x['end'] for x in flattened_words]):
        text = ''
        for word in flattened_words:
            if word['start'] >= start_time and word['end'] <= end_time:
                text += word['word']
        
        text_windows[f"{str(start_time).zfill(2)}/{end_time}"] = text
        start_time += window_len
        end_time += window_len
    return text_windows




def draw_landmarks_on_video(mediapipe_result, output_path, height=1080, width=1920):
        # Get the dimensions of the first frame

        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

        # Iterate through each frame in the mediapipe_result
        for mediapipe_frame in mediapipe_result:
            # Draw the face landmarks on the frame
            canvas = np.zeros((height,width,3),dtype=np.uint8)
            canvas = draw_landmarks_on_image(canvas, mediapipe_frame)
            show_img_np(canvas)
            # Write the frame to the video file
            out.write(canvas)

        # Release the VideoWriter object
        out.release()

def get_mediapipe_windows(mediapipe_path, clip_len=15, window_len=3, fps=25, straight = False, debug=False):
    with gzip.open(mediapipe_path, 'r') as fp:
        # load pickle
        mediapipe_result = pickle.load(fp)
    # print(len(mediapipe_result), "frames") 
    # print(len(mediapipe_result[0].face_landmarks[0]), "landmarks per frame")
    # print(mediapipe_result[0].face_landmarks[0][0])
    # demo = True
    if debug:
        canvas = np.zeros((1080,1920,3),dtype=np.uint8)
        canvas = draw_landmarks_on_image(canvas, mediapipe_result[0])
        show_img_np(canvas)
        draw_landmarks_on_video(mediapipe_result, './data/output.mp4')
    
    # save raw landmark clips
    start_time, end_time = 0, clip_len
    face_windows = {}
    while start_time*fps < len(mediapipe_result):
        mediapipe_result_clip = mediapipe_result
        face_sequence = []
        for i,mediapipe_frame in enumerate(mediapipe_result_clip):
            if i >= start_time*fps and i < end_time*fps:
                if len(mediapipe_frame.face_landmarks) > 0:
                    face_landmarks = mediapipe_frame.face_landmarks[0]
                    face_sequence.append(face_landmarks)
                else:
                    # if no face landmarks, use last frame
                    if face_sequence:
                        face_sequence.append(face_sequence[-1])
                    else:
                        face_sequence.append([])
        face_windows[f"{str(start_time).zfill(2)}/{end_time}"] = face_sequence
        start_time+=window_len
        end_time+=window_len
    return face_windows

def numpyify_face_windows(face_windows, sequence_len = 15*25):
    # convert face windows to numpy array, add 0 for no face landmarks and padding
    numpy_window = np.zeros((sequence_len,478,3))
    for i,face_window in enumerate(face_windows):
        for j,face_landmark in enumerate(face_window):
            if face_landmark:
                numpy_window[i][j] = [face_landmark.x, face_landmark.y, face_landmark.z]
    return numpy_window

    
    
def normalize_face_window(numpy_window):
    numpy_window_normalized = numpy_window.copy()
    numpy_window_normalized[:,:,0] = (numpy_window[:,:,0] - np.min(numpy_window[:,:,0])) / (np.max(numpy_window[:,:,0]) - np.min(numpy_window[:,:,0]))
    numpy_window_normalized[:,:,1] = (numpy_window[:,:,1] - np.min(numpy_window[:,:,1])) / (np.max(numpy_window[:,:,1]) - np.min(numpy_window[:,:,1]))
    numpy_window_normalized[:,:,2] = (numpy_window[:,:,2] - np.min(numpy_window[:,:,2])) / (np.max(numpy_window[:,:,2]) - np.min(numpy_window[:,:,2]))
    return numpy_window_normalized