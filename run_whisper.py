import torch
from time import sleep
import whisper
from glob import glob
from time import time
import argparse
import os
import json


print('CUDA enabled:', torch.cuda.is_available())
sleep(5)

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--partition", help="partition 1/2")
args = parser.parse_args()


# Rest of the code

if __name__ == "__main__":
    save_root = "/data/agent_h/whisper_result_v3/"
    os.makedirs(save_root,exist_ok=True)
    part_x,part_y = int(args.partition.split("/")[0]),int(args.partition.split("/")[1])
    all_video_paths = glob("/data/agent_h/datasets_chunyu/clip_videos_v3/**/*.mp4", recursive=True)
    print("total n videos", len(all_video_paths))
    # parition
    part_video_paths = []
    for i,video_path in enumerate(all_video_paths):
        if i%part_y==part_x:
            part_video_paths.append(video_path)
    all_video_paths = part_video_paths
    
    model = whisper.load_model("large-v3")

    for video_path in all_video_paths:
        print(video_path)
        t = time()
        result = model.transcribe(video_path,
                                word_timestamps=True,
                                language="zh")

        print("time: ", time()-t)
        save_path = os.path.join(save_root,os.path.basename(video_path)[:-4]+"_whisper.json")
        with open(save_path, 'w') as fp:
            json.dump(result, fp, sort_keys=True, indent=4, ensure_ascii=False)
     
'''

CUDA_VISIBLE_DEVICES=0 python run_whisper.py -x 0/16 &
CUDA_VISIBLE_DEVICES=0 python run_whisper.py -x 1/16 &
CUDA_VISIBLE_DEVICES=1 python run_whisper.py -x 2/16 &
CUDA_VISIBLE_DEVICES=1 python run_whisper.py -x 3/16 &
CUDA_VISIBLE_DEVICES=2 python run_whisper.py -x 4/16 &
CUDA_VISIBLE_DEVICES=2 python run_whisper.py -x 5/16 &
CUDA_VISIBLE_DEVICES=3 python run_whisper.py -x 6/16 &
CUDA_VISIBLE_DEVICES=3 python run_whisper.py -x 7/16 &
CUDA_VISIBLE_DEVICES=4 python run_whisper.py -x 8/16 &
CUDA_VISIBLE_DEVICES=4 python run_whisper.py -x 9/16 &
CUDA_VISIBLE_DEVICES=5 python run_whisper.py -x 10/16 &
CUDA_VISIBLE_DEVICES=5 python run_whisper.py -x 11/16 &
CUDA_VISIBLE_DEVICES=6 python run_whisper.py -x 12/16 &
CUDA_VISIBLE_DEVICES=6 python run_whisper.py -x 13/16 &
CUDA_VISIBLE_DEVICES=7 python run_whisper.py -x 14/16 &
CUDA_VISIBLE_DEVICES=7 python run_whisper.py -x 15/16 &

'''