import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize_landmarks(landmarks):
    # normalize landmarks to 0-1 x,y,z
    # input, batch_size,478,3
    # normalize landmark to 0-1 x,y,z
    landmarks[:,:,0] = (landmarks[:,:,0] - landmarks[:,:,0].min())/(landmarks[:,:,0].max()-landmarks[:,:,0].min())
    landmarks[:,:,1] = (landmarks[:,:,1] - landmarks[:,:,1].min())/(landmarks[:,:,1].max()-landmarks[:,:,1].min())
    landmarks[:,:,2] = (landmarks[:,:,2] - landmarks[:,:,2].min())/(landmarks[:,:,2].max()-landmarks[:,:,2].min())
    # print(np.min(landmarks[:,:,0],axis=1))
    return landmarks

def get_video_frames(video_path,begin_frame_no,end_frame_no):
    frame_list = []
    frame_counter = 0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter >= begin_frame_no and frame_counter < end_frame_no:
            frame_list.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        frame_counter +=1
    return frame_list

def get_3d_points_bbox(landmarks):
    x = landmarks[:, :, 0]
    y = landmarks[:, :, 1]
    z = landmarks[:, :, 2]
    x1 = np.min(x, axis=1)
    y1 = np.min(y, axis=1)
    z1 = np.min(z, axis=1)
    x2 = np.max(x, axis=1)
    y2 = np.max(y, axis=1)
    z2 = np.max(z, axis=1)
    
    return x1, y1, z1, x2, y2, z2

def get_mouth_square(landmarks,h,w):
    mouth_left = landmarks[:,57,:]
    mouth_right = landmarks[:,287,:]
    bbox_x1,bbox_x2 = np.int32(mouth_left[:,0]*w),np.int32(mouth_right[:,0]*w)
    bbox_width = bbox_x2-bbox_x1
    bbox_ymid = np.int32((mouth_left[:,1]+mouth_right[:,1])/2*h)
    y_shift = np.int32(bbox_width*0.1)
    bbox_y1,bbox_y2 = bbox_ymid-bbox_width//2+y_shift,bbox_ymid+(bbox_width-bbox_width//2)+y_shift
    return bbox_x1,bbox_y1,bbox_x2,bbox_y2

def gen_mouth_grid(mouth_frames, output_config="336/14", method="simple"):
    """given a batch of mouth frames, generate mouth grid
    Args:
        mouth_frames_np (np.uint8): frames*h*w*c mouth images
        output_config (str, optional): output model config, represents how vit use it, such as 
            which means a square is 384*384 with each tile of 14*14"384/14".
        method (str, optional): method used for preprocessing and fitting into tiles,
        Defaults to "simple".
        "simple": simply resize the mouth frames to fit into each tile, black tiles if no mouth
            detected or video not long enough.
    """
    canvas_h,tile_h = output_config.split("/")
    canvas_h,canvas_w,tile_h,tile_w = int(canvas_h),int(canvas_h),int(tile_h),int(tile_h)
    if method == "simple":
        if mouth_frames.shape[0] > (canvas_h//tile_h)**2:
            raise ValueError(
                "video too long, mouth_frames shape: {}, total num of grid: {}".format(
                mouth_frames.shape[0],(canvas_h//tile_h)**2))
        mouth_grid = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i in range(mouth_frames.shape[0]):
            row = i // (canvas_h//tile_h)
            col = i % (canvas_h//tile_h)
            tile = cv2.resize(mouth_frames[i], (tile_w, tile_h))
            if len(tile.shape) == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
            mouth_grid[row*tile_h:(row+1)*tile_h,col*tile_w:(col+1)*tile_w] = tile
        return mouth_grid
    raise NotImplementedError("method {} not implemented".format(method))

def show_img_np(img, max_h=3, max_w=20, save=False, cmap='gray'):
    """
    :param np_array: input image, one channel or 3 channel,
    :param save: if save image
    :param size:
    :return:
    """
    if len(img.shape) < 3:
        plt.rcParams['image.cmap'] = cmap
    plt.figure(figsize=(max_w, max_h), facecolor='w', edgecolor='k')
    plt.imshow(img)
    if save:
        cv2.imwrite('debug.png', img)
    else:
        plt.show()
        