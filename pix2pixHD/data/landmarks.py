import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def get_relative_landmarks(meta, frame_num):
    centerx, centery, l = meta['bbox'][frame_num - 1]
    orig_height = meta['length'].item()
    orig_width = meta['width'].item()
    landmarks = meta['landmarks_2d'][frame_num - 1]
    
    # Go from frame landmarks to cropped and resized frame landmarks
    x_left = max(0, centerx-l)
    x_right = min(centerx+l, orig_height)
    y_up = max(0, centery-l)
    y_down = min(centery+l, orig_width)
    w = x_right - x_left
    h = y_down - y_up
    ar_h = 255. / h
    ar_w = 255. / w

    landmarks[:,0] -= (centery - l)
    landmarks[:,1] -= (centerx - l)
    landmarks[:,0] *= ar_h
    landmarks[:,1] *= ar_w
    
    return landmarks

def plot_landmarks(landmarks):
    fig = plt.figure(figsize=(256, 256), dpi=1)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones((256, 256, 3)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    lw = 100

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=lw)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=lw)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=lw)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=lw)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=lw)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=lw)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=lw)
    ax.plot([landmarks[36, 0], landmarks[41, 0]], [landmarks[36, 1], landmarks[41, 1]], 
            linestyle='-', color='red', lw=lw)
    ax.plot([landmarks[42, 0], landmarks[47, 0]], [landmarks[42, 1], landmarks[47, 1]], 
            linestyle='-', color='red', lw=lw)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=lw)
    ax.plot([landmarks[48, 0], landmarks[59, 0]], [landmarks[48, 1], landmarks[59, 1]], 
            linestyle='-', color='purple', lw=lw)

    fig.canvas.draw()
    data = Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data