import torch
import cv2
import time
import argparse
import numpy as np
from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, read_cap, draw_skel_and_kp
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import math
import tkinter as tk # Tkinter
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image # Pillow
import cv2 as cv # OpenCV
import os
import keyboard


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--conf_thres', type=float, default=0.2)
args = parser.parse_args()
if args.model == "movenet_lightning":
    args.size = 192
    args.ft_size = 48
else:
    args.size = 256
    args.ft_size = 64


def cosin_smi(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def webcam(file):
    model = load_model(args.model, ft_size=args.ft_size)
    cap = cv2.VideoCapture(cv2.CAP_DSHOW + args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    input_image, draw_image = read_imgfile(file, args.size)
    with torch.no_grad():
        input_image = torch.Tensor(input_image)

        kpt_with_conf_img = model(input_image)[0, 0, :, :]
        kpt_with_conf_img = kpt_with_conf_img.numpy()
        overlay_image1 = draw_skel_and_kp(
            draw_image, kpt_with_conf_img, conf_thres=args.conf_thres)
        keypoint_scores_img = kpt_with_conf_img[:, 2]
        keypoint_coords_img = kpt_with_conf_img[:, :2]

        img_pts = []
        for ks, kc in zip(keypoint_scores_img, keypoint_coords_img):
            if ks < args.conf_thres:
                img_pts.append([])
            else:
                img_pts.append([kc[1], kc[0]])

    #cv2.imshow('m',overlay_image1)



    while True:
        input_image, display_image = read_cap(cap, size=args.size)
        with torch.no_grad():
            input_image = torch.Tensor(input_image)
            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()

        overlay_image = draw_skel_and_kp(
            display_image, kpt_with_conf, conf_thres=args.conf_thres)

        keypoint_scores = kpt_with_conf[:, 2]
        keypoint_coords = kpt_with_conf[:, :2]

        pts = []
        for ks, kc in zip(keypoint_scores, keypoint_coords):
            if ks < args.conf_thres:
                pts.append([0, 0])
            else:
                pts.append([kc[1], kc[0]])

        try:
            cal_pts = np.array(pts[5]) - np.array(pts[7])
            cal_img_pts = np.array(img_pts[5]) - np.array(img_pts[7])
            smi = cosin_smi(cal_img_pts, cal_pts)
        except ValueError:
            pass

        keypoint_arr = [(5,7), (6,8), (7,9), (8,10), (11,13), (12,14), (13,15), (14,16)]
        smi_arr = []
        for i,k in keypoint_arr:
            try:
                cal_pts = np.array(pts[i]) - np.array(pts[k])
                cal_img_pts = np.array(img_pts[i]) - np.array(img_pts[k])
                smi = cosin_smi(cal_img_pts,cal_pts)
                smi_arr.append(smi)
            except ValueError:
                pass
        for idx,k in enumerate(smi_arr):
            i1,i2 = keypoint_arr[idx]
            pts_s = np.array((pts[i1], pts[i2]), np.int32).reshape(2, 2)
            if k < 0.85:
                if [0, 0] not in pts_s:
                    overlay_image = cv2.polylines(overlay_image, [pts_s], isClosed=False, color=(0, 0, 255), thickness=5)
            else:

                if [0, 0] not in pts_s:
                    overlay_image = cv2.polylines(overlay_image, [pts_s], isClosed=False, color=(0, 255,0), thickness=2)
        # if smi < 0.9:
        #     if [0, 0] not in pts_s:
        #         overlay_image = cv2.polylines(overlay_image, [pts_s], isClosed=False, color=(0, 0, 255), thickness=5)
        # else:
        #     overlay_image = cv2.polylines(overlay_image, [pts_s], isClosed=False, color=(0, 255, 0), thickness=2)
        try:
            smi_arr = [x for x in smi_arr if math.isnan(x) == False]
            smi_mean = str(sum(smi_arr) / len(smi_arr))
        except(ZeroDivisionError):
            pass
        cv2.putText(overlay_image, smi_mean, (150, 440), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,thickness=4, color=(0, 255, 0))
        model_img = cv2.imread(file)
        model_img = cv2.resize(model_img,dsize=(args.cam_width,args.cam_height))
        img = np.hstack((model_img,overlay_image))
        cv2.imshow('movenet', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

window = tk.Tk()
window.title("Nice Body")
def press():

    window.file = filedialog.askopenfilename(
        initialdir=f'{os.getcwd()}',
        title='사진 선택',
        filetypes=(('Image files','*.jpg'),('all files','*,*'),('Image files','*.jpeg'),('Image files','*.png'))
    )
    
    webcam(window.file)
    

def main():
    window.geometry('760x550')
    img =Image.open('background.png')
    bg = ImageTk.PhotoImage(img)
    label = Label(window, image=bg)
    label.place(x = -10,y = 0)

    
    L1 = tk.Label(window, text = 'NICE BODY',font=('Malgun Gothic',25),bg = "white")
    L1.pack(side='top',pady= 25)
    L2 = tk.Label(window, text = '따라할 운동사진을 선택 해주세요',font=('Malgun Gothic',12),bg = "white")
    L2.place(x=385,y=150)
    B1 = tk.Button(window,text = '사진 선택',font=('Malgun Gothic',12), bg = "white",command = press, width=15, height=5)
    B1.place(x= 400,y = 200)

    window.mainloop()

if __name__ == "__main__":
    main()