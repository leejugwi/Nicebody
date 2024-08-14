import tkinter as tk # Tkinter
from PIL import ImageTk, Image # Pillow
import cv2 as cv # OpenCV
import os

win = tk.Tk() # 인스턴스 생성
win.title("Nicebody") # 제목 표시줄 추가
win.geometry("920x640") # 지오메트리: 너비x높이+x좌표+y좌표
win.resizable(False, False) # x축, y축 크기 조정 비활성화


lbl = tk.Label(win, text="Tkinter와 OpenCV를 이용한 GUI 프로그래밍")
lbl.grid(row=0, column=0) # 라벨 행, 열 배치

frm = tk.Frame(win, bg="blue", width=720, height=480) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치

lbl1 = tk.Label(frm)
lbl1.grid()

win.mainloop()
