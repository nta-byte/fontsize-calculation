import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import time


def get_img_paths(dir_, extensions=('.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG', 'jfif')):
    img_paths = []

    for root, dirs, files in os.walk(dir_):
        for file in files:
            for e in extensions:
                if file.endswith(e):
                    p = os.path.join(root, file)
                    img_paths.append(p)

    return img_paths


def get_font_size2(text, wid, height, font_path=None):
    if not font_path:
        font_path = "DejaVuSans.ttf"
    img_fraction = .3
    photo = Image.new("1", (wid, height))
    breakpoint = img_fraction * photo.size[0]
    jumpsize = 7
    fontsize = 6
    font = ImageFont.FreeTypeFont(font_path, fontsize)
    # print(font.getlength(text))
    while True:
        if font.getlength(text) < breakpoint:
            fontsize += jumpsize
        else:
            jumpsize = int(jumpsize / 2)
            fontsize -= jumpsize
        font = ImageFont.FreeTypeFont(font_path, fontsize)
        if jumpsize <= 1:
            break
    fontsize = int(fontsize/img_fraction)
    return fontsize


img_dir = 'images'
box_dir = 'GT_word_icdar_refined1508'

images = get_img_paths(img_dir)

for img_path in images:
    img = cv2.imread(img_path)
    label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
    print(label_path)
    data = pd.read_csv(label_path, usecols=np.r_[0:9], header=None)
    wid_list = []
    hei_list = []
    start = time.time()
    for index, row in data.iterrows():
        x0, y0, x1, y1, x2, y2, x3, y3, text = list(row)
        wid = x1 - x0
        hei = y3 - y0

        font_size = get_font_size2(text, wid, hei)
        # print(font_size, wid, hei, text)
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        for c in text:
            # print(c)
            wc, hc = font.getsize(c)
            wid_list.append(wc)
            hei_list.append(hc)
    wid_list = np.array(wid_list)
    hei_list = np.array(hei_list)
    mean_wid = np.mean(wid_list)
    mean_hei = np.mean(hei_list)
    stop = time.time()
    print('mean_wid:', mean_wid, '---mean_hei: ', mean_hei)
    print('processing time: ', stop - start)
