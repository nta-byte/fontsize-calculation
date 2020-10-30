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


def get_font_size2(text, wid, height, font_path=None, fontsize=6, jumpsize=7, img_fraction=.3):
    if not font_path:
        font_path = "DejaVuSans.ttf"
    photo = Image.new("1", (wid, height))
    break_point = img_fraction * photo.size[0]
    font_ = ImageFont.FreeTypeFont(font_path, fontsize)
    while True:
        if font_.getlength(text) < break_point:
            fontsize += jumpsize
        else:
            jumpsize = int(jumpsize / 2)
            fontsize -= jumpsize
        font_ = ImageFont.FreeTypeFont(font_path, fontsize)
        if jumpsize <= 1:
            break
    fontsize = int(fontsize / img_fraction)
    return fontsize


def calculate_mean_size_char(words, fontpath):
    start = time.time()
    previous_font_size = None
    wid_list = []
    hei_list = []
    for word in words:
        x, y, width, height, text = word
        if previous_font_size is not None:
            font_size = get_font_size2(text, width, height, font_path=fontpath, fontsize=previous_font_size, jumpsize=7,
                                       img_fraction=1)
        else:
            font_size = get_font_size2(text, width, height, font_path=fontpath)
            previous_font_size = font_size
        font = ImageFont.truetype(fontpath, font_size)
        for c in text:
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
    return mean_wid, mean_hei


if __name__ == '__main__':

    img_dir = 'images'
    box_dir = 'GT_word_icdar_refined1508'
    font_path = "DejaVuSans.ttf"
    images = get_img_paths(img_dir)

    for img_path in images:
        img = cv2.imread(img_path)
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(label_path)
        data = pd.read_csv(label_path, usecols=np.r_[0:9], header=None)
        words_list = []
        for index, row in data.iterrows():
            x0, y0, x1, y1, x2, y2, x3, y3, text = list(row)
            wid = x1 - x0
            hei = y3 - y0
            words_list.append([x0, y0, wid, hei, text])

        calculate_mean_size_char(words_list, font_path)
