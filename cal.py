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


def calculate_mean_size_char(words, fontpath, debug=False, inf_mean_hei=1.35):
    start = time.time()
    previous_font_size = None
    wid_list = []
    hei_list = []
    for word in words:
        x, y, width, height, text = word
        if isinstance(text, str) is False:
            continue
        if text is None or len(text) < 1 or isinstance(text, str) is False:
            continue
        if previous_font_size is not None:
            font_size = get_font_size2(text, width, height, font_path=fontpath, fontsize=previous_font_size, jumpsize=7,
                                       img_fraction=1)
        else:
            font_size = get_font_size2(text, width, height, font_path=fontpath)
            previous_font_size = font_size
        font = ImageFont.truetype(fontpath, font_size)
        for c in text:
            wc, hc = font.getsize(c)
            # image = Image.new("RGB", (800, 600))
            # draw = ImageDraw.Draw(image)
            # draw.text((0, 0), c, font=font)
            # wc, hc = image.getbbox()[2:]
            wid_list.append(wc)
            hei_list.append(hc)
    wid_list = np.array(wid_list)
    hei_list = np.array(hei_list)
    mean_wid = np.mean(wid_list)
    mean_hei = np.mean(hei_list)
    stop = time.time()
    if debug:
        print('mean_wid:', np.round_(mean_wid), '---mean_hei: ', np.round_(mean_hei), '---infer mean_hei: ',
              np.round_(mean_wid * inf_mean_hei), 'ratio: ', np.round_(mean_hei / mean_wid, 2))
        print('processing time: ', stop - start)
    if inf_mean_hei:
        return np.round_(mean_wid), np.round_(mean_wid * inf_mean_hei)
    return mean_wid, mean_hei


def Eval_wordbase_check():
    img_dir = 'images'
    box_dir = 'GT_word_icdar_refined1508'
    font_path = "font/timesbd.ttf"
    images = get_img_paths(img_dir)
    print('Eval_wordbase_check')
    for img_path in images:
        img = cv2.imread(img_path)
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(os.path.basename(label_path))
        data = pd.read_csv(label_path, usecols=np.r_[0:9], header=None)
        words_list = []
        for index, row in data.iterrows():
            x0, y0, x1, y1, x2, y2, x3, y3, text = list(row)
            wid = x1 - x0
            hei = y3 - y0
            if text != '':
                words_list.append([x0, y0, wid, hei, text])

        calculate_mean_size_char(words_list, font_path, debug=True, inf_mean_hei=1.35)


def Eval_charbase_check():
    img_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/images'
    box_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/GT_char_yolo_v1'
    images = get_img_paths(img_dir)
    # images = [images[3]]
    print('Eval_charbase_check')
    for img_path in images:
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(os.path.basename(label_path))
        all_line = []
        with open(label_path, 'r', encoding='utf-8') as lf:
            all_line = lf.read().splitlines()
        wid_list = []
        hei_list = []
        for index, row in enumerate(all_line):
            if row == '':
                continue
            row = row.split()
            box = row[1:]
            for idx, p in enumerate(box):
                box[idx] = int(p)
            x0, y0, wid, hei = box
            text = row[0]
            # if text in '\'*:,@.-(#%")/~!^&_+={}[]\;<>?※”$€£¥₫°²™ā–':
            #     # print(text)
            #     continue
            if isinstance(text, str) is False:
                continue
            if text != '' and text is not None:
                wid_list.append(wid)
                hei_list.append(hei)
        wid_list = np.array(wid_list)
        hei_list = np.array(hei_list)
        mean_wid = np.mean(wid_list)
        mean_hei = np.mean(hei_list)
        # stop = time.time()
        print('mean_wid:', np.round_(mean_wid), '---mean_hei: ', np.round_(mean_hei), 'ratio: ',
              np.round_(mean_hei / mean_wid, 2))

        # calculate_mean_size_char(words_list, font_path)


def Eval_charbase_check_2():
    img_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/images'
    box_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/GT_char_yolo_v1'
    font_path = "font/timesbd.ttf"
    images = get_img_paths(img_dir)
    # images = [images[3]]
    print('Eval_charbase_check_2')
    for img_path in images:
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(os.path.basename(label_path))
        all_line = []
        with open(label_path, 'r', encoding='utf-8') as lf:
            all_line = lf.read().splitlines()
        words_list = []
        for index, row in enumerate(all_line):
            if row == '':
                continue
            row = row.split()
            box = row[1:]
            for idx, p in enumerate(box):
                box[idx] = int(p)
            x0, y0, wid, hei = box
            text = row[0]
            if isinstance(text, str) is False:
                continue
            if text != '' and text is not None:
                words_list.append([x0, y0, wid, hei, text])

        calculate_mean_size_char(words_list, font_path, debug=True, inf_mean_hei=1.35)


def Cello_wordbase_check():
    img_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/images'
    box_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/GT_word_icdar_1908'
    font_path = "DejaVuSans.ttf"
    images = get_img_paths(img_dir)
    # images = [images[3]]
    print('Cello_wordbase_check')
    for img_path in images:
        img = cv2.imread(img_path)
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(os.path.basename(label_path))
        all_line = []
        with open(label_path, 'r', encoding='utf-8') as lf:
            # print()
            all_line = lf.read().splitlines()
        # for
        # data = pd.read_csv(label_path, usecols=np.r_[0:9], header=None)
        words_list = []
        for index, row in enumerate(all_line):
            if row == '':
                continue
            row = row.split(',')
            box = row[:8]
            for idx, p in enumerate(box):
                box[idx] = int(p)
            x0, y0, x1, y1, x2, y2, x3, y3 = box
            text = row[8:]
            text = ','.join(text)
            wid = x1 - x0
            hei = y3 - y0
            if isinstance(text, str) is False:
                continue
            if text != '' and text is not None:
                words_list.append([x0, y0, wid, hei, text])

        calculate_mean_size_char(words_list, font_path, debug=True)


def Cello_charbase_check():
    img_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/images'
    box_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/GT_char_yolo_mod/v7'
    font_path = "DejaVuSans.ttf"
    images = get_img_paths(img_dir)
    images = [images[3]]
    for img_path in images:
        img = cv2.imread(img_path)
        label_path = os.path.join(box_dir, os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        print(label_path)
        all_line = []
        with open(label_path, 'r', encoding='utf-8') as lf:
            all_line = lf.read().splitlines()
        words_list = []
        for index, row in enumerate(all_line):
            if row == '':
                continue
            row = row.split()
            box = row[1:]
            for idx, p in enumerate(box):
                box[idx] = int(p)
            x0, y0, wid, hei = box
            text = row[0]
            # text = ','.join(text)
            # wid = x1 - x0
            # hei = y3 - y0
            if isinstance(text, str) is False:
                continue
            if text != '' and text is not None:
                words_list.append([x0, y0, wid, hei, text])

        calculate_mean_size_char(words_list, font_path, debug=True)


if __name__ == '__main__':
    # Cello_wordbase_check()
    # Cello_charbase_check()
    Eval_wordbase_check()
    Eval_charbase_check()
    Eval_charbase_check_2()
    # image = Image.new("RGB", (800, 600))
    # draw = ImageDraw.Draw(image)
    # txt = "m"
    # font = ImageFont.truetype('font/timesbd.ttf', 100)
    # draw.text((0, 0), txt, font=font)  # put the text on the image
    # image = image.crop(image.getbbox())
    # image.save('hsvwheel_txt.png')  # save it
