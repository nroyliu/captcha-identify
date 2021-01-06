# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time

import numpy as np
from PIL import Image
import cv2
import pytesseract
from urllib.request import urlretrieve

def captcha(image_url):
    # 读取图片
    img = cv2.imread(image_url)
    # 图片灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化
    ret, img_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
    # 提取内容轮廓
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历内容轮廓
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.selectROI(windowName="roi", img=img, showCrosshair=True, fromCenter=False)
        img_roi = img_inv[y - 1:y + h + 1, x - 1:x + w + 1]

        img_std = cv2.resize(img_roi,(11, 15))
        cv2.imshow('image', img_std)
        key = cv2.waitKey(0)
        char = chr(key)
        timestamp = int(time.time() * 1e6);
        print(timestamp)
        cv2.imwrite('./number/{}_{}.jpg'.format(timestamp + y, char), img_std)

def draw_roi(image_url):
    # 读取图片
    img = cv2.imread(image_url)
    # 图片灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化
    ret, img_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # 提取内容轮廓
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历内容轮廓
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.selectROI(windowName="roi", img=img, showCrosshair=True, fromCenter=False)
        # cv2.imshow('image', img_std)
        cv2.rectangle(img_inv, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 77, 77), 1)
        cv2.imshow('image_roi',img_inv)
    # cv2.imwrite('./image_gray.jpg', img_inv)
    # text = pytesseract.image_to_string(img_gray, lang="eng")
    # print(text)
    # img = Image.open('./image_gray.jpg')
    # img.show()

def train():
    files = os.listdir('number')
    samples = np.empty((0, 165))
    labels = []
    for filename in files:
        filepath = os.path.join('number', filename)
        labels.append(filename.split(".")[0].split('_')[-1])
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        sample = im.reshape((1,165)).astype(np.float32)
        samples = np.append(samples, sample, 0)
    unique_labels = list(set(labels))
    unique_ids = list(range(len(unique_labels)))
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    label_ids = list(map(lambda x: label_id_map[x], labels))
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
    samples = samples.astype(np.float32)

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)

    img = cv2.imread('./image.jpg')
    # 图片灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化
    ret, img_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # 提取内容轮廓
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历内容轮廓
    code = ''
    for contour in reversed(contours):
        x, y, w, h = cv2.boundingRect(contour)
        img_roi = img_inv[y - 1:y + h + 1, x - 1:x + w + 1]
        img_std = cv2.resize(img_roi, (11, 15))
        sample = img_std.reshape((1, 165)).astype(np.float32)
        ret, results, neighbours, distances = model.findNearest(sample, k=3)
        label_id = int(results[0][0])
        label = id_label_map[label_id]
        code += label
    print(code)
    cv2.imshow('code', img)
    cv2.waitKey(0)

def urllib_download(image_url):
    # 下载图片
    urlretrieve(image_url, './image.jpg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # while(1):
    urllib_download('http://cdsjwytx.139erp.com/image.jsp?t=0.28822175107232817')
    #     captcha('./image.jpg')
    # draw_roi('./image.jpg')
    train();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
