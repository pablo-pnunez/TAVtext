# -*- coding: utf-8 -*-
import cv2
import numpy as np


def read_and_normalize_images(paths=[], img_shape=None, base_path=None, augmenter=None):
    imgs = []

    for f in paths:
        train_img = cv2.imread(base_path + f)
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
        train_img = cv2.resize(train_img, dsize=img_shape)

        imgs.append(train_img)

    if augmenter is not None:
        imgs = augmenter(images=imgs)

    imgs = np.array(imgs) * 1.0
    imgs /= 127.5
    imgs -= 1.

    return imgs