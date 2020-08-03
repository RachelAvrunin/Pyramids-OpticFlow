import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 304976335


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    ker = np.array([[1, 0, -1]])
    I_x = cv2.filter2D(im2, -1, ker, cv2.BORDER_REPLICATE)
    I_y = cv2.filter2D(im2, -1, ker.T, cv2.BORDER_REPLICATE)

    I_t = im2 - im1
    pts = []
    uv = []
    h, w = I_x.shape[:2]
    half_win = win_size // 2
    for i in range(half_win, h - half_win + 1, step_size):
        for j in range(half_win, w - half_win + 1, step_size):
            mat_x = I_x[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
            mat_y = I_y[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
            ATA = np.array([[np.sum(mat_x * mat_x), np.sum(mat_x * mat_y)],
                            [np.sum(mat_x * mat_y), np.sum(mat_y * mat_y)]])
            lambdas = np.linalg.eigvals(ATA)
            if lambdas.min() > 1 and lambdas.max() / lambdas.min() < 100:
                mat_t = I_t[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
                ATb = np.array([-np.sum(mat_x * mat_t), -np.sum(mat_y * mat_t)])
                local_uv = np.linalg.inv(ATA).dot(ATb)
                uv.append(local_uv * (-1))
                pts.append([j, i])
    pts = np.array(pts)
    uv = np.array(uv)
    return pts, uv


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    lapLst = []
    gauss_pyr = gaussianPyr(img, levels)

    gaussian = gaussianKer(5)
    for i in range(1, levels):
        expand = gaussExpand(gauss_pyr[i], gaussian)
        lap = gauss_pyr[i - 1] - expand
        lapLst.append(lap)

    lapLst.append(gauss_pyr[levels - 1])

    return lapLst


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaussian = gaussianKer(5)
    n = len(lap_pyr) - 1
    gauss_Pyr = lap_pyr[n]

    for i in range(n, 0, -1):
        expand = gaussExpand(gauss_Pyr, gaussian)
        gauss_Pyr = expand + lap_pyr[i - 1]
    return gauss_Pyr


def cropPic(img: np.ndarray, levels: int) -> np.ndarray:
    twoPowLevel = pow(2, levels)
    h, w = img.shape[:2]
    h = twoPowLevel * np.floor(h / twoPowLevel).astype(np.uint8)
    w = twoPowLevel * np.floor(w / twoPowLevel).astype(np.uint8)

    if img.ndim == 3:
        img = img[0:h, 0:w, :]
    else:
        img = img[0:h, 0:w]
    return img


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    img = cropPic(img, levels)
    pyrLst = [img]
    gaussian = gaussianKer(5)

    for i in range(1, levels):
        I_temp = cv2.filter2D(pyrLst[i - 1], -1, gaussian, cv2.BORDER_REPLICATE)
        I_temp = I_temp[::2, ::2]
        pyrLst.append(I_temp)
    return pyrLst


def gaussianKer(kernel_size: int) -> np.ndarray:
    gaussian = cv2.getGaussianKernel(kernel_size, -1)
    gaussian = gaussian.dot(gaussian.T)
    return gaussian


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    gs_k = (gs_k / gs_k.sum()) * 4
    if img.ndim == 3:
        h, w, d = img.shape[:3]
        newImg = np.zeros((2 * h, 2 * w, d))
    else:
        h, w = img.shape[:2]
        newImg = np.zeros((2 * h, 2 * w))
    newImg[::2, ::2] = img
    image = cv2.filter2D(newImg, -1, gs_k, cv2.BORDER_REPLICATE)
    return image


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    img_1_lap = laplaceianReduce(img_1, levels)
    img_2_lap = laplaceianReduce(img_2, levels)
    mask_gauss = gaussianPyr(mask)

    merge = (img_1_lap[levels - 1] * mask_gauss[levels - 1]) + ((1 - mask_gauss[levels - 1]) * img_2_lap[levels - 1])
    gaussian = gaussianKer(5)
    for i in range(levels - 2, -1, -1):
        merge = gaussExpand(merge, gaussian)
        merge = merge + (img_1_lap[i] * mask_gauss[i]) + ((1 - mask_gauss[i]) * img_2_lap[i])

    img_1 = cropPic(img_1, levels)
    img_2 = cropPic(img_2, levels)
    naive = (img_1 * mask_gauss[0]) + ((1 - mask_gauss[0]) * img_2)


    return naive, merge