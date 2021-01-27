# -*- coding: utf-8 -*-
import cv2
import numpy as np


def plot_regression_samples(imgs, y0, y1, y0_hat, y1_hat, fpaths):
    y0 = y0.data.cpu().numpy()
    y1 = y1.data.cpu().numpy()
    y0_hat = y0_hat.data.cpu().numpy()
    y1_hat = y1_hat.data.cpu().numpy()
    for i, _ in enumerate(imgs):
        img0, img1 = imgs[i].copy(), imgs[i].copy()
        start = y0[i] * np.array(img0.shape[0:2][::-1])
        end = y1[i] * np.array(img0.shape[0:2][::-1])
        start_hat = y0_hat[i] * np.array(img0.shape[0:2][::-1])
        end_hat = y1_hat[i] * np.array(img0.shape[0:2][::-1])
        cv2.circle(img0, tuple(start.astype(np.int32)), 3, (255, 0, 0), -1)
        cv2.circle(img0, tuple(end.astype(np.int32)), 3, (0, 0, 255), -1)
        cv2.circle(img1, tuple(start_hat.astype(np.int32)), 3, (255, 0, 0), -1)
        cv2.circle(img1, tuple(end_hat.astype(np.int32)), 3, (0, 0, 255), -1)

        stacked = np.hstack((img0, img1))
        cv2.imwrite(fpaths[i], stacked)


def plot_real_and_synth_samples(x_synth, y, y_hat_synth, x_real, y_hat_real, fpaths):
    x_synth = x_synth.data.cpu().numpy().transpose(0, 2, 3, 1)
    x_real = x_real.data.cpu().numpy().transpose(0, 2, 3, 1)
    y = y.data.cpu().numpy().astype(np.uint8)
    y_hat_synth = y_hat_synth.data.cpu().numpy().transpose(0, 2, 3, 1)
    y_hat_real = y_hat_real.data.cpu().numpy().transpose(0, 2, 3, 1)

    for i in range(x_synth.shape[0]):
        stacked = np.hstack(
            (
                255 * x_synth[i, :, :, 0:3],
                255 * cv2.cvtColor(y[i], cv2.COLOR_GRAY2BGR),
                255 * cv2.cvtColor(y_hat_synth[i, :, :, 1], cv2.COLOR_GRAY2BGR),
                255 * x_real[i, :, :, 0:3],
                255 * cv2.cvtColor(y_hat_real[i, :, :, 1], cv2.COLOR_GRAY2BGR),
            )
        ).astype(np.uint8)
        cv2.imwrite(fpaths[i], stacked)


def plot_coarse_samples(x, y, y_hat, fpaths):
    x = x.data.cpu().numpy().transpose(0, 2, 3, 1)
    y = y.data.cpu().numpy().astype(np.uint8)
    y_hat = y_hat.data.cpu().numpy().transpose(0, 2, 3, 1)

    for i in range(x.shape[0]):
        stacked = np.hstack(
            (
                255 * x[i, :, :, 0:3],
                255 * cv2.cvtColor(y[i], cv2.COLOR_GRAY2BGR),
                255 * cv2.cvtColor(y_hat[i, :, :, 1], cv2.COLOR_GRAY2BGR),
            )
        ).astype(np.uint8)
        cv2.imwrite(fpaths[i], stacked)
