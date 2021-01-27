# -*- coding: utf-8 -*-
import algo
import cv2
import h5py
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt  # NOQA
import numpy as np
import pandas as pd
import pickle
import torch
import torch.utils.data as data
import utils

from matplotlib.transforms import Affine2D
from os.path import basename, isfile, join, splitext
from scipy.interpolate import BPoly


def split_train_val(datafile):
    df = pd.read_csv(datafile)
    train_idx = df['Train'].values.astype(np.bool)
    df_train = df[train_idx]
    train_list = [(row.Image, row.Name, row.Contour) for row in df_train.itertuples()]
    df_valid = df[~train_idx]
    valid_list = [(row.Image, row.Name, row.Contour) for row in df_valid.itertuples()]

    return train_list, valid_list


class CoarseDataset(data.Dataset):
    def __init__(self, input_list, height, width, pad, random_warp=False):
        super(CoarseDataset, self).__init__()
        self.input_list = input_list
        self.height = height
        self.width = width
        self.pad = pad
        self.random_warp = random_warp

    def __getitem__(self, index):
        image_fpath, _, contour_fpath = self.input_list[index]
        image = cv2.imread(image_fpath)
        with open(contour_fpath, 'rb') as f:
            contour_data = pickle.load(f)
        pts = contour_data['contour']
        pts_xy_int = np.round(pts).astype(np.int32)
        radii, occluded = contour_data['radii'], contour_data['occluded']

        radii = np.zeros(radii.shape) + 12  # Test

        radii_int = np.floor(radii).astype(np.int32)

        # probs = utils.points_to_mask(pts_xy_int, radii_int, occluded, image.shape[0:2])

        # Expand the bounding box to include the coarse contour.
        x0 = (pts_xy_int[:, 0] - radii_int).min() - 1
        x1 = (pts_xy_int[:, 0] + radii_int).max() + 1
        y0 = (pts_xy_int[:, 1] - radii_int).min() - 1
        y1 = (pts_xy_int[:, 1] + radii_int).max() + 1

        # Pad to ensure we still get the whole contour after rotation.
        x0 = x0 - self.pad * (x1 - x0)
        x1 = x1 + self.pad * (x1 - x0)
        y0 = y0 - self.pad * (y1 - y0)
        y1 = y1 + self.pad * (y1 - y0)

        x0 = x0.clip(0, image.shape[1])
        x1 = x1.clip(0, image.shape[1])
        y0 = y0.clip(0, image.shape[0])
        y1 = y1.clip(0, image.shape[0])

        if self.random_warp:
            # Random rotation.
            max_theta = np.pi / 6.0
            theta = 2 * max_theta * np.random.random() - max_theta
        else:
            theta = 0.0

        # Width/height of the box does not change under a similarity trans.
        width = int(np.round(np.linalg.norm(np.array([x0, y0]) - np.array([x1, y0]))))
        height = int(np.round(np.linalg.norm(np.array([x0, y0]) - np.array([x0, y1]))))

        # Approximately the center of the contour bounding box.
        center = (x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0)

        v_x = (np.cos(theta), np.sin(theta))
        v_y = (-np.sin(theta), np.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
        mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y], [0, 0, 1]])
        mapping = np.linalg.inv(mapping)

        pts_xy_int = np.reshape(pts_xy_int, (pts_xy_int.shape[0], 1, 2))
        new_pts_xy_int = cv2.transform(pts_xy_int, mapping)[:, 0, :2]
        new_pts_xy_int = np.round(
            np.array([self.width / width, self.height / height]) * new_pts_xy_int
        ).astype(np.int32)
        target = utils.points_to_mask(
            new_pts_xy_int, radii_int, occluded, np.array([self.height, self.width])
        )

        crop = utils.sub_image(
            image, center, theta, width, height, border_mode=cv2.BORDER_CONSTANT
        )
        # target = utils.sub_image(probs, center, theta, width, height,
        #                         border_mode=cv2.BORDER_CONSTANT)
        crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # target = cv2.resize(target, (self.width, self.height),
        #                    interpolation=cv2.INTER_AREA)
        # Clean up any interpolation artifacts.
        _, target = cv2.threshold(target, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        crop = crop.transpose(2, 0, 1) / 255.0
        crop = torch.FloatTensor(crop)
        target = torch.LongTensor(target)

        return crop, target, index

    def __len__(self):
        return len(self.input_list)


class CoarseEvalDataset(data.Dataset):
    def __init__(self, rows, images_targets, height, width, pad):
        super(CoarseEvalDataset, self).__init__()
        self.rows = rows
        self.images_targets = images_targets
        self.height = height
        self.width = width
        self.pad = pad

    def __getitem__(self, index):
        row = self.rows[index]

        img = cv2.imread(self.images_targets[row].path)
        crop, _ = utils.crop_with_padding(img, row.x, row.y, row.w, row.h, self.pad)
        if row.Mirror:
            crop = crop[:, ::-1]
        crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
        crop = crop.transpose(2, 0, 1) / 255.0
        crop = torch.FloatTensor(crop)

        return crop, index

    def __len__(self):
        return len(self.rows)


class FineDataset(data.Dataset):
    def __init__(
        self,
        input_list,
        cachedir,
        height1,
        width1,
        height2,
        width2,
        patch_size,
        num_samples,
        num_fixed,
    ):
        super(FineDataset, self).__init__()
        # Height and width of images used to compute contour points.
        self.height1 = height1
        self.width1 = width1
        # Height and width of images used to extract patches.
        self.height2 = height2
        self.width2 = width2
        self.input_list = input_list
        self.cachedir = cachedir
        # The height and width of the patch to extract.
        self.patch_size = patch_size
        # The number of contour points to sample for a single image.
        self.num_samples = num_samples
        # The number of points to give supervised labels.
        self.num_fixed = num_fixed

    def __getitem__(self, index):
        image_fpath, _, contour_fpath = self.input_list[index]

        target = join(self.cachedir, '%s.h5' % (splitext(basename(contour_fpath))[0]))
        if isfile(target):
            with h5py.File(target, 'r') as h5f:
                img = h5f['image'][:]
                pts_xy = h5f['pts_xy'][:]
                pts_normal_xy = h5f['pts_normal_xy'][:]
                height_ratio = h5f['height_ratio'][()]
                width_ratio = h5f['width_ratio'][()]
        else:
            img = cv2.imread(image_fpath)
            with open(contour_fpath, 'rb') as f:
                contour_data = pickle.load(f)
            pts = contour_data['contour']
            pts_xy_int = np.round(pts).astype(np.int32)
            radii, occluded = contour_data['radii'], contour_data['occluded']
            radii_int = np.floor(radii).astype(np.int32)

            probs = utils.points_to_mask(pts_xy_int, radii_int, occluded, img.shape[0:2])

            x0 = (pts_xy_int[:, 0] - radii_int).min() - 1
            x1 = (pts_xy_int[:, 0] + radii_int).max() + 1
            y0 = (pts_xy_int[:, 1] - radii_int).min() - 1
            y1 = (pts_xy_int[:, 1] + radii_int).max() + 1

            x0 = x0.clip(0, img.shape[1])
            x1 = x1.clip(0, img.shape[1])
            y0 = y0.clip(0, img.shape[0])
            y1 = y1.clip(0, img.shape[0])

            part_img = img[y0:y1, x0:x1]
            part_probs = probs[y0:y1, x0:x1]

            # working_size = (self.width2, self.height2)
            # working_part_img = cv2.resize(part_img, working_size,
            #                              interpolation=cv2.INTER_AREA)
            # Computes points at 256x256 but upsamples to 1024x1024.
            resized_probs = cv2.resize(
                part_probs, (self.width1, self.height1), interpolation=cv2.INTER_AREA
            )
            peaks_ij, normals, is_max = algo.control_points(resized_probs)
            pts_xy = peaks_ij[is_max][:, ::-1]
            pts_normal_xy = normals[is_max][:, ::-1]

            M = np.array(
                [
                    [1.0 * part_img.shape[1] / self.width1, 0.0],
                    [0.0, 1.0 * part_img.shape[0] / self.height1],
                ]
            )
            pts_xy = cv2.transform(np.array([pts_xy]), M)[0]
            pts_xy += np.array([x0, y0])
            pts_normal_xy[:, 0] *= M[0, 0]
            pts_normal_xy[:, 1] *= M[1, 1]
            pts_normal_xy /= np.linalg.norm(pts_normal_xy, axis=1)[:, None]

            height_ratio = 1.0 * part_img.shape[0] / self.height2
            width_ratio = 1.0 * part_img.shape[1] / self.width2

            with h5py.File(target, 'w') as h5f:
                h5f.create_dataset('image', data=img)
                h5f.create_dataset('height_ratio', data=height_ratio)
                h5f.create_dataset('width_ratio', data=width_ratio)
                h5f.create_dataset('pts_xy', data=pts_xy)
                h5f.create_dataset('pts_normal_xy', data=pts_normal_xy)

        # Use replace=True, might not have enough samples.
        sample_idx = np.random.choice(np.arange(pts_xy.shape[0]), size=self.num_samples)

        pts_xy = pts_xy[sample_idx]
        pts_normal_xy = pts_normal_xy[sample_idx]

        thetas = np.arctan2(pts_normal_xy[:, 1], pts_normal_xy[:, 0])

        # Width refers to the direction along the normal.
        patch_diag = int(np.round(np.sqrt(2 * self.patch_size ** 2)))
        # Crop dims. in the working size image (part).
        crop_width = max(10 * self.num_fixed, patch_diag)
        crop_height = patch_diag
        patch_size = self.patch_size

        # Crop dims. in the original image.
        img_crop_width = int(np.round(width_ratio * crop_width))
        img_crop_height = int(np.round(height_ratio * crop_height))

        # The distance from either end to the end of the stitching region.
        x0 = 2 * self.num_fixed
        x1 = x0 + 2 * self.num_fixed

        def bernstein_poly(x, coeffs, interval):
            f = BPoly(coeffs, interval, extrapolate=False)

            return f(x)

        def stitch(width, height):
            coeffs = np.random.random(10).reshape(-1, 1)
            x = np.linspace(0, height - 1, height)
            fx = width * bernstein_poly(x, coeffs, np.array([0.0, height]))
            y = np.linspace(0, width - 1, width)
            grid = np.repeat(y.reshape(-1, 1), height, axis=1)
            dist = grid - fx[None]

            poly = np.vstack((x, fx)).T
            t = 2
            matte = dist.clip(-t, t)
            matte = (matte + t) / (2.0 * t)

            return matte.T[:, :, None], poly

        synth_patches = np.empty(
            (pts_xy.shape[0], 3, patch_size, patch_size), dtype=np.float32
        )
        real_patches = np.empty(
            (pts_xy.shape[0], 3, patch_size, patch_size), dtype=np.float32
        )
        targets = np.empty((pts_xy.shape[0], patch_size, patch_size), dtype=np.float32)
        for i, _ in enumerate(pts_xy):
            crop = utils.sub_image(
                img, pts_xy[i], thetas[i], img_crop_width, img_crop_height
            )
            crop = cv2.resize(
                crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA
            )

            neg_source = crop[:, x0:x1]
            pos_source = crop[:, -x1:-x0]
            matte, poly = stitch(2 * self.num_fixed, crop_height)

            pos_fill = crop[:, -x0:]
            neg_fill = crop[:, :x0]
            stitched = matte * pos_source + (1.0 - matte) * neg_source
            stitched = stitched.astype(np.uint8)
            synth = np.hstack((neg_fill, stitched, pos_fill))

            # 1 in the transition region and 0 elsewhere.
            transition = np.zeros(synth.shape[0:2], dtype=np.float32)
            idx = np.argwhere((0.0 < matte[:, :, 0]) & (matte[:, :, 0] < 1.0))
            transition[idx[:, 0], x0 + idx[:, 1]] = 1.0

            # The center of the synthetic patch is the center of the original
            # crop without the real boundary and one overlap region.
            synth_center = (
                np.array([crop_width - 4.0 * self.num_fixed, crop_height]) / 2.0
            )
            theta = -thetas[i]
            # The axis-aligned patch cropped from inside the synthetic patch.
            patch = utils.sub_image(synth, synth_center, theta, patch_size, patch_size)
            target = utils.sub_image(
                transition, synth_center, theta, patch_size, patch_size
            )
            # The real patch is taken from the center of the original crop.
            real_center = np.array([crop_width, crop_height]) / 2.0
            real = utils.sub_image(crop, real_center, theta, patch_size, patch_size)
            synth_patches[i] = patch.transpose(2, 0, 1) / 255.0
            real_patches[i] = real.transpose(2, 0, 1) / 255.0
            targets[i] = target

            visualize = False
            if visualize:
                # Represent the poly. in the same coordinates as the patch.
                poly = poly[:, ::-1]
                poly[:, 0] += x0

                # Rotate-and-shift the poly.
                trans = cv2.getRotationMatrix2D(
                    tuple(synth_center), np.rad2deg(theta), 1.0
                )
                poly_rot = cv2.transform(np.array([poly]), trans)[0]
                poly_rot -= synth_center - patch_size / 2.0

                (valid_idx,) = np.where(
                    (poly_rot[:, 0] >= 0)
                    & (poly_rot[:, 0] < patch_size)
                    & (poly_rot[:, 1] >= 0)
                    & (poly_rot[:, 1] < patch_size)
                )
                poly_rot = poly_rot[valid_idx]

                f, axarr = plt.subplots(3, 2)
                xy = pts_xy[i] - np.array([img_crop_width / 2.0, img_crop_height / 2.0])
                axarr[:, 0] = plt.subplot2grid((2, 2), (0, 0), rowspan=3)
                axarr[0, 0].imshow(img[:, :, ::-1])
                axarr[0, 0].scatter(pts_xy[i, 0], pts_xy[i, 1], color='red', s=5)
                crop_rect = mpl_patches.Rectangle(
                    xy,
                    img_crop_width,
                    img_crop_height,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                )
                crop_trans = Affine2D().rotate_around(
                    pts_xy[i, 0], pts_xy[i, 1], thetas[i]
                )
                crop_rect.set_transform(crop_trans + plt.gca().transData)
                xy = pts_xy[i] - np.array(
                    [width_ratio * patch_size / 2.0, height_ratio * patch_size / 2.0]
                )
                axarr[0, 0].add_patch(crop_rect)
                patch_rect = mpl_patches.Rectangle(
                    xy,
                    width_ratio * patch_size,
                    height_ratio * patch_size,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                )
                axarr[0, 0].add_patch(patch_rect)

                xy = synth_center - np.array([patch_size / 2.0, patch_size / 2.0])
                crop_rect = mpl_patches.Rectangle(
                    xy,
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                )
                crop_trans = Affine2D().rotate_around(
                    synth_center[0], synth_center[1], -thetas[i]
                )
                crop_rect.set_transform(crop_trans + axarr[0, 1].transData)
                axarr[0, 1].add_patch(crop_rect)
                axarr[0, 1].set_title('Synthetic Patch')
                axarr[0, 1].imshow(synth[:, :, ::-1])
                axarr[0, 1].plot(poly[:, 0], poly[:, 1], color='red')
                axarr[1, 1].plot(poly[:, 0], poly[:, 1], color='red')
                axarr[1, 1].plot(
                    crop_width - (x1 + x0) + poly[:, 0], poly[:, 1], color='red'
                )
                axarr[1, 1].set_title('Real Patch')
                axarr[1, 1].axvline(x=x0, color='blue')
                axarr[1, 1].axvline(x=x1, color='blue')
                axarr[1, 1].axvline(x=crop_width - x1, color='red')
                axarr[1, 1].axvline(x=crop_width - x0, color='red')
                axarr[1, 1].imshow(crop[:, :, ::-1])
                axarr[2, 1].imshow(patch[:, :, ::-1])
                axarr[2, 1].plot(poly_rot[:, 0], poly_rot[:, 1], color='red')
                plt.show()
                for ax in axarr.flat:
                    ax.clear()

        if visualize:
            exit(0)

        X1 = torch.FloatTensor(synth_patches)
        X2 = torch.FloatTensor(real_patches)
        y = torch.LongTensor(targets)

        return X1, X2, y, index

    def __len__(self):
        return len(self.input_list)


class RegressionDataset(data.Dataset):
    def __init__(self, input_list, height, width, pad, random_warp=False):
        super(RegressionDataset, self).__init__()
        self.input_list = input_list
        self.height = height
        self.width = width
        self.pad = pad
        self.random_warp = random_warp

    def __getitem__(self, index):
        image_fpath, _, contour_fpath = self.input_list[index]
        image = cv2.imread(image_fpath)
        with open(contour_fpath, 'rb') as f:
            contour_data = pickle.load(f)
        pts_xy = contour_data['contour']
        pts_xy_int = np.round(pts_xy).astype(np.int32)

        # radii, occluded = contour_data['radii'], contour_data['occluded']
        radii = contour_data['radii']
        radii_int = np.floor(radii).astype(np.int32)
        # Expand the bounding box to include the coarse contour.
        x0 = (pts_xy_int[:, 0] - radii_int).min() - 1
        x1 = (pts_xy_int[:, 0] + radii_int).max() + 1
        y0 = (pts_xy_int[:, 1] - radii_int).min() - 1
        y1 = (pts_xy_int[:, 1] + radii_int).max() + 1

        # Pad to ensure we still get the whole contour after rotation.
        x0 = x0 - self.pad * (x1 - x0)
        x1 = x1 + self.pad * (x1 - x0)
        y0 = y0 - self.pad * (y1 - y0)
        y1 = y1 + self.pad * (y1 - y0)

        x0 = x0.clip(0, image.shape[1])
        x1 = x1.clip(0, image.shape[1])
        y0 = y0.clip(0, image.shape[0])
        y1 = y1.clip(0, image.shape[0])

        # Contours are traced from a consistent starting point
        start, end = pts_xy[0], pts_xy[-1]

        # if self.random_warp:
        #     # Random rotation.
        #     max_theta = np.pi / 12.0
        #     theta = 2 * max_theta * np.random.random() - max_theta
        # else:
        #     theta = 0.0

        # Width/height of the box does not change under a similarity trans.
        width = int(np.round(np.linalg.norm(np.array([x0, y0]) - np.array([x1, y0]))))
        height = int(np.round(np.linalg.norm(np.array([x0, y0]) - np.array([x0, y1]))))
        # Approximately the center of the contour bounding box.
        center = (x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0)
        crop = utils.sub_image(
            image, center, 0.0, width, height, border_mode=cv2.BORDER_CONSTANT
        )
        start[0] = (start[0] - x0) * (self.width / width)
        start[1] = (start[1] - y0) * (self.height / height)
        end[0] = (end[0] - x0) * (self.width / width)
        end[1] = (end[1] - y0) * (self.height / height)

        crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # center = (self.width / 2., self.height / 2.)
        # M = cv2.getRotationMatrix2D(center, np.rad2deg(theta), 1.)
        # start = cv2.transform(np.array([[start]]), M)[0][0]
        # end = cv2.transform(np.array([[end]]), M)[0][0]

        start[0] /= 1.0 * self.width
        start[1] /= 1.0 * self.height
        end[0] /= 1.0 * self.width
        end[1] /= 1.0 * self.height

        x = crop[:, :, ::-1] / 255.0
        # ImageNet normalization.
        x -= np.array([0.485, 0.456, 0.406])
        x /= np.array([0.229, 0.224, 0.225])
        x = x.transpose(2, 0, 1)
        x = torch.FloatTensor(x)

        start = torch.FloatTensor(start)
        end = torch.FloatTensor(end)

        return crop, x, start, end, index

    def __len__(self):
        return len(self.input_list)


class RegressionEvalDataset(data.Dataset):
    def __init__(self, rows, images_targets, height, width, pad):
        super(RegressionEvalDataset, self).__init__()
        self.rows = rows
        self.images_targets = images_targets
        self.height = height
        self.width = width
        self.pad = pad

    def __getitem__(self, index):
        row = self.rows[index]

        img = cv2.imread(self.images_targets[row].path)
        crop, _ = utils.crop_with_padding(img, row.x, row.y, row.w, row.h, self.pad)
        if row.Mirror:
            crop = crop[:, ::-1]
        crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
        crop = crop[:, :, ::-1] / 255.0
        crop -= np.array([0.485, 0.456, 0.406])
        crop /= np.array([0.229, 0.224, 0.225])
        crop = crop.transpose(2, 0, 1)
        crop = torch.FloatTensor(crop)

        return crop, index

    def __len__(self):
        return len(self.rows)
