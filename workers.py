import affine
import cv2
import cPickle as pickle
import dorsal_utils
import imutils
import numpy as np


def preprocess_images(fpath, imsize, output_targets):
    resz_target = output_targets[fpath]['resized']
    trns_target = output_targets[fpath]['transform']

    img = cv2.imread(fpath)
    resz, M = imutils.center_pad_with_transform(img, imsize)
    _, resz_buf = cv2.imencode('.png', resz)

    with resz_target.open('wb') as f1,\
            trns_target.open('wb') as f2:
        f1.write(resz_buf)
        pickle.dump(M, f2, pickle.HIGHEST_PROTOCOL)


# input_targets: localization_segmentation_targets
def extract_low_resolution_outline(fpath, input_targets, output_targets):
    coords_target = output_targets[fpath]['outline-coords']
    visual_target = output_targets[fpath]['outline-visual']

    loc_fpath = input_targets[fpath]['loc-lr'].path
    seg_fpath = input_targets[fpath]['seg-lr'].path
    loc = cv2.imread(loc_fpath)
    seg = cv2.imread(seg_fpath, cv2.IMREAD_GRAYSCALE)

    outline = dorsal_utils.extract_outline(seg)

    # TODO: what to write for failed extractions?
    if outline.shape[0] > 0:
        loc[outline[:, 1], outline[:, 0]] = (255, 0, 0)
    _, visual_buf = cv2.imencode('.png', loc)
    with coords_target.open('wb') as f1,\
            visual_target.open('wb') as f2:
        pickle.dump(outline, f1, pickle.HIGHEST_PROTOCOL)
        f2.write(visual_buf)


# input1_targets: preprocess_images
# input2_targets: localization_segmentation
def refine_localization_segmentation(fpath, scale, imsize,
                                     input1_targets, input2_targets,
                                     output_targets):
    loc_hr_target = output_targets[fpath]['loc-hr']
    seg_hr_target = output_targets[fpath]['seg-hr']

    Mpre_path = input1_targets[fpath]['transform'].path
    Mloc_path = input2_targets[fpath]['transform'].path
    seg_lr_path = input2_targets[fpath]['seg-lr'].path

    # the original image
    img = cv2.imread(fpath)
    # the output of the segmentation network
    seg_lr = cv2.imread(seg_lr_path, cv2.IMREAD_GRAYSCALE)

    with open(Mpre_path, 'rb') as f1,\
            open(Mloc_path, 'rb') as f2:
        # the preprocessing transform: constrained similarity
        Mpre = pickle.load(f1)
        # the localization transform: affine
        Mloc = pickle.load(f2)

    loc_hr, seg_hr = imutils.refine_localization_segmentation(
        img, seg_lr, Mpre, Mloc, scale, imsize
    )
    _, loc_hr_buf = cv2.imencode('.png', loc_hr)
    _, seg_hr_buf = cv2.imencode('.png', seg_hr)
    with loc_hr_target.open('wb') as f1,\
            seg_hr_target.open('wb') as f2:
        f1.write(loc_hr_buf)
        f2.write(seg_hr_buf)


#input1_targets: extract_low_resolution_outline_targets
#input2_targets: localization_segmentation_targets
def extract_high_resolution_outline(fpath, scale,
                                    input1_targets, input2_targets,
                                    output_targets):
    coords_target = output_targets[fpath]['outline-coords']
    visual_target = output_targets[fpath]['outline-visual']
    lr_coord_fpath =\
        input1_targets[fpath]['outline-coords'].path
    loc_fpath = input2_targets[fpath]['loc-hr'].path
    seg_fpath = input2_targets[fpath]['seg-hr'].path

    loc = cv2.imread(loc_fpath)
    seg = cv2.imread(seg_fpath, cv2.IMREAD_GRAYSCALE)

    with open(lr_coord_fpath, 'rb') as f:
        lr_coords = pickle.load(f)

    if lr_coords.shape[0] > 0:
        # NOTE: original contour is at 128x128
        Mscale = affine.build_scale_matrix(2 * scale)
        hr_coords = affine.transform_points(Mscale, lr_coords)

        hr_coords_refined = dorsal_utils.extract_contour_refined(
            loc, seg, hr_coords
        )
    # the lr outline extraction failed
    else:
        hr_coords_refined = np.array([])

    if hr_coords_refined.shape[0] > 0:
        # round, convert to int, and clip before plotting
        hr_coords_refined_int = np.round(
            hr_coords_refined).astype(np.int32)
        hr_coords_refined_int[:, 0] = np.clip(
            hr_coords_refined_int[:, 0], 0, loc.shape[1] - 1)
        hr_coords_refined_int[:, 1] = np.clip(
            hr_coords_refined_int[:, 1], 0, loc.shape[0] - 1)
        loc[
            hr_coords_refined_int[:, 1], hr_coords_refined_int[:, 0]
        ] = (255, 0, 0)

    _, visual_buf = cv2.imencode('.png', loc)
    with coords_target.open('wb') as f1,\
            visual_target.open('wb') as f2:
        pickle.dump(hr_coords_refined, f1, pickle.HIGHEST_PROTOCOL)
        f2.write(visual_buf)


#input_targets: extract_high_resolution_outline_targets
def compute_block_curvature(fpath, scales, input_targets, output_targets):
    input_target = input_targets[fpath]['outline-coords']
    with open(input_target.path, 'rb') as f:
        outline = pickle.load(f)

    # no successful outline could be found
    if outline.shape[0] > 0:
        idx = dorsal_utils.separate_leading_trailing_edges(outline)
        # could not separate leading/trailing edges
        if idx is not None:
            te = outline[idx:]
            curv = dorsal_utils.block_curvature(te, scales)
        else:
            curv = None
    else:
        curv = None

    curv_target = output_targets[fpath]['curvature']
    # write the failures too or it seems like the task did not complete
    with curv_target.open('wb') as f1:
        pickle.dump(curv, f1, pickle.HIGHEST_PROTOCOL)
