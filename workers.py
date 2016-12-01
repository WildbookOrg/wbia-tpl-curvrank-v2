import cv2
import cPickle as pickle
import dorsal_utils
import imutils


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


# input_targets: segmentation_targets
def extract_outline(fpath, input1_targets, input2_targets, output_targets):
    leading_coords_target = output_targets[fpath]['leading-coords']
    trailing_coords_target = output_targets[fpath]['trailing-coords']
    visual_target = output_targets[fpath]['outline-visual']

    loc_fpath = input1_targets[fpath]['localization-full'].path
    seg_fpath = input2_targets[fpath]['seg-data']
    key_fpath = input2_targets[fpath]['key-data']
    loc = cv2.imread(loc_fpath)
    with seg_fpath.open('rb') as f1,\
            key_fpath.open('rb') as f2:
        segm = pickle.load(f1)
        keyp = pickle.load(f2)

    leading, trailing = dorsal_utils.extract_outline(loc, segm, keyp)

    # TODO: what to write for failed extractions?
    if leading.shape[0] > 0:
        loc[leading[:, 0], leading[:, 1]] = (255, 0, 0)
    if trailing.shape[0] > 0:
        loc[trailing[:, 0], trailing[:, 1]] = (0, 0, 255)

    _, visual_buf = cv2.imencode('.png', loc)
    with leading_coords_target.open('wb') as f1,\
            trailing_coords_target.open('wb') as f2,\
            visual_target.open('wb') as f3:
        pickle.dump(leading, f1, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trailing, f2, pickle.HIGHEST_PROTOCOL)
        f3.write(visual_buf)


#input_targets: extract_high_resolution_outline_targets
def compute_block_curvature(fpath, scales, input_targets, output_targets):
    trailing_coords_target = input_targets[fpath]['trailing-coords']
    with open(trailing_coords_target.path, 'rb') as f:
        trailing = pickle.load(f)

    # no successful outline could be found
    if trailing.shape[0] > 0:
        curv = dorsal_utils.block_curvature(trailing, scales)
    else:
        curv = None

    curv_target = output_targets[fpath]['curvature']
    # write the failures too or it seems like the task did not complete
    with curv_target.open('wb') as f1:
        pickle.dump(curv, f1, pickle.HIGHEST_PROTOCOL)
