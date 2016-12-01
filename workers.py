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
    coords_target = output_targets[fpath]['outline-coords']
    visual_target = output_targets[fpath]['outline-visual']

    loc_fpath = input1_targets[fpath]['localization-full'].path
    seg_fpath = input2_targets[fpath]['seg-data']
    key_fpath = input2_targets[fpath]['key-data']
    loc = cv2.imread(loc_fpath)
    with seg_fpath.open('rb') as f1,\
            key_fpath.open('rb') as f2:
        segm = pickle.load(f1)
        keyp = pickle.load(f2)

    outline = dorsal_utils.extract_outline(loc, segm, keyp)

    # TODO: what to write for failed extractions?
    if outline.shape[0] > 0:
        loc[outline[:, 0], outline[:, 1]] = (255, 0, 0)

    _, visual_buf = cv2.imencode('.png', loc)
    with coords_target.open('wb') as f1,\
            visual_target.open('wb') as f3:
        pickle.dump(outline, f1, pickle.HIGHEST_PROTOCOL)
        f3.write(visual_buf)


#input_targets: extract_high_resolution_outline_targets
def compute_block_curvature(fpath, scales, input_targets, output_targets):
    outline_coords_target = input_targets[fpath]['outline-coords']
    with open(outline_coords_target.path, 'rb') as f:
        outline = pickle.load(f)

    # no successful outline could be found
    if outline.shape[0] > 0:
        idx = dorsal_utils.separate_leading_trailing_edges(outline)
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
