import cPickle as pickle
import cv2
import h5py
import pandas as pd
import luigi
import costs
import datasets
import model
import logging
import multiprocessing as mp
import numpy as np

from collections import defaultdict
from functools import partial
from luigi.util import inherits
from time import time
from tqdm import tqdm
from os.path import basename, exists, isfile, join, splitext


logger = logging.getLogger('luigi-interface')


class HDF5LocalTarget(luigi.LocalTarget):
    def __init__(self, path):
        super(HDF5LocalTarget, self).__init__(path)

    def open(self, mode):
        if mode in ('a', 'w'):
            self.makedirs()
        return h5py.File(self.path, mode)

    def exists(self):
        return isfile(self.path)


class PrepareData(luigi.Task):
    dataset = luigi.ChoiceParameter(
        choices=['nz', 'sdrp', 'fb', 'crc'], var_type=str,
        description='Name of the dataset to use.'
    )

    def requires(self):
        return []

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__,)
        return {
            'csv': luigi.LocalTarget(join(basedir, '%s.csv' % self.dataset)),
            'pkl': luigi.LocalTarget(join(basedir, '%s.pickle' % self.dataset))
        }

    def run(self):
        data_list = datasets.load_dataset(self.dataset)

        output = self.output()
        logger.info('%d data tuples returned' % (len(data_list)))

        with output['csv'].open('w') as f:
            f.write('impath,individual,encounter,side\n')
            for img_fpath, indiv_name, enc_name, side in data_list:
                f.write('%s,%s,%s,%s\n' % (
                    img_fpath, indiv_name, enc_name, side)
                )
        with output['pkl'].open('wb') as f:
            pickle.dump(data_list, f, pickle.HIGHEST_PROTOCOL)

    def get_input_list(self):
        if not exists(self.output()['csv'].path):
            self.run()
        with self.output()['pkl'].open('rb') as f:
            return pickle.load(f)


@inherits(PrepareData)
class EncounterStats(luigi.Task):

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData)
        }

    def complete(self):
        if not exists(self.requires()['PrepareData'].output()['csv'].path):
            return False
        else:
            return all(map(
                lambda output: output.exists(),
                luigi.task.flatten(self.output())
            ))

    def output(self):
        return luigi.LocalTarget(
            join('data', self.dataset, self.__class__.__name__,
                 '%s.png' % self.dataset)
        )

    def run(self):
        import matplotlib.pyplot as plt

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        ind_enc_count_dict = {}
        for img, ind, enc, _  in input_filepaths:
            if ind not in ind_enc_count_dict:
                ind_enc_count_dict[ind] = {}
            if enc not in ind_enc_count_dict[ind]:
                ind_enc_count_dict[ind][enc] = 0
            ind_enc_count_dict[ind][enc] += 1

        individuals_to_remove = []
        for ind in ind_enc_count_dict:
            if len(ind_enc_count_dict[ind]) == 1:
                logger.info('%s has only 1 encounter' % (ind))
                individuals_to_remove.append(ind)

        for ind in individuals_to_remove:
            ind_enc_count_dict.pop(ind)

        image_counts, encounter_counts = [], []
        for ind in ind_enc_count_dict:
            for enc in ind_enc_count_dict[ind]:
                image_counts.append(ind_enc_count_dict[ind][enc])
                encounter_counts.append(len(ind_enc_count_dict[ind]))

        images_per_encounter, enc_bins = np.histogram(
            image_counts, bins=range(1, 20), density=True,
        )
        encounters_per_individual, indiv_bins = np.histogram(
            encounter_counts, bins=range(1, 20), density=True,
        )

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(22., 12))

        ax1.set_title('Number of encounters f(x) with x images')
        ax1.set_xlabel('Images')
        ax1.set_ylabel('Encounters')
        ax1.bar(enc_bins[:-1], images_per_encounter, 0.25, color='b')

        ax2.set_title('Number of individuals f(x) with x encounters')
        ax2.set_xlabel('Encounters')
        ax2.set_ylabel('Individuals')
        ax2.bar(indiv_bins[:-1], encounters_per_individual, 0.25, color='b')
        with self.output().open('wb') as f:
            plt.savefig(f, bbox_inches='tight')


@inherits(PrepareData)
class Preprocess(luigi.Task):
    height = luigi.IntParameter(
        default=256, description='Height of images after resizing.'
    )
    width = luigi.IntParameter(
        default=256, description='Width of images after resizing.'
    )

    def requires(self):
        return {'PrepareData': self.clone(PrepareData)}

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        to_process = [(fpath, side) for fpath, _, _, side in input_filepaths if
                      not exists(output[fpath]['resized'].path) or
                      not exists(output[fpath]['mask'].path) or
                      not exists(output[fpath]['transform'].path)]

        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths)))

        return to_process

    def output(self):
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'resized': luigi.LocalTarget(
                    join(basedir, 'resized', png_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', pkl_fname)),
                'mask': luigi.LocalTarget(
                    join(basedir, 'mask', png_fname)),
            }

        return outputs

    def run(self):
        from workers import preprocess_images_star

        t_start = time()
        output = self.output()
        to_process = self.get_incomplete()

        partial_preprocess_images = partial(
            preprocess_images_star,
            height=self.height,
            width=self.width,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_preprocess_images(fpath)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_preprocess_images, to_process)
        finally:
            pool.close()
            pool.join()
        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(Preprocess)
class Localization(luigi.Task):
    batch_size = luigi.IntParameter(
        default=32, description='Batch size of data passed to GPU.'
    )
    scale = luigi.IntParameter(default=4)
    no_localization = luigi.BoolParameter(
        default=False, description='Use the identity transform instead of STN.'
    )

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Preprocess': self.clone(Preprocess),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        to_process = [fpath for fpath, _, _, _ in input_filepaths if
                      not exists(output[fpath]['localization'].path) or
                      not exists(output[fpath]['mask'].path) or
                      not exists(output[fpath]['transform'].path)]
        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname =  splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'localization': luigi.LocalTarget(
                    join(basedir, 'localization', png_fname)),
                'mask': luigi.LocalTarget(
                    join(basedir, 'mask', png_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', pkl_fname)),
            }

        return outputs

    def run(self):
        import localization
        import theano_funcs

        t_start = time()
        output = self.output()
        to_process = self.get_incomplete()
        preprocess_images_targets = self.requires()['Preprocess'].output()
        if self.no_localization:
            from workers import localization_identity
            partial_localization_identity = partial(
                localization_identity,
                height=self.height,
                width=self.width,
                input_targets=preprocess_images_targets,
                output_targets=output
            )

            try:
                pool = mp.Pool(processes=None)
                pool.map(partial_localization_identity, to_process)
            finally:
                pool.close()
                pool.join()
        else:
            from workers import localization_stn
            logger.info('Building localization model')
            layers = localization.build_model(
                (None, 3, self.height, self.width), downsample=1)

            localization_weightsfile = join(
                'data', 'weights', 'weights_localization.pickle'
            )
            logger.info(
                'Loading weights for the localization network from %s' % (
                    localization_weightsfile))
            model.load_weights([
                layers['trans'], layers['loc']],
                localization_weightsfile
            )

            logger.info('Compiling theano functions for localization')
            localization_func = theano_funcs.create_localization_infer_func(
                layers)

            # we don't parallelize this function because it uses the gpu
            localization_stn(to_process,
                             self.batch_size, self.height, self.width,
                             localization_func,
                             preprocess_images_targets, output)

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(Preprocess)
@inherits(Localization)
class Refinement(luigi.Task):
    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Preprocess': self.clone(Preprocess),
            'Localization': self.clone(Localization),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        to_process = [(fpath, side) for fpath, _, _, side in input_filepaths if
                      not exists(output[fpath]['refn'].path) or
                      not exists(output[fpath]['mask'].path)]
        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname =  splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'refn': luigi.LocalTarget(
                    join(basedir, 'refn', png_fname)),
                'mask': luigi.LocalTarget(
                    join(basedir, 'mask', pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import refine_localization_star
        preprocess_images_targets = self.requires()['Preprocess'].output()
        localization_targets = self.requires()['Localization'].output()
        to_process = self.get_incomplete()
        output = self.output()

        partial_refine_localizations = partial(
            refine_localization_star,
            scale=self.scale,
            height=self.height,
            width=self.width,
            input1_targets=preprocess_images_targets,
            input2_targets=localization_targets,
            output_targets=output,
        )

        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_refine_localizations, to_process)
        finally:
            pool.close()
            pool.join()


@inherits(PrepareData)
@inherits(Refinement)
class Segmentation(luigi.Task):
    batch_size = luigi.IntParameter(
        default=32, description='Batch size of data passed to GPU.'
    )
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Refinement': self.clone(Refinement),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        to_process = []
        for fpath, _, _, _ in input_filepaths:
            seg_img_fpath = output[fpath]['segmentation-image'].path
            seg_data_fpath = output[fpath]['segmentation-data'].path
            seg_full_img_fpath = output[fpath]['segmentation-full-image'].path
            seg_full_data_fpath = output[fpath]['segmentation-full-data'].path
            if not exists(seg_img_fpath) \
                    or not exists(seg_data_fpath) \
                    or not exists(seg_full_img_fpath) \
                    or not exists(seg_full_data_fpath):
                to_process.append(fpath)

        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'segmentation-image': luigi.LocalTarget(
                    join(basedir, 'segmentation-image', png_fname)),
                'segmentation-data': luigi.LocalTarget(
                    join(basedir, 'segmentation-data', pkl_fname)),
                'segmentation-full-image': luigi.LocalTarget(
                    join(basedir, 'segmentation-full-image', png_fname)),
                'segmentation-full-data': luigi.LocalTarget(
                    join(basedir, 'segmentation-full-data', pkl_fname)),
            }

        return outputs

    def run(self):
        import imutils
        import segmentation
        import theano_funcs

        t_start = time()
        input_shape = (None, 3, self.height, self.width)

        logger.info('Building segmentation model with input shape %r' % (
            input_shape,))
        layers_segm = segmentation.build_model_batchnorm_full(input_shape)

        segmentation_weightsfile = join(
            'data', 'weights', 'weights_segmentation.pickle'
            #'data', 'weights', 'weights_humpbacks_segmentation.pickle'
        )
        logger.info('Loading weights for the segmentation network from %s' % (
            segmentation_weightsfile))
        model.load_weights(layers_segm['seg_out'], segmentation_weightsfile)

        logger.info('Compiling theano functions for segmentation')
        segm_func = theano_funcs.create_segmentation_func(layers_segm)

        output = self.output()
        refinement_targets = self.requires()['Refinement'].output()

        to_process = self.get_incomplete()
        num_batches = (
            len(to_process) + self.batch_size - 1) / self.batch_size
        logger.info('%d batches of size %d to process' % (
            num_batches, self.batch_size))
        for i in tqdm(range(num_batches), total=num_batches, leave=False):
            idx_range = range(i * self.batch_size,
                              min((i + 1) * self.batch_size, len(to_process)))
            X_batch = np.empty(
                (len(idx_range), 3, self.height, self.width), dtype=np.float32
            )
            M_batch = np.empty(
                (len(idx_range), 1,
                    self.scale * self.height, self.scale * self.width),
                dtype=np.float32
            )

            for i, idx in enumerate(idx_range):
                fpath = to_process[idx]
                img_path = refinement_targets[fpath]['refn'].path
                msk_path = refinement_targets[fpath]['mask'].path
                img = cv2.imread(img_path)

                resz = cv2.resize(img, (self.width, self.height))
                X_batch[i] = resz.transpose(2, 0, 1) / 255.
                M_batch[i, 0] = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

            S_batch  = segm_func(X_batch)
            for i, idx in enumerate(idx_range):
                fpath = to_process[idx]
                segm_img_target = output[fpath]['segmentation-image']
                segm_data_target = output[fpath]['segmentation-data']
                segm_full_img_target = output[fpath]['segmentation-full-image']
                segm_full_data_target = output[fpath]['segmentation-full-data']

                segm = S_batch[i].transpose(1, 2, 0)
                mask = M_batch[i].transpose(1, 2, 0)

                segm_refn = imutils.refine_segmentation(segm, self.scale)

                segm_refn[mask[:, :, 0] < 255] = 0.

                _, segm_buf = cv2.imencode('.png', 255. * segm)
                _, segm_refn_buf = cv2.imencode('.png', 255. * segm_refn)
                with segm_img_target.open('wb') as f1,\
                        segm_data_target.open('wb') as f2,\
                        segm_full_img_target.open('wb') as f3,\
                        segm_full_data_target.open('wb') as f4:
                    f1.write(segm_buf)
                    pickle.dump(segm, f2, pickle.HIGHEST_PROTOCOL)
                    f3.write(segm_refn_buf)
                    pickle.dump(segm_refn, f4, pickle.HIGHEST_PROTOCOL)
        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(Localization)
@inherits(Segmentation)
class Keypoints(luigi.Task):

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Localization': self.clone(Localization),
            'Segmentation': self.clone(Segmentation),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        to_process = [fpath for fpath, _, _, _ in input_filepaths if
                      not exists(output[fpath]['keypoints-visual'].path) or
                      not exists(output[fpath]['keypoints-coords'].path)]
        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'keypoints-visual': luigi.LocalTarget(
                    join(basedir, 'keypoints-visual', png_fname)),
                'keypoints-coords': luigi.LocalTarget(
                    join(basedir, 'keypoints-coords', pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import find_keypoints
        t_start = time()
        output = self.output()
        localization_targets = self.requires()['Localization'].output()
        segmentation_targets = self.requires()['Segmentation'].output()
        to_process = self.get_incomplete()

        partial_find_keypoints = partial(
            find_keypoints,
            input1_targets=localization_targets,
            input2_targets=segmentation_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_find_keypoints(fpath)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_find_keypoints, to_process)
        finally:
            pool.close()
            pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(Refinement)
@inherits(Segmentation)
@inherits(Keypoints)
class ExtractOutline(luigi.Task):

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Refinement': self.clone(Refinement),
            'Segmentation': self.clone(Segmentation),
            'Keypoints': self.clone(Keypoints),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        to_process = [fpath for fpath, _, _, _ in input_filepaths if
                      not exists(output[fpath]['outline-visual'].path) or
                      not exists(output[fpath]['outline-coords'].path)]
        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'outline-visual': luigi.LocalTarget(
                    join(basedir, 'outline-visual', png_fname)),
                'outline-coords': luigi.LocalTarget(
                    join(basedir, 'outline-coords', pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import extract_outline

        t_start = time()
        output = self.output()
        refinement_targets = self.requires()['Refinement'].output()
        segmentation_targets = self.requires()['Segmentation'].output()
        keypoints_targets = self.requires()['Keypoints'].output()
        to_process = self.get_incomplete()

        partial_extract_outline = partial(
            extract_outline,
            scale=self.scale,
            input1_targets=refinement_targets,
            input2_targets=segmentation_targets,
            input3_targets=keypoints_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_extract_outline(fpath)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_extract_outline, to_process)
        finally:
            pool.close()
            pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(Refinement)
@inherits(ExtractOutline)
class SeparateEdges(luigi.Task):

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'Refinement': self.clone(Refinement),
            'ExtractOutline': self.clone(ExtractOutline),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        to_process = [fpath for fpath, _, _, _ in input_filepaths if
                      not exists(output[fpath]['visual'].path) or
                      not exists(output[fpath]['leading-coords'].path) or
                      not exists(output[fpath]['trailing-coords'].path)]
        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'visual': luigi.LocalTarget(
                    join(basedir, 'visual', png_fname)),
                'leading-coords': luigi.LocalTarget(
                    join(basedir, 'leading-coords', pkl_fname)),
                'trailing-coords': luigi.LocalTarget(
                    join(basedir, 'trailing-coords', pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import separate_edges

        t_start = time()
        refinement_targets = self.requires()['Refinement'].output()
        extract_outline_targets = self.requires()['ExtractOutline'].output()
        output = self.output()
        to_process = self.get_incomplete()

        partial_separate_edges = partial(
            separate_edges,
            input1_targets=refinement_targets,
            input2_targets=extract_outline_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_separate_edges(fpath)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_separate_edges, to_process)
        finally:
            pool.close()
            pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(SeparateEdges)
class BlockCurvature(luigi.Task):
    serial = luigi.BoolParameter(default=False)
    trans_dims = luigi.BoolParameter(
        default=False,
        description='Transpose (x, y) -> (y, x) (use for humpback flukes).'
    )

    curv_scales = luigi.ListParameter(
        description='List providing fractions of height and/or width '
        'to use for curvature blocks/circles.'
    )

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'SeparateEdges': self.clone(SeparateEdges),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        # an image is incomplete if:
        # 1) no hdf5 file exists for it, or
        # 2) the hdf5 file exists, but some scales are missing
        to_process = []
        for fpath, _, _, _ in input_filepaths:
            target = output[fpath]['curvature']
            if target.exists():
                with target.open('r') as h5f:
                    scales_computed = h5f.keys()
                scales_to_compute = []
                for scale in self.curv_scales:
                    # only compute the missing scales
                    if '%.3f' % scale not in scales_computed:
                        scales_to_compute.append(scale)
                if scales_to_compute:
                    to_process.append((fpath, tuple(scales_to_compute)))
            else:
                to_process.append((fpath, tuple(self.curv_scales)))

        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        input_filepaths = self.requires()['PrepareData'].get_input_list()

        outputs = {}
        for fpath, indiv, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            h5py_fname = '%s.h5py' % fname
            for s in self.curv_scales:
                outputs[fpath] = {
                    'curvature': HDF5LocalTarget(
                        join(basedir, h5py_fname)),
                }

        return outputs

    def run(self):
        from workers import compute_curvature_star

        t_start = time()
        separate_edges_targets = self.requires()['SeparateEdges'].output()
        output = self.output()
        to_process = self.get_incomplete()

        partial_compute_block_curvature = partial(
            compute_curvature_star,
            transpose_dims=self.trans_dims,
            input_targets=separate_edges_targets,
            output_targets=output,
        )

        if self.serial:
            for fpath in tqdm(to_process, total=len(to_process)):
                partial_compute_block_curvature(fpath)
        else:
            try:
                pool = mp.Pool(processes=None)
                pool.map(partial_compute_block_curvature, to_process)
            finally:
                pool.close()
                pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(SeparateEdges)
class SeparateDatabaseQueries(luigi.Task):
    num_db_encounters = luigi.IntParameter(
        default=10, description='Number of encounters to use to represent '
        'each individual in the database.'
    )

    eval_dir = luigi.Parameter(
        description='The directory in which to store the splits.'
    )

    runs = luigi.IntParameter(
        description='The number of database/query splits to do.'
    )

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'SeparateEdges': self.clone(SeparateEdges),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outdir = join(
            basedir, self.eval_dir,
            '%s' % self.runs, '%s' % self.num_db_encounters
        )
        db_targets = [
            luigi.LocalTarget(join(outdir, 'db%d.pickle' % i))
            for i in range(self.runs)
        ]
        qr_targets = [
            luigi.LocalTarget(join(outdir, 'qr%d.pickle' % i))
            for i in range(self.runs)
        ]
        return {'database': db_targets, 'queries': qr_targets}

    def run(self):
        t_start = time()
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        filepaths, individuals, encounters, _ = zip(*input_filepaths)

        fname_trailing_edge_dict = {}
        trailing_edge_dict = self.requires()['SeparateEdges'].output()
        trailing_edge_filepaths = trailing_edge_dict.keys()
        logger.info('Collecting trailing edge extractions.')
        for fpath in tqdm(trailing_edge_filepaths,
                          total=len(trailing_edge_filepaths), leave=False):
            trailing_edge_target = trailing_edge_dict[fpath]['trailing-coords']
            with open(trailing_edge_target.path, 'rb') as f:
                trailing_edge = pickle.load(f)
            # no trailing edge could be extracted for this image
            if trailing_edge is None:
                continue
            else:
                fname = splitext(basename(fpath))[0]
                fname_trailing_edge_dict[fname] = fpath

        logger.info('Successful trailing edge extractions: %d of %d' % (
            len(fname_trailing_edge_dict.keys()), len(trailing_edge_filepaths))
        )
        for i in range(self.runs):
            db_dict, qr_dict = datasets.separate_database_queries(
                self.dataset, filepaths, individuals, encounters,
                fname_trailing_edge_dict,
                num_db_encounters=self.num_db_encounters
            )

            output = self.output()
            db_target = output['database'][i]
            logger.info('Saving database with %d individuals to %s' % (
                len(db_dict), db_target.path))
            with db_target.open('wb') as f:
                pickle.dump(db_dict, f, pickle.HIGHEST_PROTOCOL)
            qr_target = output['queries'][i]
            logger.info('Saving queries with %d individuals to %s' % (
                len(qr_dict), qr_target.path))
            with qr_target.open('wb') as f:
                pickle.dump(qr_dict, f, pickle.HIGHEST_PROTOCOL)

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(SeparateEdges)
class GaussDescriptors(luigi.Task):
    serial = luigi.BoolParameter(default=False)
    descriptor_m = luigi.ListParameter(default=(2, 2, 2, 2))
    descriptor_s = luigi.ListParameter(default=(1, 2, 4, 8))
    uniform = luigi.BoolParameter(default=False)
    feat_dim = luigi.IntParameter(default=16)
    contour_length = luigi.IntParameter(default=1024)
    num_keypoints = luigi.IntParameter(default=50)

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'SeparateEdges': self.clone(SeparateEdges),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        scales = zip(self.descriptor_m, self.descriptor_s)
        to_process = []
        for fpath, _, _, _ in input_filepaths:
            target = output[fpath]['descriptors']
            if target.exists():
                with target.open('r') as h5f:
                    scales_computed = h5f.keys()
                scales_to_compute = []
                for s in scales:
                    # only compute the missing scales
                    if '%s' % (s,) not in scales_computed:
                        scales_to_compute.append(s)
                if scales_to_compute:
                    to_process.append((fpath, tuple(scales_to_compute)))
            else:
                to_process.append((fpath, tuple(scales)))

        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        unifdir = 'uniform' if self.uniform else 'standard'
        featdir = '%d' % self.feat_dim

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            h5py_fname = '%s.h5py' % fname
            outputs[fpath] = {
                'descriptors': HDF5LocalTarget(
                    join(basedir, unifdir, featdir, h5py_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_gauss_descriptors_star

        t_start = time()
        separate_edges_targets = self.requires()['SeparateEdges'].output()
        output = self.output()

        to_process = self.get_incomplete()

        partial_compute_descriptors = partial(
            compute_gauss_descriptors_star,
            num_keypoints=self.num_keypoints,
            feat_dim=self.feat_dim,
            contour_length=self.contour_length,
            uniform=self.uniform,
            input_targets=separate_edges_targets,
            output_targets=output,
        )
        if self.serial:
            for fpath in tqdm(to_process, total=len(to_process)):
                partial_compute_descriptors(fpath)
        else:
            try:
                pool = mp.Pool(processes=None)
                pool.map(partial_compute_descriptors, to_process)
            finally:
                pool.close()
                pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(BlockCurvature)
class CurvatureDescriptors(luigi.Task):
    serial = luigi.BoolParameter(default=False)
    uniform = luigi.BoolParameter(default=False)
    num_keypoints = luigi.IntParameter(default=50)
    feat_dim = luigi.IntParameter(default=16)
    curv_length = luigi.IntParameter(default=1024)

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'BlockCurvature': self.clone(BlockCurvature),
        }

    def get_incomplete(self):
        output = self.output()
        input_filepaths = self.requires()['PrepareData'].get_input_list()

        to_process = []
        for fpath, _, _, _ in input_filepaths:
            target = output[fpath]['descriptors']
            if target.exists():
                with target.open('r') as h5f:
                    scales_computed = h5f.keys()
                scales_to_compute = []
                for s in self.curv_scales:
                    # only compute the missing scales
                    if '%.3f' % s not in scales_computed:
                        scales_to_compute.append(s)
                if scales_to_compute:
                    to_process.append((fpath, tuple(scales_to_compute)))
            else:
                to_process.append((fpath, tuple(self.curv_scales)))

        logger.info('%s has %d of %d images to process' % (
            self.__class__.__name__, len(to_process), len(input_filepaths))
        )

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        input_filepaths = self.requires()['PrepareData'].get_input_list()
        unifdir = 'uniform' if self.uniform else 'standard'
        featdir = '%d' % self.feat_dim

        outputs = {}
        for fpath, _, _, _ in input_filepaths:
            fname = splitext(basename(fpath))[0]
            h5py_fname = '%s.h5py' % fname
            outputs[fpath] = {
                'descriptors': HDF5LocalTarget(
                    join(basedir, unifdir, featdir, h5py_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_curv_descriptors_star

        t_start = time()
        block_curv_targets = self.requires()['BlockCurvature'].output()
        output = self.output()
        to_process = self.get_incomplete()

        partial_compute_curv_descriptors = partial(
            compute_curv_descriptors_star,
            num_keypoints=self.num_keypoints,
            feat_dim=self.feat_dim,
            curv_length=self.curv_length,
            uniform=self.uniform,
            input_targets=block_curv_targets,
            output_targets=output,
        )

        if self.serial:
            for fpath in tqdm(to_process, total=len(to_process)):
                partial_compute_curv_descriptors(fpath)
        else:
            try:
                pool = mp.Pool(processes=None)
                pool.map(partial_compute_curv_descriptors, to_process)
            finally:
                pool.close()
                pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
# TODO: this is kind of a hack, it now requires the parameters for both tasks
@inherits(CurvatureDescriptors)
@inherits(GaussDescriptors)
@inherits(SeparateDatabaseQueries)
class DescriptorsId(luigi.Task):
    k = luigi.IntParameter(default=3)
    descriptor_type = luigi.ChoiceParameter(
        choices=['gauss', 'curv'], var_type=str
    )

    def requires(self):
        if self.descriptor_type == 'gauss':
            descriptors_task = self.clone(GaussDescriptors)
        elif self.descriptor_type == 'curv':
            descriptors_task = self.clone(CurvatureDescriptors)
        return {
            'PrepareData': self.clone(PrepareData),
            'Descriptors': descriptors_task,
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
        }

    def get_incomplete(self):
        output = self.output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        qr_fpath_dict_targets = db_qr_target.output()['queries']

        to_process = defaultdict(list)
        # use the qr_dict to determine which encounters have not been quieried
        for i, qr_fpath_dict_target in enumerate(qr_fpath_dict_targets):
            with qr_fpath_dict_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            for qind in qr_fpath_dict:
                for qenc in qr_fpath_dict[qind]:
                    target = output[i][qind][qenc]
                    if not target.exists():
                        to_process[i].append((qind, qenc))

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def _get_descriptor_scales(self):
        if self.descriptor_type == 'gauss':
            descriptor_scales = [
                '%s' % (s,)
                for s in zip(self.descriptor_m, self.descriptor_s)
            ]
        elif self.descriptor_type == 'curv':
            descriptor_scales = ['%.3f' % s for s in  self.curv_scales]
        else:
            assert False, 'bad descriptor type: %s' % (self.descriptor_type)

        return descriptor_scales

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        scales = self._get_descriptor_scales()
        descdir = ','.join(['%s' % s for s in scales])
        kdir = '%d' % self.k
        unifdir = 'uniform' if self.uniform else 'standard'
        featdir = '%d' % self.feat_dim
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        output = {}
        for i in range(self.runs):
            outdir = join(
                basedir, self.eval_dir,
                self.descriptor_type, kdir, unifdir, featdir, descdir,
                '%s' % self.num_db_encounters, '%s' % i
            )
            qr_fpath_dict_target = db_qr_target.output()['queries'][i]
            if not qr_fpath_dict_target.exists():
                self.requires()['SeparateDatabaseQueries'].run()
            with qr_fpath_dict_target.open('rb') as f:
                qr_curv_dict = pickle.load(f)
            eval_dict = {}
            for qind in qr_curv_dict:
                if qind not in output:
                    eval_dict[qind] = {}
                for qenc in qr_curv_dict[qind]:
                    # an encounter may belong to multiple individuals
                    eval_dict[qind][qenc] = luigi.LocalTarget(
                        join(outdir, qind, '%s.pickle' % qenc)
                    )
            output[i] = eval_dict

        return output

    def run(self):
        import dorsal_utils
        from collections import defaultdict
        from workers import identify_encounter_descriptors_star
        from workers import build_annoy_index_star

        desc_targets = self.requires()['Descriptors'].output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        db_targets = db_qr_target.output()['database']
        qr_targets = db_qr_target.output()['queries']

        descriptor_scales = self._get_descriptor_scales()
        logger.info('Using descriptor type = %s and feature dimension = %s' % (
            self.descriptor_type, self.feat_dim))
        t_start = time()
        for run_idx, (db_target, qr_target) in enumerate(
                zip(db_targets, qr_targets)):
            with db_target.open('rb') as f:
                db_fpath_dict = pickle.load(f)
            with qr_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            db_descs_list = [len(db_fpath_dict[ind]) for ind in db_fpath_dict]
            qr_descs_list = []
            for ind in qr_fpath_dict:
                for enc in qr_fpath_dict[ind]:
                    qr_descs_list.append(len(qr_fpath_dict[ind][enc]))

            db_names_dict = defaultdict(list)
            db_descs_dict = defaultdict(list)
            logger.info('Loading descriptors for %d database individuals' % (
                len(db_fpath_dict)))

            dindivs = db_fpath_dict.keys()
            for dind in tqdm(dindivs, total=len(db_fpath_dict), leave=False):
                for fpath in db_fpath_dict[dind]:
                    target = desc_targets[fpath]['descriptors']
                    descriptors = dorsal_utils.load_descriptors_from_h5py(
                        target, descriptor_scales
                    )
                    for sidx, s in enumerate(descriptor_scales):
                        db_descs_dict[s].append(descriptors[s])
                        # label each feature with the individual name
                        for _ in range(descriptors[s].shape[0]):
                            db_names_dict[s].append(dind)

            # stack list of features per encounter into a single array
            for s in db_descs_dict:
                db_descs_dict[s] = np.vstack(db_descs_dict[s])

            # check that each descriptor is labeled with an individual name
            for s in descriptor_scales:
                num_names = len(db_names_dict[s])
                num_descs = db_descs_dict[s].shape[0]
                assert num_names == num_descs, (
                    '%d != %d' % (num_names, num_descs))

            index_fpath_dict = {
                s: join('data', 'tmp', '%s.ann') % s for s in descriptor_scales
            }
            indexes_to_build = [
                (db_descs_dict[s], index_fpath_dict[s])
                for s in descriptor_scales
            ]

            logger.info('Building %d kdtrees for scales: %s' % (
                len(descriptor_scales),
                ', '.join('%s' % s for s in descriptor_scales))
            )

            try:
                pool = mp.Pool(processes=len(indexes_to_build))
                pool.map(build_annoy_index_star, indexes_to_build)
            finally:
                pool.close()
                pool.join()

            to_process = self.get_incomplete()[run_idx]
            qindivs = qr_fpath_dict.keys()
            logger.info(
                'Running id %d of %d for %d encounters from %d individuals'
                % (
                    1 + run_idx, self.runs, len(to_process), len(qindivs)))
            logger.info(
                'Database/Queries split: (%.2f,%.2f,%.2f / %.2f,%.2f,%.2f)' % (
                    np.max(db_descs_list),
                    np.mean(db_descs_list),
                    np.min(db_descs_list),
                    np.max(qr_descs_list),
                    np.mean(qr_descs_list),
                    np.min(db_descs_list))
            )
            output = self.output()[run_idx]
            partial_identify_encounter_descriptors = partial(
                identify_encounter_descriptors_star,
                db_names=db_names_dict,
                scales=descriptor_scales,
                k=self.k,
                qr_fpath_dict=qr_fpath_dict,
                db_fpath_dict=db_fpath_dict,
                input1_targets=desc_targets,
                input2_targets=index_fpath_dict,
                output_targets=output,
            )

            if self.serial:
                for (qind, qenc) in to_process:
                    partial_identify_encounter_descriptors((qind, qenc))
            else:
                try:
                    pool = mp.Pool(processes=None)
                    pool.map(
                        partial_identify_encounter_descriptors, to_process)
                finally:
                    pool.close()
                    pool.join()
        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(BlockCurvature)
@inherits(SeparateDatabaseQueries)
class TimeWarpingId(luigi.Task):
    window = luigi.IntParameter(
        default=8, description='Sakoe-Chiba bound for time-warping alignment.'
    )
    curv_length = luigi.IntParameter(
        default=128, description='Number of spatial points in curvature '
        'vectors after resampling.'
    )
    serial = luigi.BoolParameter(default=False)
    cost_func = luigi.ChoiceParameter(
        choices=costs.get_cost_func_dict().keys(), var_type=str,
        description='Function to compute similarity of two curvature vectors.'
    )
    spatial_weights = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'BlockCurvature': self.clone(BlockCurvature),
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
        }

    def get_incomplete(self):
        output = self.output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        qr_fpath_dict_targets = db_qr_target.output()['queries']

        to_process = defaultdict(list)
        # use the qr_dict to determine which encounters have not been queried
        for i, qr_fpath_dict_target in enumerate(qr_fpath_dict_targets):
            with qr_fpath_dict_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            for qind in qr_fpath_dict:
                for qenc in qr_fpath_dict[qind]:
                    target = output[i][qind][qenc]
                    if not target.exists():
                        to_process[i].append((qind, qenc))

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curv_scales])
        curvdir = join(self.cost_func, curvdir)
        weightdir = 'weighted' if self.spatial_weights else 'uniform'

        db_qr_target = self.requires()['SeparateDatabaseQueries']
        output = {}
        for i in range(self.runs):
            outdir = join(
                basedir, self.eval_dir,
                weightdir, curvdir, '%s' % self.num_db_encounters, '%s' % i
            )
            qr_fpath_dict_target = db_qr_target.output()['queries'][i]
            if not qr_fpath_dict_target.exists():
                self.requires()['SeparateDatabaseQueries'].run()
            # query dict tells us which encounters become result objects
            with qr_fpath_dict_target.open('rb') as f:
                qr_curv_dict = pickle.load(f)
            eval_dict = {}
            for qind in qr_curv_dict:
                if qind not in output:
                    eval_dict[qind] = {}
                for qenc in qr_curv_dict[qind]:
                    # an encounter may belong to multiple individuals
                    eval_dict[qind][qenc] = luigi.LocalTarget(
                        join(outdir, qind, '%s.pickle' % qenc)
                    )

            output[i] = eval_dict

        return output

    def run(self):
        import dorsal_utils
        from scipy.interpolate import BPoly
        from workers import identify_encounter_star

        curv_targets = self.requires()['BlockCurvature'].output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        db_targets = db_qr_target.output()['database']
        qr_targets = db_qr_target.output()['queries']

        # coefficients for the sum of polynomials
        # coeffs for the SDRP Bottlenose dataset
        if self.dataset in ('sdrp', 'nz'):
            coeffs = np.array([0.0960, 0.6537, 1.0000, 0.7943, 1.0000,
                               0.3584, 0.4492, 0.0000, 0.4157, 0.0626])
        # coeffs for the CRC Humpback dataset
        elif self.dataset in ('crc', 'fb'):
            coeffs = np.array([0.0944, 0.5629, 0.7286, 0.6028, 0.0000,
                               0.0434, 0.6906, 0.7316, 0.4671, 0.0258])
        coeffs = coeffs.reshape(coeffs.shape[0], 1)
        def bernstein_poly(x, coeffs):
            interval = np.array([0, 1])
            f = BPoly(coeffs, interval, extrapolate=False)

            return f(x)

        if self.spatial_weights:
            # coefficients to weights on the interval [0, 1]
            weights = bernstein_poly(
                np.linspace(0, 1, self.curv_length), coeffs
            )
        else:
            weights = np.ones(self.curv_length, dtype=np.float32)
        weights = weights.reshape(-1, 1).astype(np.float32)

        # set the appropriate distance measure for time-warping alignment
        cost_func = costs.get_cost_func(
            self.cost_func, weights=weights, window=self.window
        )

        t_start = time()
        logger.info('Using cost function = %s and spatial weights = %s' % (
            self.cost_func, self.spatial_weights))
        for run_idx, (db_target, qr_target) in enumerate(
                zip(db_targets, qr_targets)):
            with db_target.open('rb') as f:
                db_fpath_dict = pickle.load(f)
            with qr_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            db_curv_dict = {}
            num_db_curvs = np.sum(
                [len(db_fpath_dict[dind]) for dind in db_fpath_dict]
            )
            logger.info(
                'Loading %d curv vectors for %d database individuals' %
                (num_db_curvs, len(db_fpath_dict))
            )
            for dind in tqdm(db_fpath_dict, leave=False):
                if dind not in db_curv_dict:
                    db_curv_dict[dind] = []
                for fpath in db_fpath_dict[dind]:
                    curv_matrix = dorsal_utils.load_curv_mat_from_h5py(
                        curv_targets[fpath]['curvature'],
                        self.curv_scales, self.curv_length
                    )
                    db_curv_dict[dind].append(curv_matrix)

            qr_curv_dict = {}
            num_qr_curvs = np.sum([
                len(qr_fpath_dict[qind][qenc]) for qind in qr_fpath_dict
                for qenc in qr_fpath_dict[qind]
            ])
            logger.info('Loading %d curv vectors for %d query individuals' % (
                num_qr_curvs, len(qr_fpath_dict)))
            for qind in tqdm(qr_fpath_dict, leave=False):
                if qind not in qr_curv_dict:
                    qr_curv_dict[qind] = {}
                for qenc in qr_fpath_dict[qind]:
                    if qenc not in qr_curv_dict[qind]:
                        qr_curv_dict[qind][qenc] = []
                    for fpath in qr_fpath_dict[qind][qenc]:
                        curv_matrix = dorsal_utils.load_curv_mat_from_h5py(
                            curv_targets[fpath]['curvature'],
                            self.curv_scales, self.curv_length
                        )
                        qr_curv_dict[qind][qenc].append(curv_matrix)

            db_curvs_list = [len(db_curv_dict[ind]) for ind in db_curv_dict]
            qr_curvs_list = []
            for ind in qr_curv_dict:
                for enc in qr_curv_dict[ind]:
                    qr_curvs_list.append(len(qr_curv_dict[ind][enc]))

            to_process = self.get_incomplete()[run_idx]
            qindivs = qr_curv_dict.keys()
            logger.info(
                'Running id %d of %d for %d encounters from %d individuals' % (
                    1 + run_idx, self.runs, len(to_process), len(qindivs)))
            logger.info(
                'Database/Queries split: (%.2f,%.2f,%.2f / %.2f,%.2f,%.2f)' % (
                    np.max(db_curvs_list),
                    np.mean(db_curvs_list),
                    np.min(db_curvs_list),
                    np.max(qr_curvs_list),
                    np.mean(qr_curvs_list),
                    np.min(db_curvs_list))
            )
            output = self.output()[run_idx]
            partial_identify_encounters = partial(
                identify_encounter_star,
                qr_curv_dict=qr_curv_dict,
                db_curv_dict=db_curv_dict,
                simfunc=cost_func,
                output_targets=output,
            )

            if self.serial:
                for qind, qenc in tqdm(
                        to_process, total=len(qindivs), leave=False):
                    partial_identify_encounters((qind, qenc))
            else:
                try:
                    pool = mp.Pool(processes=None)
                    pool.map(partial_identify_encounters, to_process)
                finally:
                    pool.close()
                    pool.join()

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(SeparateDatabaseQueries)
class HotSpotterId(luigi.Task):
    def requires(self):
        return {
            'PrepareData': self.clone(PrepareData),
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
        }

    def get_incomplete(self):
        output = self.output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        qr_fpath_dict_targets = db_qr_target.output()['queries']

        to_process = defaultdict(list)
        # use the qr_dict to determine which encounters have not been quieried
        for i, qr_fpath_dict_target in enumerate(qr_fpath_dict_targets):
            with qr_fpath_dict_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            for qind in qr_fpath_dict:
                for qenc in qr_fpath_dict[qind]:
                    target = output[i][qind][qenc]
                    if not target.exists():
                        to_process[i].append((qind, qenc))

        return to_process

    def complete(self):
        to_process = self.get_incomplete()
        return not bool(to_process)

    def output(self):
        basedir = join(
            '/home/hendrik/projects/dolphin-identification/data',
            self.dataset, self.__class__.__name__
        )
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        output = {}
        ibs_target = luigi.LocalTarget(join(basedir, 'ibsdb'))
        for i in range(self.runs):
            outdir = join(
                basedir, self.eval_dir,
                '%s' % self.num_db_encounters, '%s' % i
            )
            qr_fpath_dict_target = db_qr_target.output()['queries'][i]
            if not qr_fpath_dict_target.exists():
                self.requires()['SeparateDatabaseQueries'].run()
            with qr_fpath_dict_target.open('rb') as f:
                qr_curv_dict = pickle.load(f)
            eval_dict = {}
            for qind in qr_curv_dict:
                if qind not in output:
                    eval_dict[qind] = {}
                for qenc in qr_curv_dict[qind]:
                    # an encounter may belong to multiple individuals
                    eval_dict[qind][qenc] = luigi.LocalTarget(
                        join(outdir, qind, '%s.pickle' % qenc)
                    )
            output[i] = eval_dict
        output['ibs'] = ibs_target

        return output

    def run(self):
        import ibeis
        import utool as ut
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        db_targets = db_qr_target.output()['database']
        qr_targets = db_qr_target.output()['queries']

        ibs = ibeis.opendb(
            #self.output()[run_idx]['ibs'].path, allow_newdir=True
            self.output()['ibs'].path
        )
        for run_idx, (db_target, qr_target) in enumerate(
                zip(db_targets, qr_targets)):
            image_list, name_list, db_qr_list = [], [], []
            with db_target.open('rb') as f:
                db_fpath_dict = pickle.load(f)
            with qr_target.open('rb') as f:
                qr_fpath_dict = pickle.load(f)

            qr_enc_list = []
            for ind in qr_fpath_dict:
                for enc in qr_fpath_dict[ind]:
                    for fpath in qr_fpath_dict[ind][enc]:
                        image_list.append(fpath)
                        name_list.append(ind)
                        db_qr_list.append('qr')
                        qr_enc_list.append(enc)
            for ind in db_fpath_dict:
                for fpath in db_fpath_dict[ind]:
                    image_list.append(fpath)
                    name_list.append(ind)
                    db_qr_list.append('db')

            gid_list = ibs.add_images(image_list, as_annots=True)
            aid_list = ibs.get_image_aids(gid_list)
            aid_list = ut.flatten(aid_list)

            qaids, daids = [], []
            for aid, db_qr in zip(aid_list, db_qr_list):
                if db_qr == 'db':
                    daids.append(aid)
                elif db_qr == 'qr':
                    qaids.append(aid)
                else:
                    assert False, '%s' % db_qr

            ibs.set_annot_names(aid_list, name_list, notify_wildbook=False)
            ibs.set_annot_static_encounter(qaids, qr_enc_list)

            qreq = ibs.new_query_request(qaids, daids, {'sv_on': False})
            chipmatch_list = qreq.execute()
            #db_indivs = db_fpath_dict.keys()
            #qgids = ibs.get_annot_gids(qaids)
            qinds = ibs.get_annot_names(qaids)
            qencs = ibs.get_annot_static_encounter(qaids)

            qr_cm_dict = {}
            for (cm, qind, qenc) in zip(chipmatch_list, qinds, qencs):
                cm_all = cm.extend_results(qreq)
                qdf = cm_all.pandas_name_info()
                if qind not in qr_cm_dict:
                    qr_cm_dict[qind] = {}
                if qenc not in qr_cm_dict[qind]:
                    qr_cm_dict[qind][qenc] = defaultdict(list)
                ranked_nids = list(qdf['dnid'].values)
                ranked_indivs = ibs.get_name_texts(ranked_nids)
                ranked_scores = list(-1. * qdf['score'].values)
                for dind, score in zip(ranked_indivs, ranked_scores):

                    qr_cm_dict[qind][qenc][dind].append(score)

            output_target = self.output()[run_idx]
            for qind in qr_cm_dict:
                for qenc in qr_cm_dict[qind]:

                    with output_target[qind][qenc].open('wb') as f:
                        scores = qr_cm_dict[qind][qenc]
                        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)


@inherits(SeparateDatabaseQueries)
@inherits(TimeWarpingId)
class TimeWarpingResults(luigi.Task):
    serial = luigi.BoolParameter(
        default=False, description='Disable use of multiprocessing.Pool'
    )

    def requires(self):
        return {
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
            'TimeWarpingId': self.clone(TimeWarpingId),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curv_scales])
        curvdir = join(self.cost_func, curvdir)

        weightdir = 'weighted' if self.spatial_weights else 'uniform'
        outdir = join(
            basedir, self.eval_dir,
            weightdir, curvdir, '%s' % self.num_db_encounters,
        )

        all_targets = [
            luigi.LocalTarget(join(outdir, 'all%d.csv' % i))
            for i in range(self.runs)
        ]

        mrr_targets = [
            luigi.LocalTarget(join(outdir, 'mrr%d.csv' % i))
            for i in range(self.runs)
        ]

        topk_targets = [
            luigi.LocalTarget(join(outdir, 'topk%d.csv' % i))
            for i in range(self.runs)
        ]

        aggr_target = luigi.LocalTarget(join(outdir, 'aggr.csv'))

        return {
            'all': all_targets,
            'mrr': mrr_targets,
            'topk': topk_targets,
            'agg': aggr_target,
        }

    def run(self):
        from collections import defaultdict
        evaluation_targets = self.requires()['TimeWarpingId'].output()
        db_qr_output = self.requires()['SeparateDatabaseQueries'].output()
        topk_aggr = defaultdict(list)
        t_start = time()
        for run_idx in range(self.runs):
            with db_qr_output['database'][run_idx].open('rb') as f:
                db_dict = pickle.load(f)
            db_indivs = db_dict.keys()
            indiv_rank_indices = defaultdict(list)
            with self.output()['all'][run_idx].open('w') as f:
                f.write('Enc,Ind,Rank,%s\n' % (
                    ','.join('%s' % s for s in range(1, 1 + len(db_indivs))))
                )
                qind_eval_targets = evaluation_targets[run_idx]
                for qind in tqdm(qind_eval_targets, leave=False):
                    for qenc in qind_eval_targets[qind]:
                        with qind_eval_targets[qind][qenc].open('rb') as f1:
                            result_dict = pickle.load(f1)
                        scores = np.zeros(len(db_indivs), dtype=np.float32)
                        for i, dind in enumerate(db_indivs):
                            result_matrix = result_dict[dind]
                            scores[i] = result_matrix.min(axis=None)

                        asc_scores_idx = np.argsort(scores)
                        ranked_indivs = [
                            db_indivs[idx] for idx in asc_scores_idx
                        ]
                        #ranked_scores = [
                        #    scores[idx] for idx in asc_scores_idx
                        #]

                        # handle unknown individuals
                        try:
                            rank = 1 + ranked_indivs.index(qind)
                            indiv_rank_indices[qind].append(rank)
                        except ValueError:
                            rank = -1

                        f.write('"%s","%s",%s,%s\n' % (
                            qenc, qind, rank,
                            ','.join('%s' % r for r in ranked_indivs)
                        ))
                        #f.write('%s\n' % (
                        #    ','.join(['%.6f' % s for s in ranked_scores])))

            with self.output()['mrr'][run_idx].open('w') as f:
                f.write('individual,mrr\n')
                for qind in indiv_rank_indices.keys():
                    mrr = np.mean(1. / np.array(indiv_rank_indices[qind]))
                    num = len(indiv_rank_indices[qind])
                    f.write('%s (%d enc.),%.6f\n' % (qind, num, mrr))

            rank_indices = []
            for ind in indiv_rank_indices:
                for rank in indiv_rank_indices[ind]:
                    rank_indices.append(rank)

            topk_scores = [1, 5, 10, 25]
            rank_indices = np.array(rank_indices)
            num_queries = rank_indices.shape[0]
            num_indivs = len(indiv_rank_indices)
            with self.output()['topk'][run_idx].open('w') as f:
                f.write('topk,accuracy\n')
                for k in range(1, 1 + num_indivs):
                    topk = (100. / num_queries) * (rank_indices <= k).sum()
                    topk_aggr[run_idx].append(topk)
                    f.write('top-%d,%.6f\n' % (k, topk))

        aggr = np.vstack([topk_aggr[i] for i in range(self.runs)]).T
        with self.output()['agg'].open('w') as f:
            logger.info('Accuracy scores over %d runs for k = %s:' % (
                self.runs, ', '.join(['%d' % k for k in topk_scores])))
            f.write('mean,min,max,std\n')
            for i, k in enumerate(range(1, 1 + num_indivs)):
                f.write('%.6f,%.6f,%.6f,%.6f\n' % (
                    aggr[i].mean(), aggr[i].min(), aggr[i].max(), aggr[i].std()
                ))
                if k in topk_scores:
                    logger.info(' top-%d: %.2f%%' % (k, aggr[i].mean()))

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(SeparateDatabaseQueries)
@inherits(DescriptorsId)
class DescriptorsResults(luigi.Task):
    serial = luigi.BoolParameter(
        default=False, description='Disable use of multiprocessing.Pool'
    )

    def requires(self):
        return {
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
            'DescriptorsId': self.clone(DescriptorsId),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        scales = self.requires()['DescriptorsId']._get_descriptor_scales()
        descdir = ','.join(['%s' % s for s in scales])
        kdir = '%d' % self.k
        unifdir = 'uniform' if self.uniform else 'standard'
        featdir = '%d' % self.feat_dim
        outdir = join(
            basedir, self.eval_dir,
            self.descriptor_type, kdir, unifdir, featdir, descdir,
            '%s' % self.num_db_encounters
        )

        all_targets = [
            luigi.LocalTarget(join(outdir, 'all%d.csv' % i))
            for i in range(self.runs)
        ]

        mrr_targets = [
            luigi.LocalTarget(join(outdir, 'mrr%d.csv' % i))
            for i in range(self.runs)
        ]

        topk_targets = [
            luigi.LocalTarget(join(outdir, 'topk%d.csv' % i))
            for i in range(self.runs)
        ]

        aggr_target = luigi.LocalTarget(join(outdir, 'aggr.csv'))

        return {
            'all': all_targets,
            'mrr': mrr_targets,
            'topk': topk_targets,
            'agg': aggr_target,
        }

    def run(self):
        from collections import defaultdict
        evaluation_targets = self.requires()['DescriptorsId'].output()
        db_qr_output = self.requires()['SeparateDatabaseQueries'].output()
        topk_aggr = defaultdict(list)
        t_start = time()
        for run_idx in range(self.runs):
            with db_qr_output['database'][run_idx].open('rb') as f:
                db_dict = pickle.load(f)
            db_indivs = db_dict.keys()
            indiv_rank_indices = defaultdict(list)
            with self.output()['all'][run_idx].open('w') as f:
                f.write('Enc,Ind,Rank,%s\n' % (
                    ','.join('%s' % s for s in range(1, 1 + len(db_indivs))))
                )
                qind_eval_targets = evaluation_targets[run_idx]
                for qind in tqdm(qind_eval_targets, leave=False):
                    for qenc in qind_eval_targets[qind]:
                        with qind_eval_targets[qind][qenc].open('rb') as f1:
                            result_dict = pickle.load(f1)
                        scores = np.zeros(len(db_indivs), dtype=np.float32)
                        for i, dind in enumerate(db_indivs):
                            result_matrix = result_dict[dind]
                            scores[i] = result_matrix

                        asc_scores_idx = np.argsort(scores)
                        ranked_indivs = [
                            db_indivs[idx] for idx in asc_scores_idx
                        ]
                        #ranked_scores = [
                        #    scores[idx] for idx in asc_scores_idx]
                        #]

                        # handle unknown individuals
                        try:
                            rank = 1 + ranked_indivs.index(qind)
                            indiv_rank_indices[qind].append(rank)
                        except ValueError:
                            rank = -1

                        f.write('"%s","%s",%s,%s\n' % (
                            qenc, qind, rank,
                            ','.join('%s' % r for r in ranked_indivs)
                        ))
                        #f.write('%s\n' % (
                        #    ','.join(['%.6f' % s for s in ranked_scores])))

            with self.output()['mrr'][run_idx].open('w') as f:
                f.write('individual,mrr\n')
                for qind in indiv_rank_indices.keys():
                    mrr = np.mean(1. / np.array(indiv_rank_indices[qind]))
                    num = len(indiv_rank_indices[qind])
                    f.write('%s (%d enc.),%.6f\n' % (qind, num, mrr))

            rank_indices = []
            for ind in indiv_rank_indices:
                for rank in indiv_rank_indices[ind]:
                    rank_indices.append(rank)

            topk_scores = [1, 5, 10, 25]
            rank_indices = np.array(rank_indices)
            num_queries = rank_indices.shape[0]
            num_indivs = len(indiv_rank_indices)
            with self.output()['topk'][run_idx].open('w') as f:
                f.write('topk,accuracy\n')
                for k in range(1, 1 + num_indivs):
                    topk = (100. / num_queries) * (rank_indices <= k).sum()
                    topk_aggr[run_idx].append(topk)
                    f.write('top-%d,%.6f\n' % (k, topk))

        aggr = np.vstack([topk_aggr[i] for i in range(self.runs)]).T
        with self.output()['agg'].open('w') as f:
            logger.info('Accuracy scores over %d runs for k = %s:' % (
                self.runs, ', '.join(['%d' % k for k in topk_scores])))
            f.write('mean,min,max,std\n')
            for i, k in enumerate(range(1, 1 + num_indivs)):
                f.write('%.6f,%.6f,%.6f,%.6f\n' % (
                    aggr[i].mean(), aggr[i].min(), aggr[i].max(), aggr[i].std()
                ))
                if k in topk_scores:
                    logger.info(' top-%d: %.2f%%' % (k, aggr[i].mean()))

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(SeparateDatabaseQueries)
@inherits(HotSpotterId)
class HotSpotterResults(luigi.Task):
    serial = luigi.BoolParameter(
        default=False, description='Disable use of multiprocessing.Pool'
    )

    def requires(self):
        return {
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
            'HotSpotterId': self.clone(HotSpotterId),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outdir = join(
            basedir, self.eval_dir,
            '%s' % self.num_db_encounters
        )

        all_targets = [
            luigi.LocalTarget(join(outdir, 'all%d.csv' % i))
            for i in range(self.runs)
        ]

        mrr_targets = [
            luigi.LocalTarget(join(outdir, 'mrr%d.csv' % i))
            for i in range(self.runs)
        ]

        topk_targets = [
            luigi.LocalTarget(join(outdir, 'topk%d.csv' % i))
            for i in range(self.runs)
        ]

        aggr_target = luigi.LocalTarget(join(outdir, 'aggr.csv'))

        return {
            'all': all_targets,
            'mrr': mrr_targets,
            'topk': topk_targets,
            'agg': aggr_target,
        }

    def run(self):
        from collections import defaultdict
        evaluation_targets = self.requires()['HotSpotterId'].output()
        db_qr_output = self.requires()['SeparateDatabaseQueries'].output()
        topk_aggr = defaultdict(list)
        t_start = time()
        for run_idx in range(self.runs):
            with db_qr_output['database'][run_idx].open('rb') as f:
                db_dict = pickle.load(f)
            db_indivs = db_dict.keys()
            indiv_rank_indices = defaultdict(list)
            with self.output()['all'][run_idx].open('w') as f:
                f.write('Enc,Ind,Rank,%s\n' % (
                    ','.join('%s' % s for s in range(1, 1 + len(db_indivs))))
                )
                qind_eval_targets = evaluation_targets[run_idx]
                for qind in tqdm(qind_eval_targets, leave=False):
                    for qenc in qind_eval_targets[qind]:
                        with qind_eval_targets[qind][qenc].open('rb') as f1:
                            result_dict = pickle.load(f1)
                        scores = np.zeros(len(db_indivs), dtype=np.float32)
                        for i, dind in enumerate(db_indivs):
                            result_matrix = np.min(result_dict[dind])
                            scores[i] = result_matrix

                        asc_scores_idx = np.argsort(scores)
                        ranked_indivs = [
                            db_indivs[idx] for idx in asc_scores_idx
                        ]
                        #ranked_scores = [
                        #    scores[idx] for idx in asc_scores_idx]
                        #]

                        # handle unknown individuals
                        try:
                            rank = 1 + ranked_indivs.index(qind)
                            indiv_rank_indices[qind].append(rank)
                        except ValueError:
                            rank = -1

                        f.write('"%s","%s",%s,%s\n' % (
                            qenc, qind, rank,
                            ','.join('%s' % r for r in ranked_indivs)
                        ))
                        #f.write('%s\n' % (
                        #    ','.join(['%.6f' % s for s in ranked_scores])))

            with self.output()['mrr'][run_idx].open('w') as f:
                f.write('individual,mrr\n')
                for qind in indiv_rank_indices.keys():
                    mrr = np.mean(1. / np.array(indiv_rank_indices[qind]))
                    num = len(indiv_rank_indices[qind])
                    f.write('%s (%d enc.),%.6f\n' % (qind, num, mrr))

            rank_indices = []
            for ind in indiv_rank_indices:
                for rank in indiv_rank_indices[ind]:
                    rank_indices.append(rank)

            topk_scores = [1, 5, 10, 25]
            rank_indices = np.array(rank_indices)
            num_queries = rank_indices.shape[0]
            num_indivs = len(indiv_rank_indices)
            with self.output()['topk'][run_idx].open('w') as f:
                f.write('topk,accuracy\n')
                for k in range(1, 1 + num_indivs):
                    topk = (100. / num_queries) * (rank_indices <= k).sum()
                    topk_aggr[run_idx].append(topk)
                    f.write('top-%d,%.6f\n' % (k, topk))

        aggr = np.vstack([topk_aggr[i] for i in range(self.runs)]).T
        with self.output()['agg'].open('w') as f:
            logger.info('Accuracy scores over %d runs for k = %s:' % (
                self.runs, ', '.join(['%d' % k for k in topk_scores])))
            f.write('mean,min,max,std\n')
            for i, k in enumerate(range(1, 1 + num_indivs)):
                f.write('%.6f,%.6f,%.6f,%.6f\n' % (
                    aggr[i].mean(), aggr[i].min(), aggr[i].max(), aggr[i].std()
                ))
                if k in topk_scores:
                    logger.info(' top-%d: %.2f%%' % (k, aggr[i].mean()))

        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


class VisualizeIndividuals(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [PrepareData(dataset=self.dataset),
                SeparateEdges(dataset=self.dataset,
                              imsize=self.imsize,
                              batch_size=self.batch_size,
                              scale=self.scale)]

    def output(self):
        csv_fpath = self.requires()[0].output().path
        # hack for when the csv file doesn't exist
        if not exists(csv_fpath):
            self.requires()[0].run()
        df = pd.read_csv(
            csv_fpath, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )
        basedir = join('data', self.dataset, self.__class__.__name__)
        image_filepaths = df['impath'].values
        individuals = df['individual'].values

        outputs = {}
        for (indiv, fpath) in zip(individuals, image_filepaths):
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            outputs[fpath] = {
                'image': luigi.LocalTarget(
                    join(basedir, indiv, png_fname)),
            }

        return outputs

    def run(self):
        from workers import visualize_individuals
        output = self.output()

        separate_edges_targets = self.requires()[1].output()
        image_filepaths = separate_edges_targets.keys()

        to_process = [fpath for fpath, _, _, _ in image_filepaths if
                      not exists(output[fpath]['image'].path)]

        logger.info('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        partial_visualize_individuals = partial(
            visualize_individuals,
            input_targets=separate_edges_targets,
            output_targets=output
        )

        t_start = time()
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_visualize_individuals(fpath)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_visualize_individuals, to_process)
        finally:
            pool.close()
            pool.join()
        t_end = time()
        logger.info('%s completed in %.3fs' % (
            self.__class__.__name__, t_end - t_start))


@inherits(PrepareData)
@inherits(SeparateEdges)
@inherits(SeparateDatabaseQueries)
@inherits(BlockCurvature)
@inherits(TimeWarpingId)
class VisualizeMisidentifications(luigi.Task):
    num_qr_visualizations = luigi.IntParameter(default=3)
    num_db_visualizations = luigi.IntParameter(default=5)

    def requires(self):
        return {
            'SeparateEdges': self.clone(SeparateEdges),
            'SeparateDatabaseQueries': self.clone(SeparateDatabaseQueries),
            'BlockCurvature': self.clone(BlockCurvature),
            'TimeWarpingId': self.clone(TimeWarpingId),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curv_scales])

        output = {}
        evaluation_targets = self.requires()['TimeWarpingId'].output()
        for qind in evaluation_targets:
            if qind not in output:
                output[qind] = {}
            for qenc in evaluation_targets[qind]:
                # an encounter may belong to multiple individuals, hence qind
                output[qind][qenc] = {
                    'separate-edges': luigi.LocalTarget(
                        join(basedir, curvdir, qind, '%s_edges.png' % qenc)),
                    'curvature': luigi.LocalTarget(
                        join(basedir, curvdir, qind, '%s_curvs.png' % qenc)),
                }

        return output

    def run(self):
        from workers import visualize_misidentifications
        output = self.output()
        edges_targets = self.requires()['SeparateEdges'].output()
        db_qr_targets = self.requires()['SeparateDatabaseQueries'].output()
        block_curv_targets = self.requires()['BlockCurvature'].output()
        with db_qr_targets['database'].open('rb') as f:
            db_dict = pickle.load(f)
        with db_qr_targets['queries'].open('rb') as f:
            qr_dict = pickle.load(f)

        evaluation_targets = self.requires()['TimeWarpingId'].output()
        qindivs = evaluation_targets.keys()
        partial_visualize_misidentifications = partial(
            visualize_misidentifications,
            qr_dict=qr_dict,
            db_dict=db_dict,
            num_qr=self.num_qr_visualizations,
            num_db=self.num_db_visualizations,
            scales=self.curv_scales,
            curv_length=self.curv_length,
            input1_targets=evaluation_targets,
            input2_targets=edges_targets,
            input3_targets=block_curv_targets,
            output_targets=output,
        )

        #for qind in qindivs:
        #    partial_visualize_misidentifications(qind)
        try:
            pool = mp.Pool(processes=None)
            pool.map(partial_visualize_misidentifications, qindivs)
        finally:
            pool.close()
            pool.join()


if __name__ == '__main__':
    luigi.run()
