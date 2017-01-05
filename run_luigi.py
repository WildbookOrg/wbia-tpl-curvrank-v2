import cPickle as pickle
import cv2
import pandas as pd
import luigi
import datasets
import model
import multiprocessing as mp
import numpy as np

from functools import partial
from tqdm import tqdm
from os.path import basename, exists, join, splitext


class PrepareData(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(
            join('data', self.dataset, self.__class__.__name__,
                 '%s.csv' % self.dataset)
        )

    def run(self):
        data_list = datasets.load_dataset(self.dataset)

        print('%d data tuples returned' % (len(data_list)))

        with self.output().open('w') as f:
            f.write('impath,individual,encounter\n')
            for img_fpath, indiv_name, enc_name in data_list:
                f.write('%s,%s,%s\n' % (img_fpath, indiv_name, enc_name))


class EncounterStats(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)

    def requires(self):
        return [PrepareData(dataset=self.dataset)]

    def complete(self):
        if not exists(self.requires()[0].output().path):
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
        csv_fpath = self.requires()[0].output().path
        # hack for when the csv file doesn't exist
        if not exists(csv_fpath):
            self.requires()[0].run()
        df = pd.read_csv(
            csv_fpath, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )

        ind_enc_count_dict = {}
        for img, ind, enc in df.values:
            if ind not in ind_enc_count_dict:
                ind_enc_count_dict[ind] = {}
            if enc not in ind_enc_count_dict[ind]:
                ind_enc_count_dict[ind][enc] = 0
            ind_enc_count_dict[ind][enc] += 1

        individuals_to_remove = []
        for ind in ind_enc_count_dict:
            if len(ind_enc_count_dict[ind]) == 1:
                print('%s has only 1 encounter' % (ind))
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


class PreprocessImages(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)

    def requires(self):
        return [PrepareData(dataset=self.dataset)]

    def complete(self):
        if not exists(self.requires()[0].output().path):
            return False
        else:
            return all(map(
                lambda output: output.exists(),
                luigi.task.flatten(self.output())
            ))

    def output(self):
        csv_fpath = self.requires()[0].output().path
        # hack for when the csv file doesn't exist
        if not exists(csv_fpath):
            self.requires()[0].run()
        df = pd.read_csv(
            csv_fpath, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )
        image_filepaths = df['impath'].values

        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in image_filepaths:
            fname = splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'resized': luigi.LocalTarget(
                    join(basedir, 'resized', png_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import preprocess_images

        output = self.output()
        image_filepaths = output.keys()

        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['resized'].path) or
                      not exists(output[fpath]['transform'].path)]

        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))
        partial_preprocess_images = partial(
            preprocess_images,
            imsize=self.imsize,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_preprocess_images(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_preprocess_images, to_process)


class Localization(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [PreprocessImages(dataset=self.dataset, imsize=self.imsize)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
            fname =  splitext(basename(fpath))[0]
            png_fname = '%s.png' % fname
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'localization': luigi.LocalTarget(
                    join(basedir, 'localization', png_fname)),
                'localization-full': luigi.LocalTarget(
                    join(basedir, 'localization-full', png_fname)),
                'mask': luigi.LocalTarget(
                    join(basedir, 'mask', pkl_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', pkl_fname)),
            }

        return outputs

    def run(self):
        import imutils
        import localization
        import theano_funcs
        height, width = 256, 256

        print('building localization model')
        layers = localization.build_model(
            (None, 3, height, width), downsample=1)

        localization_weightsfile = join(
            'data', 'weights', 'weights_localization.pickle'
        )
        print('loading weights for the localization network from %s' % (
            localization_weightsfile))
        model.load_weights([
            layers['trans'], layers['loc']],
            localization_weightsfile
        )

        print('compiling theano functions for localization')
        localization_func = theano_funcs.create_localization_infer_func(layers)

        output = self.output()
        preprocess_images_targets = self.requires()[0].output()
        image_filepaths = preprocess_images_targets.keys()

        # we don't parallelize this function because it uses the gpu
        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['localization'].path) or
                      not exists(output[fpath]['localization-full'].path) or
                      not exists(output[fpath]['mask'].path) or
                      not exists(output[fpath]['transform'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        num_batches = (
            len(to_process) + self.batch_size - 1) / self.batch_size
        print('%d batches to process' % (num_batches))
        for i in tqdm(range(num_batches), total=num_batches, leave=False):
            idx_range = range(i * self.batch_size,
                              min((i + 1) * self.batch_size, len(to_process)))
            X_batch = np.empty(
                (len(idx_range), 3, height, width), dtype=np.float32
            )
            trns_batch = np.empty(
                (len(idx_range), 3, 3), dtype=np.float32
            )

            for i, idx in enumerate(idx_range):
                fpath = to_process[idx]
                impath = preprocess_images_targets[fpath]['resized'].path
                img = cv2.imread(impath)
                tpath = preprocess_images_targets[fpath]['transform'].path
                with open(tpath, 'rb') as f:
                    trns_batch[i] = pickle.load(f)

                X_batch[i] = img.transpose(2, 0, 1) / 255.

            L_batch_loc, X_batch_loc = localization_func(X_batch)
            for i, idx in enumerate(idx_range):
                fpath = to_process[idx]
                loc_lr_target = output[fpath]['localization']
                loc_hr_target = output[fpath]['localization-full']
                mask_target = output[fpath]['mask']
                trns_target = output[fpath]['transform']

                prep_trns = trns_batch[i]
                lclz_trns = np.vstack((
                    L_batch_loc[i].reshape((2, 3)), np.array([0, 0, 1])
                ))

                img_loc_lr = (255. * X_batch_loc[i]).astype(
                    np.uint8).transpose(1, 2, 0)
                img_orig = cv2.imread(fpath)
                # don't need to store the mask, reconstruct it here
                msk_orig = np.ones_like(img_orig).astype(np.float32)
                img_loc_hr, mask_loc_hr = imutils.refine_localization(
                    img_orig, msk_orig, prep_trns, lclz_trns,
                    self.scale, self.imsize
                )

                _, img_loc_lr_buf = cv2.imencode('.png', img_loc_lr)
                _, img_loc_hr_buf = cv2.imencode('.png', img_loc_hr)

                with loc_lr_target.open('wb') as f1,\
                        loc_hr_target.open('wb') as f2,\
                        mask_target.open('wb') as f3,\
                        trns_target.open('wb') as f4:
                    f1.write(img_loc_lr_buf)
                    f2.write(img_loc_hr_buf)
                    pickle.dump(mask_loc_hr, f3, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(lclz_trns, f4, pickle.HIGHEST_PROTOCOL)


class Segmentation(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [Localization(dataset=self.dataset,
                             imsize=self.imsize,
                             batch_size=self.batch_size,)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
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

        height, width = 256, 256
        input_shape = (None, 3, height, width)

        print('building segmentation model with input shape %r' % (
            input_shape,))
        layers_segm = segmentation.build_model_batchnorm_full(input_shape)

        segmentation_weightsfile = join(
            'data', 'weights', 'weights_segmentation.pickle'
        )
        print('loading weights for the segmentation network from %s' % (
            segmentation_weightsfile))
        model.load_weights(layers_segm['seg_out'], segmentation_weightsfile)

        print('compiling theano functions for segmentation')
        segm_func = theano_funcs.create_segmentation_func(layers_segm)

        output = self.output()
        localization_targets = self.requires()[0].output()
        image_filepaths = localization_targets.keys()

        to_process = []
        for fpath in image_filepaths:
            seg_img_fpath = output[fpath]['segmentation-image'].path
            seg_data_fpath = output[fpath]['segmentation-data'].path
            seg_full_img_fpath = output[fpath]['segmentation-full-image'].path
            seg_full_data_fpath = output[fpath]['segmentation-full-data'].path
            if not exists(seg_img_fpath) \
                    or not exists(seg_data_fpath) \
                    or not exists(seg_full_img_fpath) \
                    or not exists(seg_full_data_fpath):
                to_process.append(fpath)

        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        num_batches = (
            len(to_process) + self.batch_size - 1) / self.batch_size
        print('%d batches to process' % (num_batches))
        for i in tqdm(range(num_batches), total=num_batches, leave=False):
            idx_range = range(i * self.batch_size,
                              min((i + 1) * self.batch_size, len(to_process)))
            X_batch = np.empty(
                (len(idx_range), 3, height, width), dtype=np.float32
            )
            M_batch = np.empty(
                (len(idx_range), 3, self.scale * height, self.scale * width),
                dtype=np.float32
            )

            for i, idx in enumerate(idx_range):
                fpath = to_process[idx]
                img_path =\
                    localization_targets[fpath]['localization-full'].path
                msk_path = localization_targets[fpath]['mask'].path
                img = cv2.imread(img_path)

                resz = cv2.resize(img, (height, width))
                X_batch[i] = resz.transpose(2, 0, 1) / 255.
                with open(msk_path, 'rb') as f:
                    M_batch[i] = pickle.load(f).transpose(2, 0, 1)

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

                segm_refn[mask[:, :, 0] < 1] = 0.

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


class FindKeypoints(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)

    def requires(self):
        return [Localization(dataset=self.dataset,
                             imsize=self.imsize,
                             batch_size=self.batch_size),
                Segmentation(dataset=self.dataset,
                             imsize=self.imsize,
                             batch_size=self.batch_size)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
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
        output = self.output()
        localization_targets = self.requires()[0].output()
        segmentation_targets = self.requires()[1].output()
        image_filepaths = segmentation_targets.keys()
        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['keypoints-visual'].path) or
                      not exists(output[fpath]['keypoints-coords'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        partial_find_keypoints = partial(
            find_keypoints,
            input1_targets=localization_targets,
            input2_targets=segmentation_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(image_filepaths)):
        #    partial_find_keypoints(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_find_keypoints, to_process)


class ExtractOutline(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [
            Localization(dataset=self.dataset,
                         imsize=self.imsize,
                         batch_size=self.batch_size,
                         scale=self.scale),
            Segmentation(dataset=self.dataset,
                         imsize=self.imsize,
                         batch_size=self.batch_size,
                         scale=self.scale),
            FindKeypoints(dataset=self.dataset,
                          imsize=self.imsize,
                          batch_size=self.batch_size),
        ]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
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
        output = self.output()
        localization_targets = self.requires()[0].output()
        segmentation_targets = self.requires()[1].output()
        keypoints_targets = self.requires()[2].output()
        image_filepaths = segmentation_targets.keys()
        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['outline-visual'].path) or
                      not exists(output[fpath]['outline-coords'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        partial_extract_outline = partial(
            extract_outline,
            scale=self.scale,
            input1_targets=localization_targets,
            input2_targets=segmentation_targets,
            input3_targets=keypoints_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(image_filepaths)):
        #    partial_extract_outline(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_extract_outline, to_process)


class SeparateEdges(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [Localization(dataset=self.dataset,
                             imsize=self.imsize,
                             batch_size=self.batch_size,
                             scale=self.scale),
                ExtractOutline(dataset=self.dataset,
                               imsize=self.imsize,
                               batch_size=self.batch_size,
                               scale=self.scale)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
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
        localization_targets = self.requires()[0].output()
        extract_outline_targets = self.requires()[1].output()
        output = self.output()
        input_filepaths = extract_outline_targets.keys()

        to_process = [fpath for fpath in input_filepaths if
                      not exists(output[fpath]['visual'].path) or
                      not exists(output[fpath]['leading-coords'].path) or
                      not exists(output[fpath]['trailing-coords'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(input_filepaths)))

        partial_separate_edges = partial(
            separate_edges,
            input1_targets=localization_targets,
            input2_targets=extract_outline_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_separate_edges(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_separate_edges, to_process)


class ComputeBlockCurvature(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    oriented = luigi.BoolParameter(default=False)
    if oriented:  # use oriented curvature
        curvature_scales = luigi.ListParameter(
            default=(0.06, 0.10, 0.14, 0.18)
        )
    else:       # use standard block curvature
        curvature_scales = luigi.ListParameter(
            default=(0.133, 0.207, 0.280, 0.353)
        )

    def requires(self):
        return [SeparateEdges(dataset=self.dataset,
                              imsize=self.imsize,
                              batch_size=self.batch_size,
                              scale=self.scale)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curvature_scales])
        if self.oriented:
            curvdir = join('oriented', curvdir)
        else:
            curvdir = join('standard', curvdir)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
            fname = splitext(basename(fpath))[0]
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'curvature': luigi.LocalTarget(
                    join(basedir, curvdir, pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_block_curvature
        separate_edges_targets = self.requires()[0].output()
        output = self.output()
        input_filepaths = separate_edges_targets.keys()

        to_process = [fpath for fpath in input_filepaths if
                      not exists(output[fpath]['curvature'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(input_filepaths)))

        partial_compute_block_curvature = partial(
            compute_block_curvature,
            scales=self.curvature_scales,
            oriented=self.oriented,
            input_targets=separate_edges_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_compute_block_curvature(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_compute_block_curvature, to_process)


class ComputeDescriptors(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [SeparateEdges(dataset=self.dataset,
                              imsize=self.imsize,
                              batch_size=self.batch_size,
                              scale=self.scale)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)

        outputs = {}
        for fpath in self.requires()[0].output().keys():
            fname = splitext(basename(fpath))[0]
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'descriptors': luigi.LocalTarget(
                    join(basedir, pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_descriptors
        separate_edges_targets = self.requires()[0].output()
        output = self.output()
        input_filepaths = separate_edges_targets.keys()

        to_process = [fpath for fpath in input_filepaths if
                      not exists(output[fpath]['descriptors'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(input_filepaths)))

        partial_compute_descriptors = partial(
            compute_descriptors,
            scales=[(2, 1), (2, 2), (2, 4), (2, 8)],
            input_targets=separate_edges_targets,
            output_targets=output,
        )
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_compute_descriptors(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_compute_descriptors, to_process)


class EvaluateDescriptors(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    k = luigi.IntParameter(default=3)

    def requires(self):
        return [
            PrepareData(dataset=self.dataset),
            ComputeDescriptors(dataset=self.dataset),
        ]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        return [
            luigi.LocalTarget(
                join(basedir, '%s_all.csv' % self.dataset)),
            luigi.LocalTarget(
                join(basedir, '%s_mrr.csv' % self.dataset)),
            luigi.LocalTarget(
                join(basedir, '%s_topk.csv' % self.dataset))
        ]

    def run(self):
        import operator
        import pyflann
        from collections import defaultdict
        df = pd.read_csv(
            self.requires()[0].output().path, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )

        fname_desc_dict = {}
        desc_dict = self.requires()[1].output()
        desc_filepaths = desc_dict.keys()

        print('loading descriptor vectors of dimension %d for %d images' % (
            256, len(desc_filepaths)))

        for fpath in tqdm(desc_filepaths,
                          total=len(desc_filepaths), leave=False):
            fname = splitext(basename(fpath))[0]
            fname_desc_dict[fname] = desc_dict[fpath]['descriptors']  # TODO

        db_dict, qr_dict = datasets.separate_database_queries(
            self.dataset, df['impath'].values,
            df['individual'].values, df['encounter'].values,
            fname_desc_dict
        )

        db_descs_list = [len(db_dict[ind]) for ind in db_dict]
        qr_descs_list = []
        for ind in qr_dict:
            for enc in qr_dict[ind]:
                qr_descs_list.append(len(qr_dict[ind][enc]))

        print('max/mean/min images per db encounter: %.2f/%.2f/%.2f' % (
            np.max(db_descs_list),
            np.mean(db_descs_list),
            np.min(db_descs_list))
        )
        print('max/mean/min images per qr encounter: %.2f/%.2f/%.2f' % (
            np.max(qr_descs_list),
            np.mean(qr_descs_list),
            np.min(qr_descs_list))
        )

        db_labels = []
        db1, db2, db3, db4 = [], [], [], []
        print('loading descriptors for %d database individuals' % (
            len(db_dict)))
        for dind in tqdm(db_dict, total=len(db_dict), leave=False):
            for target in db_dict[dind]:
                with target.open('rb') as f:
                    desc = pickle.load(f)
                if desc is None:
                    continue
                for db, d in zip([db1, db2, db3, db4], desc):
                    db.append(d)
                for _ in range(desc[0].shape[0]):
                    db_labels.append(dind)

        db1 = np.vstack(db1)
        db2 = np.vstack(db2)
        db3 = np.vstack(db3)
        db4 = np.vstack(db4)
        dbl = np.hstack(db_labels)

        flann_list, params_list = [], []
        db_list = [db1, db2, db3, db4]
        for db in db_list:
            flann_list.append(pyflann.FLANN())
        print('building kdtrees')
        for db, flann in tqdm(
                zip(db_list, flann_list), total=len(flann_list), leave=False):
            params_list.append(flann.build_index(db))

        indiv_rank_indices = defaultdict(list)
        qindivs = qr_dict.keys()
        with self.output()[0].open('w') as f:
            print('running identification for %d individuals' % (len(qindivs)))
            for qind in tqdm(qindivs, total=len(qindivs), leave=False):
                qencs = qr_dict[qind].keys()
                assert qencs, 'empty encounter list for %s' % qind
                for qenc in qencs:
                    q1, q2, q3, q4 = [], [], [], []
                    for target in qr_dict[qind][qenc]:
                        with target.open('rb') as dfile:
                            desc = pickle.load(dfile)
                        if desc is None:
                            continue
                        for q, d in zip([q1, q2, q3, q4], desc):
                            q.append(d)

                    if not q1:
                        #print('no descriptors for %s: %s' % (qind, qenc))
                        continue
                    q1 = np.vstack(q1)
                    q2 = np.vstack(q2)
                    q3 = np.vstack(q3)
                    q4 = np.vstack(q4)

                    q_list = [q1, q2, q3, q4]
                    for flann, params, q in zip(
                            flann_list, params_list, q_list):
                        ind, dist = flann.nn_index(
                            q, self.k, checks=params['checks']
                        )

                        scores = defaultdict(int)
                        for i in range(q.shape[0]):
                            classes = dbl[ind][i, :]
                            for c in np.unique(classes):
                                j, = np.where(classes == c)
                                score = dist[i, j.min()] - dist[i, -1]
                                scores[c] += score

                    ranking = sorted(scores.items(),
                                     key=operator.itemgetter(1))
                    rindivs = [x[0] for x in ranking]
                    scores = [x[1] for x in ranking]

                    try:
                        rank = 1 + rindivs.index(qind)
                    except ValueError:
                        rank = 1 + (len(rindivs) + len(qindivs)) / 2

                    indiv_rank_indices[qind].append(rank)

                    f.write('%s,%s\n' % (
                        qind, ','.join(['%s' % r for r in rindivs])))
                    f.write('%s\n' % (
                        ','.join(['%.6f' % s for s in scores])))

        with self.output()[1].open('w') as f:
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
        print('accuracy scores:')
        with self.output()[2].open('w') as f:
            f.write('topk,accuracy\n')
            for k in range(1, 1 + num_indivs):
                topk = (100. / num_queries) * (rank_indices <= k).sum()
                f.write('top-%d,%.6f\n' % (k, topk))
                if k in topk_scores:
                    print(' top-%d: %.2f%%' % (k, topk))


class EvaluateIdentification(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    window = luigi.IntParameter(default=8)
    curv_length = luigi.IntParameter(default=128)
    oriented = luigi.BoolParameter(default=False)

    if oriented:  # use oriented curvature
        curvature_scales = luigi.ListParameter(
            #default=(0.06, 0.10, 0.14, 0.18)
            default=(0.110, 0.160, 0.210, 0.260)
        )
    else:       # use standard block curvature
        curvature_scales = luigi.ListParameter(
            default=(0.133, 0.207, 0.280, 0.353)
        )

    def requires(self):
        return [
            PrepareData(dataset=self.dataset),
            ComputeBlockCurvature(dataset=self.dataset,
                                  imsize=self.imsize,
                                  batch_size=self.batch_size,
                                  scale=self.scale,
                                  oriented=self.oriented,
                                  curvature_scales=self.curvature_scales)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f-check' % s for s in self.curvature_scales])
        if self.oriented:
            curvdir = join('oriented', curvdir)
        else:
            curvdir = join('standard', curvdir)

        return [
            luigi.LocalTarget(
                join(basedir, curvdir, '%s_all.csv' % self.dataset)),
            luigi.LocalTarget(
                join(basedir, curvdir, '%s_mrr.csv' % self.dataset)),
            luigi.LocalTarget(
                join(basedir, curvdir, '%s_topk.csv' % self.dataset)),
        ]

    def run(self):
        import dorsal_utils
        import ranking
        from collections import defaultdict
        df = pd.read_csv(
            self.requires()[0].output().path, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )

        fname_curv_dict = {}
        curv_dict = self.requires()[1].output()
        curv_filepaths = curv_dict.keys()
        print('computing curvature vectors of dimension %d for %d images' % (
            self.curv_length, len(curv_filepaths)))
        #for fpath in curv_filepaths:
        for fpath in tqdm(curv_filepaths,
                          total=len(curv_filepaths), leave=False):
            curv_target = curv_dict[fpath]['curvature']
            with open(curv_target.path, 'rb') as f:
                curv = pickle.load(f)
            # no trailing edge could be extracted for this image
            if curv is None or curv.shape[0] < 2:
                continue
            else:
                assert curv.ndim == 2, 'curv.ndim == %d != 2' % (curv.ndim)

            fname = splitext(basename(fpath))[0]
            fname_curv_dict[fname] = dorsal_utils.resampleNd(
                curv, self.curv_length)

        db_dict, qr_dict = datasets.separate_database_queries(
            self.dataset, df['impath'].values,
            df['individual'].values, df['encounter'].values,
            fname_curv_dict
        )

        db_curvs_list = [len(db_dict[ind]) for ind in db_dict]
        qr_curvs_list = []
        for ind in qr_dict:
            for enc in qr_dict[ind]:
                qr_curvs_list.append(len(qr_dict[ind][enc]))

        print('max/mean/min images per db encounter: %.2f/%.2f/%.2f' % (
            np.max(db_curvs_list),
            np.mean(db_curvs_list),
            np.min(db_curvs_list))
        )
        print('max/mean/min images per qr encounter: %.2f/%.2f/%.2f' % (
            np.max(qr_curvs_list),
            np.mean(qr_curvs_list),
            np.min(qr_curvs_list))
        )

        simfunc = partial(
            ranking.dtw_alignment_cost,
            weights=np.ones(4, dtype=np.float32),
            window=self.window
        )

        indiv_rank_indices = defaultdict(list)
        qindivs = qr_dict.keys()
        with self.output()[0].open('w') as f:
            #for qind in qindivs:
            print('running identification for %d individuals' % (len(qindivs)))
            for qind in tqdm(qindivs, total=len(qindivs), leave=False):
                qencs = qr_dict[qind].keys()
                assert qencs, 'empty encounter list for %s' % qind
                for qenc in qencs:
                    rindivs, scores = ranking.rank_individuals(
                        qr_dict[qind][qenc], db_dict, simfunc)

                    rank = 1 + rindivs.index(qind)
                    indiv_rank_indices[qind].append(rank)

                    f.write('%s,%s\n' % (
                        qind, ','.join(['%s' % r for r in rindivs])))
                    f.write('%s\n' % (
                        ','.join(['%.6f' % s for s in scores])))

        with self.output()[1].open('w') as f:
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
        print('accuracy scores:')
        with self.output()[2].open('w') as f:
            f.write('topk,accuracy\n')
            for k in range(1, 1 + num_indivs):
                topk = (100. / num_queries) * (rank_indices <= k).sum()
                f.write('top-%d,%.6f\n' % (k, topk))
                if k in topk_scores:
                    print(' top-%d: %.2f%%' % (k, topk))


class ParameterSearch(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    curv_length = luigi.IntParameter(default=128)
    oriented = luigi.BoolParameter(default=False)

    def _gen_params_list(self):
        #base_scales = np.linspace(0.1, 0.3, 7)
        #incr_scales = np.linspace(0.01, 0.2, 10)
        base_scales = np.linspace(0.01, 0.2, 20)
        incr_scales = np.linspace(0.01, 0.1, 10)
        num_scales = 4
        params_list = []
        for incr in incr_scales:
            for base in base_scales:
                params_list.append(
                    [base + i * incr for i in range(num_scales)]
                )

        return params_list

    def requires(self):
        return [
            EvaluateIdentification(
                dataset=self.dataset,
                curv_length=self.curv_length,
                oriented=self.oriented,
                curvature_scales=scales)
            for scales in self._gen_params_list()
        ]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        if self.oriented:
            paramsdir = 'oriented'
        else:
            paramsdir = 'standard'

        return [
            luigi.LocalTarget(join(basedir, paramsdir, 'results.txt')),
            luigi.LocalTarget(join(basedir, paramsdir, 'best.txt')),
        ]

    def run(self):
        k_values = [1, 5, 10, 25]
        params_list = []
        evaluation_runs = self.requires()
        results = np.zeros(
            (len(evaluation_runs), len(k_values)), dtype=np.float32
        )

        with self.output()[0].open('w') as f:
            f.write('scales,%s\n' % ','.join([str(k) for k in k_values]))
            for i, task in enumerate(evaluation_runs):
                params_list.append(task.curvature_scales)
                csv_fpath = task.output()[2].path
                df = pd.read_csv(
                    csv_fpath, header='infer', usecols=['topk', 'accuracy']
                )
                scores = df['accuracy'].values
                for j, k in enumerate(k_values):
                    results[i, j] = scores[k - 1]
                f.write('%s: %s\n' % (
                    ','.join(['%.3f' % s for s in task.curvature_scales]),
                    ','.join(['%.2f' % s for s in results[i]])
                ))

        with self.output()[1].open('w') as f:
            for j, k in enumerate(k_values):
                f.write('best scales for top-%d accuracy\n' % k)
                sorted_idx = np.argsort(results[:, j])[::-1]
                for p, idx in enumerate(sorted_idx[0:5]):
                    scales = params_list[idx]
                    f.write(' %d) %s: %s\n' % (
                        1 + p, ','.join(['%.3f' % s for s in scales]),
                        ','.join(['%.2f' % r for r in results[idx]])
                    ))


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

        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['image'].path)]

        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        partial_visualize_individuals = partial(
            visualize_individuals,
            input_targets=separate_edges_targets,
            output_targets=output
        )

        #for fpath in tqdm(to_process, total=len(to_process)):
        #    partial_visualize_individuals(fpath)
        pool = mp.Pool(processes=32)
        pool.map(partial_visualize_individuals, to_process)


if __name__ == '__main__':
    luigi.run()
