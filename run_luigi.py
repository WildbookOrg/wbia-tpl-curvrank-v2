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
        basedir = join('data', self.dataset, self.__class__.__name__,)
        return {
            'csv': luigi.LocalTarget(join(basedir, '%s3.csv' % self.dataset)),
            'pkl': luigi.LocalTarget(join(basedir, '%s3.pickle' % self.dataset))
        }

    def run(self):
        data_list = datasets.load_dataset(self.dataset)

        output = self.output()
        print('%d data tuples returned' % (len(data_list)))

        with output['csv'].open('w') as f:
            f.write('impath,individual,encounter\n')
            for img_fpath, indiv_name, enc_name in data_list:
                f.write('%s,%s,%s\n' % (img_fpath, indiv_name, enc_name))
        with output['pkl'].open('wb') as f:
            pickle.dump(data_list, f, pickle.HIGHEST_PROTOCOL)


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


class Preprocess(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)

    def requires(self):
        return {'PrepareData': PrepareData(dataset=self.dataset)}

    def complete(self):
        dependency_targets = self.requires()['PrepareData'].output()
        if not exists(dependency_targets['csv'].path) or\
                not exists(dependency_targets['pkl'].path):
            return False
        else:
            return all(map(
                lambda output: output.exists(),
                luigi.task.flatten(self.output())
            ))

    def output(self):
        prepare_data_target = self.requires()['PrepareData'].output()
        with prepare_data_target['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        return {
            'PrepareData': PrepareData(dataset=self.dataset),
            'Preprocess': Preprocess(
                dataset=self.dataset, imsize=self.imsize)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        preprocess_images_targets = self.requires()['Preprocess'].output()
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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset,
            ),
            'Localization': Localization(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
            )
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        localization_targets = self.requires()['Localization'].output()
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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset,
            ),
            'Localization': Localization(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size),
            'Segmentation': Segmentation(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        localization_targets = self.requires()['Localization'].output()
        segmentation_targets = self.requires()['Segmentation'].output()
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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset),
            'Localization': Localization(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale),
            'Segmentation': Segmentation(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale),
            'FindKeypoints': FindKeypoints(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size),
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        localization_targets = self.requires()['Localization'].output()
        segmentation_targets = self.requires()['Segmentation'].output()
        keypoints_targets = self.requires()['FindKeypoints'].output()
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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset),
            'Localization': Localization(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale),
            'ExtractOutline': ExtractOutline(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
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
        localization_targets = self.requires()['Localization'].output()
        extract_outline_targets = self.requires()['ExtractOutline'].output()
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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset),
            'SeparateEdges': SeparateEdges(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        curvdir = ','.join(['%.3f' % s for s in self.curvature_scales])
        if self.oriented:
            curvdir = join('oriented', curvdir)
        else:
            curvdir = join('standard', curvdir)
        outputs = {}
        for fpath, _, _ in prepare_data_filepaths:
            fname = splitext(basename(fpath))[0]
            pkl_fname = '%s.pickle' % fname
            outputs[fpath] = {
                'curvature': luigi.LocalTarget(
                    join(basedir, curvdir, pkl_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_block_curvature
        separate_edges_targets = self.requires()['SeparateEdges'].output()
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


class SeparateDatabaseQueries(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    num_db_encounters = luigi.IntParameter(default=10)

    def requires(self):
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset),
            'SeparateEdges': SeparateEdges(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        return {
            'database': luigi.LocalTarget(
                join(basedir, '%s.pickle' % 'database')),
            'queries': luigi.LocalTarget(
                join(basedir, '%s.pickle' % 'queries')),
        }

    def run(self):
        prepare_data_targets = self.requires()['PrepareData'].output()
        with prepare_data_targets['pkl'].open('rb') as f:
            prepare_data_filepaths = pickle.load(f)

        filepaths, individuals, encounters = zip(*prepare_data_filepaths)

        fname_trailing_edge_dict = {}
        trailing_edge_dict = self.requires()['SeparateEdges'].output()
        trailing_edge_filepaths = trailing_edge_dict.keys()
        for fpath in tqdm(trailing_edge_filepaths,
                          total=len(trailing_edge_filepaths), leave=False):
            trailing_edge_target = trailing_edge_dict[fpath]['trailing-coords']
            with open(trailing_edge_target.path, 'rb') as f:
                trailing_edge = pickle.load(f)
            # no trailing edge could be extracted for this image
            if trailing_edge is None:
                continue
            fname = splitext(basename(fpath))[0]
            fname_trailing_edge_dict[fname] = fpath

        db_dict, qr_dict = datasets.separate_database_queries(
            self.dataset, filepaths, individuals, encounters,
            fname_trailing_edge_dict, num_db_encounters=self.num_db_encounters
        )

        output = self.output()
        print('saving database with %d individuals' % (len(db_dict)))
        with output['database'].open('wb') as f:
            pickle.dump(db_dict, f, pickle.HIGHEST_PROTOCOL)
        print('saving queries with %d individuals' % (len(qr_dict)))
        with output['queries'].open('wb') as f:
            pickle.dump(qr_dict, f, pickle.HIGHEST_PROTOCOL)


class EvaluateIdentification(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    window = luigi.IntParameter(default=8)
    curv_length = luigi.IntParameter(default=128)
    oriented = luigi.BoolParameter(default=False)
    normalize = luigi.BoolParameter(default=False)

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
        return {
            'PrepareData': PrepareData(
                dataset=self.dataset),
            'ComputeBlockCurvature': ComputeBlockCurvature(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale,
                oriented=self.oriented,
                curvature_scales=self.curvature_scales),
            'SeparateDatabaseQueries': SeparateDatabaseQueries(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__ + '-par')
        curvdir = ','.join(['%.3f' % s for s in self.curvature_scales])
        if self.oriented:
            curvdir = join('oriented', curvdir)
        else:
            curvdir = join('standard', curvdir)

        # query dict tells us which encounters become result objects
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        with db_qr_target.output()['queries'].open('rb') as f:
            qr_curv_dict = pickle.load(f)

        output = {}
        for qind in qr_curv_dict:
            if qind not in output:
                output[qind] = {}
            for qenc in qr_curv_dict[qind]:
                # an encounter may belong to multiple individuals, hence qind
                output[qind][qenc] = luigi.LocalTarget(
                    join(basedir, curvdir, qind, '%s.pickle' % qenc)
                )

        return output

    def run(self):
        import dorsal_utils
        import ranking
        from workers import identify_encounters
        curv_targets = self.requires()['ComputeBlockCurvature'].output()
        db_qr_target = self.requires()['SeparateDatabaseQueries']
        db_fpath_dict_target = db_qr_target.output()['database']
        qr_fpath_dict_target = db_qr_target.output()['queries']

        with db_fpath_dict_target.open('rb') as f:
            db_fpath_dict = pickle.load(f)
        with qr_fpath_dict_target.open('rb') as f:
            qr_fpath_dict = pickle.load(f)

        db_curv_dict = {}
        print('loading curvature vectors for %d database individuals' % (
            len(db_fpath_dict)))
        for dind in db_fpath_dict:
            if dind not in db_curv_dict:
                db_curv_dict[dind] = []
            for fpath in db_fpath_dict[dind]:
                with curv_targets[fpath]['curvature'].open('rb') as f:
                    curv = pickle.load(f)
                if self.normalize:
                    curv -= curv.mean(axis=0)
                    curv /= curv.std(axis=0)
                curv = dorsal_utils.resampleNd(curv, self.curv_length)
                db_curv_dict[dind].append(curv)

        qr_curv_dict = {}
        print('loading curvature vectors for %d query individuals' % (
            len(qr_fpath_dict)))
        for qind in qr_fpath_dict:
            if qind not in qr_curv_dict:
                qr_curv_dict[qind] = {}
            for qenc in qr_fpath_dict[qind]:
                if qenc not in qr_curv_dict[qind]:
                    qr_curv_dict[qind][qenc] = []
                for fpath in qr_fpath_dict[qind][qenc]:
                    with curv_targets[fpath]['curvature'].open('rb') as f:
                        curv = pickle.load(f)
                    if self.normalize:
                        curv -= curv.mean(axis=0)
                        curv /= curv.std(axis=0)
                    curv = dorsal_utils.resampleNd(curv, self.curv_length)
                    qr_curv_dict[qind][qenc].append(curv)

        db_curvs_list = [len(db_curv_dict[ind]) for ind in db_curv_dict]
        qr_curvs_list = []
        for ind in qr_curv_dict:
            for enc in qr_curv_dict[ind]:
                qr_curvs_list.append(len(qr_curv_dict[ind][enc]))

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
            weights=np.ones(len(self.curvature_scales), dtype=np.float32),
            window=self.window
        )

        output = self.output()
        qindivs = qr_curv_dict.keys()
        print('running identification for %d individuals' % (len(qindivs)))
        partial_identify_encounters = partial(
            identify_encounters,
            qr_curv_dict=qr_curv_dict,
            db_curv_dict=db_curv_dict,
            simfunc=simfunc,
            output_targets=output,
        )
        pool = mp.Pool(processes=32)
        pool.map(partial_identify_encounters, qindivs)
        #for qind in tqdm(qindivs, total=len(qindivs), leave=False):
        #    partial_identify_encounters(qind)


class SummarizeResults(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    window = luigi.IntParameter(default=8)
    curv_length = luigi.IntParameter(default=128)
    oriented = luigi.BoolParameter(default=False)
    normalize = luigi.BoolParameter(default=False)
    num_db_encounters = luigi.IntParameter(default=10)

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
        return {
            'SeparateDatabaseQueries': SeparateDatabaseQueries(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale,
                num_db_encounters=self.num_db_encounters),
            'EvaluateIdentification': EvaluateIdentification(
                dataset=self.dataset,
                imsize=self.imsize,
                batch_size=self.batch_size,
                scale=self.scale,
                window=self.window,
                curv_length=self.curv_length,
                curvature_scales=self.curvature_scales,
                oriented=self.oriented,
                normalize=self.normalize)
        }

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curvature_scales])
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
        from collections import defaultdict
        evaluation_targets = self.requires()['EvaluateIdentification'].output()
        db_qr_output = self.requires()['SeparateDatabaseQueries'].output()
        with db_qr_output['database'].open('rb') as f:
            db_dict = pickle.load(f)
        db_indivs = db_dict.keys()
        indiv_rank_indices = defaultdict(list)
        with self.output()[0].open('w') as f:
            for qind in tqdm(evaluation_targets, leave=False):
                for qenc in evaluation_targets[qind]:
                    with evaluation_targets[qind][qenc].open('rb') as f1:
                        result_dict = pickle.load(f1)
                    scores = np.zeros(len(db_indivs), dtype=np.float32)
                    for i, dind in enumerate(db_indivs):
                        result_matrix = result_dict[dind]
                        scores[i] = result_matrix.min(axis=None)

                    asc_scores_idx = np.argsort(scores)
                    ranked_indivs = [db_indivs[idx] for idx in asc_scores_idx]
                    ranked_scores = [scores[idx] for idx in asc_scores_idx]

                    rank = 1 + ranked_indivs.index(qind)
                    indiv_rank_indices[qind].append(rank)

                    f.write('%s,%s\n' % (
                        qind, ','.join(['%s' % r for r in ranked_indivs])))
                    f.write('%s\n' % (
                        ','.join(['%.6f' % s for s in ranked_scores])))

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


class VisualizeMisidentifications(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    window = luigi.IntParameter(default=8)
    curv_length = luigi.IntParameter(default=128)
    oriented = luigi.BoolParameter(default=False)
    normalize = luigi.BoolParameter(default=False)
    num_qr_visualizations = luigi.IntParameter(default=3)
    num_db_visualizations = luigi.IntParameter(default=5)

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
            SeparateEdges(dataset=self.dataset,
                          imsize=self.imsize,
                          batch_size=self.batch_size,
                          scale=self.scale),
            SeparateDatabaseQueries(dataset=self.dataset,
                                    imsize=self.imsize,
                                    batch_size=self.batch_size,
                                    scale=self.scale),
            ComputeBlockCurvature(dataset=self.dataset,
                                  imsize=self.imsize,
                                  batch_size=self.batch_size,
                                  scale=self.scale,
                                  oriented=self.oriented,
                                  curvature_scales=self.curvature_scales),
            EvaluateIdentification(dataset=self.dataset,
                                   imsize=self.imsize,
                                   batch_size=self.batch_size,
                                   scale=self.scale,
                                   window=self.window,
                                   curv_length=self.curv_length,
                                   curvature_scales=self.curvature_scales,
                                   oriented=self.oriented,
                                   normalize=self.normalize)
        ]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        curvdir = ','.join(['%.3f' % s for s in self.curvature_scales])
        if self.oriented:
            curvdir = join('oriented', curvdir)
        else:
            curvdir = join('standard', curvdir)

        output = {}
        evaluation_targets = self.requires()[3].output()
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
        edges_targets = self.requires()[0].output()
        database_queries_targets = self.requires()[1].output()
        block_curv_targets = self.requires()[2].output()
        with database_queries_targets['database'].open('rb') as f:
            db_dict = pickle.load(f)
        with database_queries_targets['queries'].open('rb') as f:
            qr_dict = pickle.load(f)

        evaluation_targets = self.requires()[3].output()
        qindivs = evaluation_targets.keys()
        partial_visualize_misidentifications = partial(
            visualize_misidentifications,
            qr_dict=qr_dict,
            db_dict=db_dict,
            num_qr=self.num_qr_visualizations,
            num_db=self.num_db_visualizations,
            input1_targets=evaluation_targets,
            input2_targets=edges_targets,
            input3_targets=block_curv_targets,
            output_targets=output,
        )

        #for qind in qindivs:
        #    partial_visualize_misidentifications(qind)
        pool = mp.Pool(processes=32)
        pool.map(partial_visualize_misidentifications, qindivs)


if __name__ == '__main__':
    luigi.run()
