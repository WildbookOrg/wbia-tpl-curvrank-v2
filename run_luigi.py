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
            resz_fname = '%s.png' % splitext(basename(fpath))[0]
            trns_fname = '%s.pickle' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'resized': luigi.LocalTarget(
                    join(basedir, 'resized', resz_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', trns_fname)),
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
        #for fpath in tqdm(to_process, total=len(to_process)):
        #    preprocess_images(fpath, self.imsize, output)
        pool = mp.Pool(processes=32)
        partial_preprocess_images = partial(
            preprocess_images,
            imsize=self.imsize,
            output_targets=output,
        )
        pool.map(partial_preprocess_images, to_process)


class LocalizationSegmentation(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    def requires(self):
        return [PreprocessImages(dataset=self.dataset, imsize=self.imsize)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
            img_fname = '%s.png' % splitext(basename(fpath))[0]
            trns_fname = '%s.pickle' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'loc-lr': luigi.LocalTarget(
                    join(basedir, 'loc-lr', img_fname)),
                'seg-lr': luigi.LocalTarget(
                    join(basedir, 'seg-lr', img_fname)),
                'transform': luigi.LocalTarget(
                    join(basedir, 'transform', trns_fname)),
            }

        return outputs

    def run(self):
        import localization
        import segmentation
        import theano_funcs
        height, width = 256, 256

        print('building localization model')
        layers_localization = localization.build_model(
            (None, 3, height, width), downsample=2)
        print('building segmentation model')
        layers_segmentation = segmentation.build_model_batchnorm(
            (None, 3, 128, 128))

        localization_weightsfile = join(
            'data', 'weights', 'weights_localization_all.pickle'
        )
        print('loading weights for the localization network from %s' % (
            localization_weightsfile))
        model.load_weights([
            layers_localization['trans'], layers_localization['loc']],
            localization_weightsfile
        )
        segmentation_weightsfile = join(
            'data', 'weights', 'weights_segmentation_all.pickle'
        )
        print('loading weights for the segmentation network from %s' % (
            segmentation_weightsfile))
        model.load_weights(
            layers_segmentation['seg_out'],
            segmentation_weightsfile
        )

        layers = {}
        for name in layers_localization:
            layers[name] = layers_localization[name]
        for name in layers_segmentation:
            layers[name] = layers_segmentation[name]

        layers.pop('seg_in')
        layers['seg_conv1'].input_layer = layers['trans']

        print('compiling theano functions for loc. and seg.')
        loc_seg_func = theano_funcs.create_end_to_end_infer_func(layers)

        output = self.output()
        preprocess_images_targets = self.requires()[0].output()
        image_filepaths = preprocess_images_targets.keys()

        # we don't parallelize this function because it uses the gpu
        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['loc-lr'].path) or
                      not exists(output[fpath]['seg-lr'].path) or
                      not exists(output[fpath]['transform'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        num_batches = (
            len(image_filepaths) + self.batch_size - 1) / self.batch_size
        print('%d batches to process' % (num_batches))
        for i in tqdm(range(num_batches), total=num_batches, leave=False):
            idx_range = range(i * self.batch_size,
                              (i + 1) * self.batch_size)
            X_batch = np.empty(
                (len(idx_range), 3, height, width), dtype=np.float32
            )
            for i, idx in enumerate(idx_range):
                fpath = image_filepaths[idx]
                impath = preprocess_images_targets[fpath]['resized'].path
                img = cv2.imread(impath)
                X_batch[i] = img.transpose(2, 0, 1) / 255.

            M_batch, X_batch_loc, X_batch_seg = loc_seg_func(X_batch)
            for i, idx in enumerate(idx_range):
                fpath = image_filepaths[idx]
                loc_lr_target = output[fpath]['loc-lr']
                seg_lr_target = output[fpath]['seg-lr']
                trns_target = output[fpath]['transform']

                Mloc = np.vstack(
                    (M_batch[i].reshape((2, 3)), np.array([0, 0, 1]))
                )
                img_loc = (255. * X_batch_loc[i]).astype(
                    np.uint8).transpose(1, 2, 0)
                img_seg = (255. * X_batch_seg[i]).astype(
                    np.uint8).transpose(1, 2, 0)

                _, loc_buf = cv2.imencode('.png', img_loc)
                _, seg_buf = cv2.imencode('.png', img_seg)

                with loc_lr_target.open('wb') as f1,\
                        seg_lr_target.open('wb') as f2,\
                        trns_target.open('wb') as f3:
                    f1.write(loc_buf)
                    f2.write(seg_buf)
                    pickle.dump(Mloc, f3, pickle.HIGHEST_PROTOCOL)


class ExtractLowResolutionOutline(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)

    def requires(self):
        return [LocalizationSegmentation(dataset=self.dataset,
                                         imsize=self.imsize,
                                         batch_size=self.batch_size)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
            img_fname = '%s.png' % splitext(basename(fpath))[0]
            outline_fname = '%s.pickle' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'outline-visual': luigi.LocalTarget(
                    join(basedir, 'outline-visual', img_fname)),
                'outline-coords': luigi.LocalTarget(
                    join(basedir, 'outline-coords', outline_fname)),
            }

        return outputs

    def run(self):
        from workers import extract_low_resolution_outline
        output = self.output()
        localization_segmentation_targets = self.requires()[0].output()
        image_filepaths = localization_segmentation_targets.keys()
        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['outline-visual'].path) or
                      not exists(output[fpath]['outline-coords'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        #for fpath in tqdm(to_process, total=len(image_filepaths)):
        pool = mp.Pool(processes=32)
        partial_extract_low_resolution_outline = partial(
            extract_low_resolution_outline,
            input_targets=localization_segmentation_targets,
            output_targets=output,
        )
        pool.map(partial_extract_low_resolution_outline, to_process)


class RefinedLocalizationSegmentation(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [
            PrepareData(dataset=self.dataset),
            PreprocessImages(dataset=self.dataset, imsize=self.imsize),
            LocalizationSegmentation(dataset=self.dataset,
                                     imsize=self.imsize,
                                     batch_size=self.batch_size)
        ]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[1].output().keys():
            img_fname = '%s.png' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'loc-hr': luigi.LocalTarget(
                    join(basedir, 'loc-hr', img_fname)),
                'seg-hr': luigi.LocalTarget(
                    join(basedir, 'seg-hr', img_fname)),
            }

        return outputs

    def run(self):
        from workers import refine_localization_segmentation
        csv_fpath = self.requires()[0].output().path
        df = pd.read_csv(
            csv_fpath, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )

        output = self.output()
        # the paths to the original images
        image_filepaths = df['impath'].values

        # the dict containing the preprocessing transforms
        preprocess_images_targets = self.requires()[1].output()

        # the dict containing the localization, segmentation, and stn transform
        localization_segmentation_targets = self.requires()[2].output()

        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['loc-hr'].path) or
                      not exists(output[fpath]['seg-hr'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        #for fpath in tqdm(to_process, total=len(image_filepaths)):
        pool = mp.Pool(processes=32)
        partial_refine_localization_segmentation = partial(
            refine_localization_segmentation,
            scale=self.scale,
            imsize=self.imsize,
            input1_targets=preprocess_images_targets,
            input2_targets=localization_segmentation_targets,
            output_targets=output,
        )
        pool.map(partial_refine_localization_segmentation, to_process)


class ExtractHighResolutionOutline(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)

    def requires(self):
        return [
            ExtractLowResolutionOutline(dataset=self.dataset,
                                        imsize=self.imsize,
                                        batch_size=self.batch_size),
            RefinedLocalizationSegmentation(dataset=self.dataset,
                                            imsize=self.imsize,
                                            batch_size=self.batch_size,
                                            scale=self.scale,)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[1].output().keys():
            img_fname = '%s.png' % splitext(basename(fpath))[0]
            outline_fname = '%s.pickle' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'outline-visual': luigi.LocalTarget(
                    join(basedir, 'outline-visual', img_fname)),
                'outline-coords': luigi.LocalTarget(
                    join(basedir, 'outline-coords', outline_fname)),
            }

        return outputs

    def run(self):
        from workers import extract_high_resolution_outline
        output = self.output()
        extract_low_resolution_outline_targets = self.requires()[0].output()
        localization_segmentation_targets = self.requires()[1].output()
        image_filepaths = localization_segmentation_targets.keys()

        to_process = [fpath for fpath in image_filepaths if
                      not exists(output[fpath]['outline-coords'].path) or
                      not exists(output[fpath]['outline-visual'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(image_filepaths)))

        #for fpath in tqdm(to_process, total=len(image_filepaths)):
        pool = mp.Pool(processes=32)
        partial_extract_high_resolution_outline = partial(
            extract_high_resolution_outline,
            scale=self.scale,
            input1_targets=extract_low_resolution_outline_targets,
            input2_targets=localization_segmentation_targets,
            output_targets=output,
        )
        pool.map(partial_extract_high_resolution_outline, to_process)


class ComputeBlockCurvature(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    curvature_scales = luigi.Parameter(default=(0.133, 0.207, 0.280, 0.353))

    def requires(self):
        return [
            ExtractHighResolutionOutline(dataset=self.dataset,
                                         imsize=self.imsize,
                                         batch_size=self.batch_size,
                                         scale=self.scale)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        outputs = {}
        for fpath in self.requires()[0].output().keys():
            curv_fname = '%s.pickle' % splitext(basename(fpath))[0]
            outputs[fpath] = {
                'curvature': luigi.LocalTarget(
                    join(basedir, 'curvature', curv_fname)),
            }

        return outputs

    def run(self):
        from workers import compute_block_curvature
        extract_high_resolution_outline_targets = self.requires()[0].output()
        output = self.output()
        input_filepaths = extract_high_resolution_outline_targets.keys()

        to_process = [fpath for fpath in input_filepaths if
                      not exists(output[fpath]['curvature'].path)]
        print('%d of %d images to process' % (
            len(to_process), len(input_filepaths)))

        #for fpath in tqdm(to_process, total=len(outline_filepaths)):
        pool = mp.Pool(processes=32)
        partial_compute_block_curvature = partial(
            compute_block_curvature,
            scales=self.curvature_scales,
            input_targets=extract_high_resolution_outline_targets,
            output_targets=output,
        )
        pool.map(partial_compute_block_curvature, to_process)


class EvaluateIdentification(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    imsize = luigi.IntParameter(default=256)
    batch_size = luigi.IntParameter(default=32)
    scale = luigi.IntParameter(default=4)
    window = luigi.IntParameter(default=8)
    curv_length = luigi.IntParameter(default=128)

    def requires(self):
        return [
            PrepareData(dataset=self.dataset),
            ComputeBlockCurvature(dataset=self.dataset,
                                  imsize=self.imsize,
                                  batch_size=self.batch_size,
                                  scale=self.scale)]

    def output(self):
        basedir = join('data', self.dataset, self.__class__.__name__)
        return [
            luigi.LocalTarget(join(basedir, '%s_all.csv' % self.dataset)),
            luigi.LocalTarget(join(basedir, '%s_mrr.csv' % self.dataset)),
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
        for fpath in tqdm(curv_filepaths,
                          total=len(curv_filepaths), leave=False):
            curv_target = curv_dict[fpath]['curvature']
            with open(curv_target.path, 'rb') as f:
                curv = pickle.load(f)
            # no trailing edge could be extracted for this image
            if curv is None:
                continue

            fname = splitext(basename(fpath))[0]
            fname_curv_dict[fname] = dorsal_utils.resampleNd(
                curv, self.curv_length)

        db_dict, qr_dict = datasets.separate_database_queries(
            self.dataset, df['impath'].values,
            df['individual'].values, df['encounter'].values,
            fname_curv_dict
        )

        simfunc = partial(
            ranking.dtw_alignment_cost,
            weights=np.ones(4, dtype=np.float32),
            window=self.window
        )

        indiv_reciprocal_rank = defaultdict(list)
        top1, top5, top10, total = 0, 0, 0, 0
        qindivs = qr_dict.keys()
        with self.output()[0].open('w') as f:
            print('running identification for %d individuals' % (len(qindivs)))
            for qind in tqdm(qindivs, total=len(qindivs)):
                qencs = qr_dict[qind].keys()
                for qenc in qencs:
                    rindivs, scores = ranking.rank_individuals(
                        qr_dict[qind][qenc], db_dict, simfunc)

                    indiv_reciprocal_rank[qind].append(
                        1. / (1. + rindivs.index(qind))
                    )

                    # first the query, then the ranking
                    f.write('%s,%s\n' % (
                        qind, ','.join(['%s' % r for r in rindivs])))
                    f.write('%s\n' % (
                        ','.join(['%.6f' % s for s in scores])))

                    total += 1
                    top1 += qind in rindivs[0:1]
                    top5 += qind in rindivs[0:5]
                    top10 += qind in rindivs[0:10]

            print('accuracy scores:')
            print('top1:  %.2f%%' % (100. * top1 / total))
            print('top5:  %.2f%%' % (100. * top5 / total))
            print('top10: %.2f%%' % (100. * top10 / total))

            with self.output()[1].open('w') as f:
                f.write('per individual mean reciprocal rank:\n')
                for qind in indiv_reciprocal_rank.keys():
                    mrr = np.mean(indiv_reciprocal_rank[qind])
                    num = len(indiv_reciprocal_rank[qind])
                    f.write(' %s (%d enc.): %.6f\n' % (qind, num, mrr))


if __name__ == '__main__':
    luigi.run()
