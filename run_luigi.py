import cPickle as pickle
import cv2
import pandas as pd
import luigi
import datasets
import imutils
import model
import numpy as np

from tqdm import tqdm
#from os import mkdir
from os.path import basename, join, splitext


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

    def output(self):
        csv_fpath = self.requires()[0].output().path
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
        image_filepaths = self.output().keys()
        print('%d images to process' % (len(image_filepaths)))

        for fpath in tqdm(image_filepaths, total=len(image_filepaths)):
            img = cv2.imread(fpath)
            resz, M = imutils.center_pad_with_transform(img, self.imsize)
            _, resz_buf = cv2.imencode('.png', resz)

            resz_path = self.output()[fpath]['resized']
            trns_path = self.output()[fpath]['transform']
            with resz_path.open('wb') as f1,\
                    trns_path.open('wb') as f2:
                f1.write(resz_buf)
                pickle.dump(M, f2, pickle.HIGHEST_PROTOCOL)


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

        print('compiling theano functions for localization and segmentation')
        loc_seg_func = theano_funcs.create_end_to_end_infer_func(layers)

        input_entries = self.requires()[0].output()
        image_filepaths = input_entries.keys()
        print('%d images to process' % (len(image_filepaths)))

        num_batches = (
            len(image_filepaths) + self.batch_size - 1) / self.batch_size
        print('%d batches to process' % (num_batches))
        for i in tqdm(range(num_batches), total=num_batches):
            idx_range = range(i * self.batch_size, (i + 1) * self.batch_size)
            X_batch = np.empty(
                (len(idx_range), 3, height, width), dtype=np.float32
            )
            for i, idx in enumerate(idx_range):
                impath = input_entries[image_filepaths[idx]]['resized'].path
                img = cv2.imread(impath)
                X_batch[i] = img.transpose(2, 0, 1) / 255.

            M_batch, X_batch_loc, X_batch_seg = loc_seg_func(X_batch)
            for i, idx in enumerate(idx_range):
                Mloc = np.vstack(
                    (M_batch[i].reshape((2, 3)), np.array([0, 0, 1]))
                )
                img_loc = (255. * X_batch_loc[i]).astype(
                    np.uint8).transpose(1, 2, 0)
                img_seg = (255. * X_batch_seg[i]).astype(
                    np.uint8).transpose(1, 2, 0)

                _, loc_buf = cv2.imencode('.png', img_loc)
                _, seg_buf = cv2.imencode('.png', img_seg)
                loc_lr_path = self.output()[image_filepaths[idx]]['loc-lr']
                seg_lr_path = self.output()[image_filepaths[idx]]['seg-lr']
                trns_path = self.output()[image_filepaths[idx]]['transform']
                with loc_lr_path.open('wb') as f1,\
                        seg_lr_path.open('wb') as f2,\
                        trns_path.open('wb') as f3:
                    f1.write(loc_buf)
                    f2.write(seg_buf)
                    pickle.dump(Mloc, f3, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    luigi.run()
