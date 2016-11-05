import cv2
import pandas as pd
import luigi
import datasets
import imutils
import model
import numpy as np

from tqdm import tqdm
from os import mkdir
from os.path import basename, join, splitext


class PrepareData(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(
            join('data', self.dataset, '%s.csv' % self.dataset)
        )

    def run(self):
        data_list = datasets.load_dataset(self.dataset)

        print('%d data tuples returned' % (len(data_list)))

        with self.output().open('w') as f:
            f.write('impath,individual,encounter\n')
            for img_fpath, indiv_name, enc_name in data_list:
                f.write('%s,%s,%s\n' % (img_fpath, indiv_name, enc_name))


class LocalizationSegmentation(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)
    batch_size = luigi.IntParameter(default=32)
    def requires(self):
        return [PrepareData(dataset=self.dataset)]

    def output(self):
        return [
            luigi.LocalTarget(
                join('data', self.dataset, 'localizations-low')),
            luigi.LocalTarget(
                join('data', self.dataset, 'segmentations-low')),
            luigi.LocalTarget(
                join('data', self.dataset, 'localizations-high')),
            luigi.LocalTarget(
                join('data', self.dataset, 'segmentations-high')),
        ]

    def run(self):
        import localization
        import segmentation
        import theano_funcs
        loc_low_dir = join('data', self.dataset, 'localizations-low')
        loc_high_dir = join('data', self.dataset, 'localizations-high')
        segm_low_dir = join('data', self.dataset, 'segmentations-low')
        segm_high_dir = join('data', self.dataset, 'segmentations-high')
        for d in (loc_low_dir, loc_high_dir, segm_low_dir, segm_high_dir):
            mkdir(d)
        height, width = 256, 256
        csv_fpath = join('data', self.dataset, '%s.csv' % (self.dataset))
        print('parsing csv at %s' % (csv_fpath))
        df = pd.read_csv(
            csv_fpath, header='infer',
            usecols=['impath', 'individual', 'encounter']
        )

        image_filepaths = df['impath'].values
        print('%d images to process' % (len(image_filepaths)))

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
        num_batches = (
            len(image_filepaths) + self.batch_size - 1) / self.batch_size
        print('%d batches to process' % (num_batches))
        for i in tqdm(range(num_batches), total=num_batches):
            idx_range = range(i * self.batch_size, (i + 1) * self.batch_size)
            X_batch = np.empty(
                (len(idx_range), 3, height, width), dtype=np.float32
            )
            for i, idx in enumerate(idx_range):
                img = cv2.imread(image_filepaths[idx])
                img = imutils.center_pad(img, height)
                X_batch[i] = img.transpose(2, 0, 1) / 255.

            M_batch, X_batch_loc, X_batch_seg = loc_seg_func(X_batch)
            for i, idx in enumerate(idx_range):
                impath = splitext(basename(image_filepaths[idx]))[0]
                #Mloc = M_batch[i].reshape((2, 3))
                img_loc = (255. * X_batch_loc[i]).astype(
                    np.uint8).transpose(1, 2, 0)
                img_seg = (255. * X_batch_seg[i]).astype(
                    np.uint8).transpose(1, 2, 0)

                cv2.imwrite(join(segm_low_dir, '%s.png' % impath), img_seg)
                cv2.imwrite(join(loc_low_dir, '%s.png' % impath), img_loc)


if __name__ == '__main__':
    luigi.run()
