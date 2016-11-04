import luigi
import datasets


class PrepareData(luigi.Task):
    dataset = luigi.ChoiceParameter(choices=['nz', 'sdrp'], var_type=str)

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('%s.csv' % self.dataset)

    def run(self):
        if self.dataset == 'nz':
            data_list = datasets.load_nz_dataset()
        elif self.dataset == 'sdrp':
            data_list = datasets.load_sdrp_dataset()
        else:
            assert False, 'bad dataset parameter'

        print('%d data tuples returned' % (len(data_list)))
        with self.output().open('w') as f:
            for img_fpath, indiv_name, enc_name in data_list:
                f.write('%s,%s,%s\n' % (img_fpath, indiv_name, enc_name))


if __name__ == '__main__':
    luigi.run()
