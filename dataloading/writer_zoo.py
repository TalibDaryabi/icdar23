import os.path
from .regex import ImageFolder


class WriterZoo:

    @staticmethod
    def new(dataset_name, dataset_type, desc, **kwargs):
        return ImageFolder(dataset_name=dataset_name, dataset_type=dataset_type, path=desc['path'], regex=desc['regex'],
                           **kwargs)
    # def new(desc, **kwargs):
    #     return ImageFolder( path=desc['path'], regex=desc['regex'],**kwargs)

    @staticmethod
    def get(dataset_name, dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]  # ex: icdar17, icdar13
        s = d['set'][set]  # set is either train or test.

        s['path'] = os.path.join(d['basepath'], s['path'])
        # return WriterZoo.new(desc=s, **kwargs)

        return WriterZoo.new(dataset_name=dataset_name, dataset_type=set, desc=s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': '/tmp/uc46epev/icdar17_data/ouput_icdar_temp/orginal_script',
            'set': {
                'test': {'path': "/tmp/uc46epev/icdar17_data/output_icdar_real_/bin/test",
                         'regex': {'writer': '(\d+)', 'page': '\d+-IMG_MAX_(\d+)'}},

                'train': {'path': "/tmp/uc46epev/icdar17_data/output_icdar_real_/bin/train_5000",
                          'regex': {'cluster': '(\d+)', 'writer': '\d+_(\d+)', 'page': '\d+_\d+-IMG_MAX_(\d+)_\d+'}},

            }
        },

        # 'icdar2013': {
        #     'basepath': '/data/mpeer/resources',
        #     'set': {
        #         'test' :  {'path': 'icdar2013_test_sift_patches_binarized',
        #                           'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},
        #
        #         'train' :  {'path': 'icdar2013_train_sift_patches_1000/',
        #                           'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}}
        #     }
        # },
        #
        # 'icdar2019': {
        #     'basepath': '/data/mpeer/resources',
        #     'set': {
        #         'test' :  {'path': 'wi_comp_19_test_patches',
        #                           'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},
        #
        #         'train' :  {'path': 'wi_comp_19_validation_patches',
        #                           'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
        #     }
        # }
    }
