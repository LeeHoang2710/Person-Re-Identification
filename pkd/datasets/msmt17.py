from __future__ import division, print_function, absolute_import
import os.path as osp

from pkd.data.datasets import ImageDataset
from pkd.data_loader.incremental_datasets import IncrementalPersonReIDSamples
import glob
import re
# Log
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}

class IncrementalSamples4msmt17(IncrementalPersonReIDSamples):
    '''
    Market Dataset
    '''
    dataset_dir = 'msmt17'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall

        self.dataset_dir = osp.join(datasets_root, self.dataset_dir)

        # has_main_dir = False
        # for main_dir in VERSION_DICT:
        #     if osp.exists(osp.join(self.dataset_dir, main_dir)):
        #         train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
        #         test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
        #         has_main_dir = True
        #         break
        # assert has_main_dir, 'Dataset folder not found'
        train_dir = 'bounding_box_train'
        test_dir = 'bounding_box_test'
        query_dir = 'query'

        self.train_dir = osp.join(self.dataset_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, test_dir)
        self.query_dir = osp.join(self.dataset_dir, query_dir)
        # self.list_train_path = osp.join(
        #     self.dataset_dir, 'list_train.txt'
        # )
        # self.list_val_path = osp.join(
        #     self.dataset_dir, 'list_val.txt'
        # )
        # self.list_query_path = osp.join(
        #     self.dataset_dir, 'list_query.txt'
        # )
        # self.list_gallery_path = osp.join(
        #     self.dataset_dir, 'list_gallery.txt'
        # )

        # train = self.process_dir(self.train_dir, self.list_train_path)
        # val = self.process_dir(self.train_dir, self.list_val_path)
        # query = self.process_dir(self.test_dir, self.list_query_path)
        # gallery = self.process_dir(self.test_dir, self.list_gallery_path)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.test_dir, relabel=False)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        # if self.combineall:
        #     train += val
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(self.train, self.query, self.gallery)

    # def process_dir(self, dir_path, list_path):
    #     with open(list_path, 'r') as txt:
    #         lines = txt.readlines()

    #     data = []

    #     for img_idx, img_info in enumerate(lines):
    #         img_path, pid = img_info.split(' ')
    #         pid = int(pid)  # no need to relabel
    #         camid = int(img_path.split('_')[2]) - 1  # index starts from 0
    #         img_path = osp.join(dir_path, img_path)
    #         data.append((img_path, pid, camid, 'msmt17', pid))

    #     return data
    def process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append([img_path, pid, camid, 'msmt17', pid])

        return data

class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid, 'msmt17', pid))

        return data
