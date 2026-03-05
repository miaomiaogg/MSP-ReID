import os
import re
import glob
import logging
import numpy as np
import os.path as osp
from tools.utils import mkdir_if_missing, write_json, read_json


class VCClothes(object):
    """ VC-Clothes (with hair-augmented train + parsing paths)

    Layout:
      ROOT/VC-Clothes/
        train/*.jpg
        hair/*.jpg
        processed/*.png
        processed_hair/*.png
        query/*.jpg
        gallery/*.jpg
    """
    dataset_dir = 'VC-Clothes'

    def __init__(self, root='data', mode='all', **kwargs):
        self.dataset_dir   = osp.join(root, self.dataset_dir)
        self.train_dir     = osp.join(self.dataset_dir, 'train')
        self.hair_dir      = osp.join(self.dataset_dir, 'hair')
        self.proc_dir      = osp.join(self.dataset_dir, 'processed')
        self.proc_hair_dir = osp.join(self.dataset_dir, 'processed_hair')
        self.query_dir     = osp.join(self.dataset_dir, 'query')
        self.gallery_dir   = osp.join(self.dataset_dir, 'gallery')
        self.mode          = mode  # 'all' | 'sc' | 'cc'

        self._check_before_run()

        # ---- Train ----
        (train, num_train_pids, num_train_imgs, num_train_clothes,
         pid2clothes, train_stats) = self._process_dir_train_with_hair()

        # ---- Test ----
        (query, gallery, num_test_pids, num_query_imgs,
         num_gallery_imgs, num_test_clothes) = self._process_dir_test()

        num_total_pids     = num_train_pids + num_test_pids
        num_total_imgs     = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_total_clothes  = num_train_clothes + num_test_clothes
        num_test_imgs      = num_query_imgs + num_gallery_imgs

        logger = logging.getLogger('reid.dataset')
        logger.info("=> VC-Clothes loaded (hair-augmented TRAIN enabled)")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids,  num_test_imgs,  num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")
        logger.info("  train RAW images      | {:8d}".format(train_stats['raw']))
        logger.info("  train HAIR images     | {:8d}".format(train_stats['hair']))
        logger.info("  parsing on (ext=.png)")
        logger.info("  train RAW parse hit   | {:8d}".format(train_stats['raw_parse_hit']))
        logger.info("  train RAW parse miss  | {:8d}".format(train_stats['raw_parse_miss']))
        logger.info("  train HAIR parse hit  | {:8d}".format(train_stats['hair_parse_hit']))
        logger.info("  train HAIR parse miss | {:8d}".format(train_stats['hair_parse_miss']))
        logger.info("  ----------------------------------------")

        self.train   = train   # 5 元组
        self.query   = query   # 4 元组
        self.gallery = gallery # 4 元组

        self.num_train_pids    = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes       = pid2clothes

    def _check_before_run(self):
        for d in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available".format(d))

    def _process_dir_train_with_hair(self):
        img_paths  = sorted(glob.glob(osp.join(self.train_dir, '*.jpg')))
        hair_paths = sorted(glob.glob(osp.join(self.hair_dir, '*.jpg'))) if osp.exists(self.hair_dir) else []
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

        pid_container, clothes_container = set(), set()
        for p in img_paths + hair_paths:
            pid, camid, clothes, _ = pattern.search(osp.basename(p)).groups()
            clothes_id = pid + clothes
            pid_container.add(int(pid))
            clothes_container.add(clothes_id)

        pid_container     = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label         = {pid: i for i, pid in enumerate(pid_container)}
        clothes2label     = {cid: i for i, cid in enumerate(clothes_container)}

        num_pids    = len(pid_container)
        num_clothes = len(clothes_container)
        dataset     = []
        pid2clothes = np.zeros((num_pids, num_clothes), dtype=np.float32)

        stats = dict(raw=0, hair=0,
                     raw_parse_hit=0, raw_parse_miss=0,
                     hair_parse_hit=0, hair_parse_miss=0)

        def add_split(paths, is_hair=False):
            for path in paths:
                pid, camid, clothes, _ = pattern.search(osp.basename(path)).groups()
                clothes_id = pid + clothes
                pid, camid = int(pid), int(camid) - 1
                pid_label  = pid2label[pid]
                clothes_label = clothes2label[clothes_id]

                if is_hair:
                    parse_path = osp.join(self.proc_hair_dir, osp.splitext(osp.basename(path))[0] + ".png")
                else:
                    parse_path = osp.join(self.proc_dir, osp.splitext(osp.basename(path))[0] + ".png")

                if osp.exists(parse_path):
                    if is_hair: stats['hair_parse_hit'] += 1
                    else:       stats['raw_parse_hit']  += 1
                else:
                    if is_hair: stats['hair_parse_miss'] += 1
                    else:       stats['raw_parse_miss']  += 1

                dataset.append((path, pid_label, camid, clothes_label, parse_path))
                pid2clothes[pid_label, clothes_label] = 1.0
                stats['hair' if is_hair else 'raw'] += 1

        add_split(img_paths, is_hair=False)
        if hair_paths: add_split(hair_paths, is_hair=True)

        return dataset, num_pids, len(dataset), num_clothes, pid2clothes, stats

    def _process_dir_test(self):
        query_img_paths   = sorted(glob.glob(osp.join(self.query_dir, '*.jpg')))
        gallery_img_paths = sorted(glob.glob(osp.join(self.gallery_dir, '*.jpg')))
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

        pid_container, clothes_container = set(), set()
        for paths in [query_img_paths, gallery_img_paths]:
            for p in paths:
                pid, camid, clothes, _ = pattern.search(osp.basename(p)).groups()
                pid, camid = int(pid), int(camid)
                if self.mode == 'sc' and camid not in [2, 3]: continue
                if self.mode == 'cc' and camid not in [3, 4]: continue
                clothes_id = str(pid) + clothes
                pid_container.add(pid)
                clothes_container.add(clothes_id)

        pid_container     = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label     = {pid: i for i, pid in enumerate(pid_container)}
        clothes2label = {cid: i for i, cid in enumerate(clothes_container)}

        query_dataset, gallery_dataset = [], []
        for p in query_img_paths:
            pid, camid, clothes, _ = pattern.search(osp.basename(p)).groups()
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]: continue
            if self.mode == 'cc' and camid not in [3, 4]: continue
            camid -= 1
            clothes_id = str(pid) + clothes
            query_dataset.append((p, pid, camid, clothes2label[clothes_id]))

        for p in gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(osp.basename(p)).groups()
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]: continue
            if self.mode == 'cc' and camid not in [3, 4]: continue
            camid -= 1
            clothes_id = str(pid) + clothes
            gallery_dataset.append((p, pid, camid, clothes2label[clothes_id]))

        return query_dataset, gallery_dataset, len(pid_container), len(query_dataset), len(gallery_dataset), len(clothes_container)


def VCClothesSameClothes(root='data', **kwargs):
    return VCClothes(root=root, mode='sc')


def VCClothesClothesChanging(root='data', **kwargs):
    return VCClothes(root=root, mode='cc')
