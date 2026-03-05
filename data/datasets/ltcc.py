# import os
# import re
# import glob
# import logging
# import numpy as np
# import os.path as osp
# from tools.utils import mkdir_if_missing, write_json, read_json


# class LTCC(object):
#     """ LTCC (hair-augmented train + parsing paths)

#     Layout (assumed):
#       ROOT/LTCC_ReID/
#         train/*.png                 # RAW full-body
#         hair/*.png                  # HAIR-augmented full-body (e.g., h1_xxx.png)
#         processed/*.png             # parsing for RAW
#         processed_hair/*.png        # parsing for HAIR
#         query/*.png
#         test/*.png                  # gallery
#     """

#     dataset_dir = 'LTCC_ReID'

#     def __init__(self, root='data', **kwargs):
#         self.dataset_dir     = osp.join(root, self.dataset_dir)
#         self.train_dir       = osp.join(self.dataset_dir, 'train')
#         self.hair_dir        = osp.join(self.dataset_dir, 'hair')              # optional
#         self.proc_dir        = osp.join(self.dataset_dir, 'processed')         # parsing for train
#         self.proc_hair_dir   = osp.join(self.dataset_dir, 'processed_hair')    # parsing for hair
#         self.query_dir       = osp.join(self.dataset_dir, 'query')
#         self.gallery_dir     = osp.join(self.dataset_dir, 'test')

#         # --- Parsing config ---
#         self.with_parsing = kwargs.get('with_parsing', True)        # 默认启用 parsing
#         self.parsing_ext  = kwargs.get('parsing_ext', '.png')       # 默认只读 .png，可切换成 .npy/.mat

#         self._check_before_run()

#         # Train (RAW + optional HAIR), with parsing paths
#         (train, num_train_pids, num_train_imgs, num_train_clothes,
#          pid2clothes, train_stats) = self._process_dir_train_with_hair(
#             self.train_dir, self.hair_dir, self.proc_dir, self.proc_hair_dir
#         )

#         # Test splits unchanged (no parsing in tuples)
#         (query, gallery, num_test_pids, num_query_imgs,
#          num_gallery_imgs, num_test_clothes) = self._process_dir_test(self.query_dir, self.gallery_dir)

#         num_total_pids     = num_train_pids + num_test_pids
#         num_test_imgs      = num_query_imgs + num_gallery_imgs
#         num_total_imgs     = num_train_imgs + num_test_imgs
#         num_total_clothes  = num_train_clothes + num_test_clothes

#         logger = logging.getLogger('reid.dataset')
#         logger.info("=> LTCC loaded (hair-augmented TRAIN enabled)")
#         logger.info("Dataset statistics:")
#         logger.info("  ----------------------------------------")
#         logger.info("  subset   | # ids | # images | # clothes")
#         logger.info("  ----------------------------------------")
#         logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
#         logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids,  num_test_imgs,  num_test_clothes))
#         logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
#         logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
#         logger.info("  ----------------------------------------")
#         logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
#         logger.info("  ----------------------------------------")
#         # extra stats
#         logger.info("  train RAW images      | {:8d}".format(train_stats['raw']))
#         logger.info("  train HAIR images     | {:8d}".format(train_stats['hair']))
#         logger.info(f"  parsing on (ext={self.parsing_ext})")
#         logger.info("  train RAW parse hit   | {:8d}".format(train_stats['raw_parse_hit']))
#         logger.info("  train RAW parse miss  | {:8d}".format(train_stats['raw_parse_miss']))
#         logger.info("  train HAIR parse hit  | {:8d}".format(train_stats['hair_parse_hit']))
#         logger.info("  train HAIR parse miss | {:8d}".format(train_stats['hair_parse_miss']))
#         logger.info("  ----------------------------------------")

#         self.train   = train   # 5 元组: (img_path, pid, camid, clothes_id, parse_path)
#         self.query   = query   # 4 元组: (img_path, pid, camid, clothes_id)
#         self.gallery = gallery # 4 元组: (img_path, pid, camid, clothes_id)

#         self.num_train_pids     = num_train_pids
#         self.num_train_clothes  = num_train_clothes
#         self.pid2clothes        = pid2clothes

#     # -------------------------
#     # helpers
#     # -------------------------
#     def _check_before_run(self):
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError("'{}' is not available".format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError("'{}' is not available".format(self.gallery_dir))
#         if not osp.exists(self.proc_dir):
#             logging.getLogger('reid.dataset').warning("Parsing dir (processed) not found: '{}'".format(self.proc_dir))
#         if not osp.exists(self.hair_dir):
#             logging.getLogger('reid.dataset').warning("Hair dir not found: '{}' (fallback to RAW only)".format(self.hair_dir))
#         if not osp.exists(self.proc_hair_dir):
#             logging.getLogger('reid.dataset').warning("Parsing dir (processed_hair) not found: '{}'".format(self.proc_hair_dir))

#     @staticmethod
#     def _strip_hair_prefix(fname):
#         """去掉前缀 'h{digit}_'，如 'h1_0001_0002_red_c1.png' -> '0001_0002_red_c1.png'"""
#         return re.sub(r'^h\d+_', '', fname)

#     @staticmethod
#     def _parse_train_name(path_or_name):
#         name = osp.basename(path_or_name)
#         base = LTCC._strip_hair_prefix(name)
#         p1 = re.search(r'(\d+)_(\d+)_c(\d+)', base)
#         p2 = re.search(r'(\w+)_c', base)
#         if p1 is None or p2 is None:
#             raise ValueError(f"Bad LTCC train filename: {name}")
#         pid, _, camid = map(int, p1.groups())
#         clothes_token = p2.group(1)
#         camid -= 1  # cam starts from 0
#         return pid, camid, clothes_token

#     @staticmethod
#     def _parse_test_name(path_or_name):
#         name = osp.basename(path_or_name)
#         p1 = re.search(r'(\d+)_(\d+)_c(\d+)', name)
#         p2 = re.search(r'(\w+)_c', name)
#         if p1 is None or p2 is None:
#             raise ValueError(f"Bad LTCC test filename: {name}")
#         pid, _, camid = map(int, p1.groups())
#         clothes_token = p2.group(1)
#         camid -= 1
#         return pid, camid, clothes_token

#     # -------------------------
#     # train with hair + parsing
#     # -------------------------
#     def _process_dir_train_with_hair(self, raw_dir, hair_dir, proc_dir, proc_hair_dir):
#         raw_imgs  = sorted(glob.glob(osp.join(raw_dir, '*.png')))
#         hair_imgs = sorted(glob.glob(osp.join(hair_dir, '*.png'))) if osp.exists(hair_dir) else []

#         pid_container      = set()
#         clothes_container  = set()

#         for path in raw_imgs:
#             pid, camid, clothes_tok = self._parse_train_name(path)
#             pid_container.add(pid)
#             clothes_container.add(clothes_tok)
#         for path in hair_imgs:
#             pid, camid, clothes_tok = self._parse_train_name(path)
#             pid_container.add(pid)
#             clothes_container.add(clothes_tok)

#         pid_container     = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label      = {pid: i for i, pid in enumerate(pid_container)}
#         clothes2label  = {ct: i for i, ct in enumerate(clothes_container)}

#         num_pids     = len(pid_container)
#         num_clothes  = len(clothes_container)

#         dataset = []
#         pid2clothes = np.zeros((num_pids, num_clothes), dtype=np.float32)

#         stats = dict(raw=0, hair=0,
#                      raw_parse_hit=0, raw_parse_miss=0,
#                      hair_parse_hit=0, hair_parse_miss=0)

#         def add_split(img_list, is_hair=False):
#             for img_path in img_list:
#                 pid, camid, clothes_tok = self._parse_train_name(img_path)
#                 pid_label      = pid2label[pid]
#                 clothes_label  = clothes2label[clothes_tok]

#                 # --- 修改点：决定后缀 ---
#                 parse_path = None
#                 if self.with_parsing:
#                     fname = osp.splitext(osp.basename(img_path))[0] + self.parsing_ext
#                     if is_hair:
#                         parse_path = osp.join(proc_hair_dir, fname) if osp.exists(proc_hair_dir) else None
#                     else:
#                         parse_path = osp.join(proc_dir, fname) if osp.exists(proc_dir) else None

#                 # parse existence stats
#                 if is_hair:
#                     if parse_path and osp.exists(parse_path): stats['hair_parse_hit'] += 1
#                     else:                                     stats['hair_parse_miss'] += 1
#                 else:
#                     if parse_path and osp.exists(parse_path): stats['raw_parse_hit'] += 1
#                     else:                                     stats['raw_parse_miss'] += 1

#                 dataset.append((img_path, pid_label, camid, clothes_label, parse_path))
#                 pid2clothes[pid_label, clothes_label] = 1.0
#                 if is_hair: stats['hair'] += 1
#                 else:        stats['raw']  += 1

#         add_split(raw_imgs, is_hair=False)
#         if hair_imgs:
#             add_split(hair_imgs, is_hair=True)

#         num_imgs = len(dataset)
#         return dataset, num_pids, num_imgs, num_clothes, pid2clothes, stats

#     # -------------------------
#     # test/query (unchanged)
#     # -------------------------
#     def _process_dir_test(self, query_path, gallery_path):
#         query_img_paths   = sorted(glob.glob(osp.join(query_path, '*.png')))
#         gallery_img_paths = sorted(glob.glob(osp.join(gallery_path, '*.png')))

#         pid_container     = set()
#         clothes_container = set()

#         for p in query_img_paths:
#             pid, _, clothes_tok = self._parse_test_name(p)
#             pid_container.add(pid)
#             clothes_container.add(clothes_tok)
#         for p in gallery_img_paths:
#             pid, _, clothes_tok = self._parse_test_name(p)
#             pid_container.add(pid)
#             clothes_container.add(clothes_tok)

#         pid_container     = sorted(pid_container)
#         clothes_container = sorted(clothes_container)
#         pid2label      = {pid: i for i, pid in enumerate(pid_container)}
#         clothes2label  = {ct: i for i, ct in enumerate(clothes_container)}

#         num_pids     = len(pid_container)
#         num_clothes  = len(clothes_container)

#         query_dataset, gallery_dataset = [], []

#         for p in query_img_paths:
#             pid, camid, clothes_tok = self._parse_test_name(p)
#             query_dataset.append((p, pid, camid, clothes2label[clothes_tok]))

#         for p in gallery_img_paths:
#             pid, camid, clothes_tok = self._parse_test_name(p)
#             gallery_dataset.append((p, pid, camid, clothes2label[clothes_tok]))

#         num_imgs_query   = len(query_dataset)
#         num_imgs_gallery = len(gallery_dataset)

#         return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
import os
import re
import glob
import logging
import numpy as np
import os.path as osp
from tools.utils import mkdir_if_missing, write_json, read_json


class LTCC(object):
    """ LTCC (hair-augmented train + parsing paths)

    Layout (assumed):
      ROOT/LTCC_ReID/
        train/*.png                 # RAW full-body
        hair/*.png                  # HAIR-augmented full-body (e.g., h1_xxx.png)
        processed/*.png             # parsing for RAW
        processed_hair/*.png        # parsing for HAIR
        query/*.png
        test/*.png                  # gallery
    """

    dataset_dir = 'LTCC_ReID'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir     = osp.join(root, self.dataset_dir)
        self.train_dir       = osp.join(self.dataset_dir, 'train')
        self.hair_dir        = osp.join(self.dataset_dir, 'hair')              # optional
        self.proc_dir        = osp.join(self.dataset_dir, 'processed')         # parsing for train
        self.proc_hair_dir   = osp.join(self.dataset_dir, 'processed_hair')    # parsing for hair
        self.query_dir       = osp.join(self.dataset_dir, 'query')
        self.gallery_dir     = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        # Train (RAW + optional HAIR), with parsing paths
        (train, num_train_pids, num_train_imgs, num_train_clothes,
         pid2clothes, train_stats) = self._process_dir_train_with_hair(
            self.train_dir, self.hair_dir, self.proc_dir, self.proc_hair_dir
        )

        # Test splits unchanged (no parsing in tuples)
        (query, gallery, num_test_pids, num_query_imgs,
         num_gallery_imgs, num_test_clothes) = self._process_dir_test(self.query_dir, self.gallery_dir)

        num_total_pids     = num_train_pids + num_test_pids
        num_test_imgs      = num_query_imgs + num_gallery_imgs
        num_total_imgs     = num_train_imgs + num_test_imgs
        num_total_clothes  = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> LTCC loaded (hair-augmented TRAIN enabled)")
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
        # extra stats
        logger.info("  train RAW images      | {:8d}".format(train_stats['raw']))
        logger.info("  train HAIR images     | {:8d}".format(train_stats['hair']))
        logger.info("  parsing on (ext=.png)")
        logger.info("  train RAW parse hit   | {:8d}".format(train_stats['raw_parse_hit']))
        logger.info("  train RAW parse miss  | {:8d}".format(train_stats['raw_parse_miss']))
        logger.info("  train HAIR parse hit  | {:8d}".format(train_stats['hair_parse_hit']))
        logger.info("  train HAIR parse miss | {:8d}".format(train_stats['hair_parse_miss']))
        logger.info("  ----------------------------------------")

        self.train   = train   # 5 元组: (img_path, pid, camid, clothes_id, parse_path)
        self.query   = query   # 4 元组: (img_path, pid, camid, clothes_id)
        self.gallery = gallery # 4 元组: (img_path, pid, camid, clothes_id)

        self.num_train_pids     = num_train_pids
        self.num_train_clothes  = num_train_clothes
        self.pid2clothes        = pid2clothes

    # -------------------------
    # helpers
    # -------------------------
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.proc_dir):
            logging.getLogger('reid.dataset').warning("Parsing dir (processed) not found: '{}'".format(self.proc_dir))
        if not osp.exists(self.hair_dir):
            logging.getLogger('reid.dataset').warning("Hair dir not found: '{}' (fallback to RAW only)".format(self.hair_dir))
        if not osp.exists(self.proc_hair_dir):
            logging.getLogger('reid.dataset').warning("Parsing dir (processed_hair) not found: '{}'".format(self.proc_hair_dir))

    @staticmethod
    def _strip_hair_prefix(fname):
        """去掉前缀 'h{digit}_'，如 'h1_0001_0002_red_c1.png' -> '0001_0002_red_c1.png'"""
        return re.sub(r'^h\d+_', '', fname)

    @staticmethod
    def _parse_train_name(path_or_name):
        """
        解析训练文件名（支持带/不带 hair 前缀）：
          pattern1: r'(\d+)_(\d+)_c(\d+)'  -> pid, frame, cam
          pattern2: r'(\w+)_c'             -> clothes id token
        """
        name = osp.basename(path_or_name)
        base = LTCC._strip_hair_prefix(name)
        p1 = re.search(r'(\d+)_(\d+)_c(\d+)', base)
        p2 = re.search(r'(\w+)_c', base)
        if p1 is None or p2 is None:
            raise ValueError(f"Bad LTCC train filename: {name}")
        pid, _, camid = map(int, p1.groups())
        clothes_token = p2.group(1)
        camid -= 1  # cam starts from 0
        return pid, camid, clothes_token

    @staticmethod
    def _parse_test_name(path_or_name):
        """测试/查询与训练同名式（不含 hair 前缀），沿用原实现"""
        name = osp.basename(path_or_name)
        p1 = re.search(r'(\d+)_(\d+)_c(\d+)', name)
        p2 = re.search(r'(\w+)_c', name)
        if p1 is None or p2 is None:
            raise ValueError(f"Bad LTCC test filename: {name}")
        pid, _, camid = map(int, p1.groups())
        clothes_token = p2.group(1)
        camid -= 1
        return pid, camid, clothes_token

    # -------------------------
    # train with hair + parsing
    # -------------------------
    def _process_dir_train_with_hair(self, raw_dir, hair_dir, proc_dir, proc_hair_dir):
        raw_imgs  = sorted(glob.glob(osp.join(raw_dir, '*.png')))
        hair_imgs = sorted(glob.glob(osp.join(hair_dir, '*.png'))) if osp.exists(hair_dir) else []

        pid_container      = set()
        clothes_container  = set()

        # collect IDs and clothes from both RAW & HAIR
        for path in raw_imgs:
            pid, camid, clothes_tok = self._parse_train_name(path)
            pid_container.add(pid)
            clothes_container.add(clothes_tok)
        for path in hair_imgs:
            pid, camid, clothes_tok = self._parse_train_name(path)  # 注意：内部会剥掉 h前缀再解析
            pid_container.add(pid)
            clothes_container.add(clothes_tok)

        pid_container     = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label      = {pid: i for i, pid in enumerate(pid_container)}
        clothes2label  = {ct: i for i, ct in enumerate(clothes_container)}

        num_pids     = len(pid_container)
        num_clothes  = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes), dtype=np.float32)

        stats = dict(raw=0, hair=0,
                     raw_parse_hit=0, raw_parse_miss=0,
                     hair_parse_hit=0, hair_parse_miss=0)

        def add_split(img_list, is_hair=False):
            for img_path in img_list:
                pid, camid, clothes_tok = self._parse_train_name(img_path)
                pid_label      = pid2label[pid]
                clothes_label  = clothes2label[clothes_tok]

                # parsing path
                fname = osp.basename(img_path)
                if is_hair:
                    parse_path = osp.join(proc_hair_dir, fname) if osp.exists(proc_hair_dir) else None
                else:
                    parse_path = osp.join(proc_dir, fname) if osp.exists(proc_dir) else None

                # parse existence stats
                if is_hair:
                    if parse_path and osp.exists(parse_path): stats['hair_parse_hit'] += 1
                    else:                                     stats['hair_parse_miss'] += 1
                else:
                    if parse_path and osp.exists(parse_path): stats['raw_parse_hit'] += 1
                    else:                                     stats['raw_parse_miss'] += 1

                dataset.append((img_path, pid_label, camid, clothes_label, parse_path))
                pid2clothes[pid_label, clothes_label] = 1.0
                if is_hair: stats['hair'] += 1
                else:        stats['raw']  += 1

        add_split(raw_imgs, is_hair=False)
        if hair_imgs:
            add_split(hair_imgs, is_hair=True)

        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes, stats

    # -------------------------
    # test/query (unchanged)
    # -------------------------
    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths   = sorted(glob.glob(osp.join(query_path, '*.png')))
        gallery_img_paths = sorted(glob.glob(osp.join(gallery_path, '*.png')))

        pid_container     = set()
        clothes_container = set()

        for p in query_img_paths:
            pid, _, clothes_tok = self._parse_test_name(p)
            pid_container.add(pid)
            clothes_container.add(clothes_tok)
        for p in gallery_img_paths:
            pid, _, clothes_tok = self._parse_test_name(p)
            pid_container.add(pid)
            clothes_container.add(clothes_tok)

        pid_container     = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label      = {pid: i for i, pid in enumerate(pid_container)}
        clothes2label  = {ct: i for i, ct in enumerate(clothes_container)}

        num_pids     = len(pid_container)
        num_clothes  = len(clothes_container)

        query_dataset, gallery_dataset = [], []

        for p in query_img_paths:
            pid, camid, clothes_tok = self._parse_test_name(p)
            query_dataset.append((p, pid, camid, clothes2label[clothes_tok]))

        for p in gallery_img_paths:
            pid, camid, clothes_tok = self._parse_test_name(p)
            gallery_dataset.append((p, pid, camid, clothes2label[clothes_tok]))

        num_imgs_query   = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
