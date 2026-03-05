import os
import glob
import random
import logging
import numpy as np
import os.path as osp

class PRCC(object):
    """PRCC with hair-augmented training import + parsing paths.

    Layout (PRCC is special-cased):
      ROOT/prcc/
        rgb/
          train/<pid>/*.jpg
          hair/<pid>/*.jpg
          val/<pid>/*.jpg
          test/A|B|C/<pid>/*.jpg
          processed/<pid>/*.png           # parsing for RAW (A0001.png ...)
          processed_hair/<pid>/*.png      # parsing for HAIR (h1_A0001.png ...)

    Notes:
    - Camera inferred by first occurrence of 'A'/'B'/'C' in filename:
      works for both 'xxx.jpg' and 'h1_xxx.jpg' (e.g., 'h2_A0001.jpg' -> 'A').
    - All HAIR images are included in TRAIN only.
    - When with_parsing=True, each sample is (img, pid, camid, clothes_id, parse_path).
      If parsing not found, parse_path=None (no exception).
    """

    dataset_dir = 'prcc'

    def __init__(self, root='data', **kwargs):
        # --- Dirs ---
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.rgb_root    = osp.join(self.dataset_dir, 'rgb')
        self.train_dir   = osp.join(self.rgb_root, 'train')
        self.hair_dir    = osp.join(self.rgb_root, 'hair')   # only for train
        self.val_dir     = osp.join(self.rgb_root, 'val')
        self.test_dir    = osp.join(self.rgb_root, 'test')

        # --- Parsing config ---
        self.proc_raw_dir  = osp.join(self.rgb_root, 'processed')
        self.proc_hair_dir = osp.join(self.rgb_root, 'processed_hair')
        self.with_parsing  = kwargs.get('with_parsing', True)     # 默认开启
        self.parsing_ext   = kwargs.get('parsing_ext', '.png')    # 你的解析图扩展名（.png/.npy/.mat）

        self._check_before_run()

        # Train (with hair)
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, train_stats = \
            self._process_dir_train_with_hair(self.train_dir, self.hair_dir)

        # Val (raw only)
        val, num_val_pids, num_val_imgs, num_val_clothes, _, val_stats = \
            self._process_dir_train_with_hair(self.val_dir, None)

        # Test (unchanged)
        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> PRCC loaded (hair-augmented TRAIN enabled)")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids,   num_val_imgs,   num_val_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids,  num_test_imgs,  num_test_clothes))
        logger.info("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        logger.info("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        logger.info("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")
        logger.info("  train RAW images      | {:8d}".format(train_stats['raw']))
        logger.info("  train HAIR images     | {:8d}".format(train_stats['hair']))
        logger.info("  val   RAW images      | {:8d}".format(val_stats['raw']))
        if self.with_parsing:
            logger.info("  --------------------------------------------")
            logger.info("  parsing on (ext={})".format(self.parsing_ext))
            logger.info("  train RAW parse hit   | {:8d}".format(train_stats['parse_raw_found']))
            logger.info("  train RAW parse miss  | {:8d}".format(train_stats['parse_raw_miss']))
            logger.info("  train HAIR parse hit  | {:8d}".format(train_stats['parse_hair_found']))
            logger.info("  train HAIR parse miss | {:8d}".format(train_stats['parse_hair_miss']))
            logger.info("  val   RAW parse hit   | {:8d}".format(val_stats['parse_raw_found']))
            logger.info("  val   RAW parse miss  | {:8d}".format(val_stats['parse_raw_miss']))
        logger.info("  --------------------------------------------")

        self.train = train
        self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx

        self.train_stats = train_stats
        self.val_stats = val_stats

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        # hair dir optional
        if not osp.exists(self.hair_dir):
            logging.getLogger('reid.dataset').warning(
                "PRCC hair dir not found: '{}'; TRAIN will use RAW only.".format(self.hair_dir)
            )
        # parsing dirs optional
        for d in [self.proc_raw_dir, self.proc_hair_dir]:
            if not osp.exists(d):
                logging.getLogger('reid.dataset').warning(
                    "Parsing dir not found: {} (parse_path will be None)".format(d)
                )

    @staticmethod
    def _infer_cam_from_name(filename):
        """Find the first 'A'/'B'/'C' in filename."""
        for ch in filename:
            if ch in ('A', 'B', 'C'):
                return ch
        raise ValueError(f"Cannot infer camera from filename '{filename}' (need A/B/C in name)")

    def _collect_pid_dirs(self, dir_path):
        if not dir_path or not osp.exists(dir_path):
            return []
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs = [p for p in pdirs if osp.isdir(p)]
        pdirs.sort()
        return pdirs

    # --------- parsing helpers ----------
    def _swap_ext(self, fname, new_ext=None):
        if new_ext is None:
            new_ext = self.parsing_ext
        stem, _ = osp.splitext(fname)
        return stem + new_ext

    def _resolve_parse_path(self, is_hair, pid_dir, fname):
        """
        is_hair=False -> processed/
        is_hair=True  -> processed_hair/
        Return parse path if exists, else None.
        """
        pid = osp.basename(pid_dir)
        proc_root = self.proc_hair_dir if is_hair else self.proc_raw_dir
        if not proc_root or not osp.exists(proc_root):
            return None
        parse_fname = self._swap_ext(fname, self.parsing_ext)  # e.g., A0001.png / h1_A0001.png
        cand = osp.join(proc_root, pid, parse_fname)
        return cand if osp.exists(cand) else None
    # -----------------------------------

    def _process_dir_train_with_hair(self, raw_dir, hair_dir):
        """
        Merge RAW (required) with optional HAIR images.
        Returns:
          dataset(list), num_pids, num_imgs, num_clothes, pid2clothes(np.array),
          stats(dict: {'raw','hair','parse_*'})
        """
        raw_pdirs  = self._collect_pid_dirs(raw_dir)
        hair_pdirs = self._collect_pid_dirs(hair_dir) if hair_dir and osp.exists(hair_dir) else []

        pid_container = set()
        clothes_container = set()

        # Collect from RAW
        for pdir in raw_pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            for img_dir in glob.glob(osp.join(pdir, '*.jpg')):
                cam = self._infer_cam_from_name(osp.basename(img_dir))
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir) + cam)

        # Collect from HAIR (train only)
        for pdir in hair_pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            for img_dir in glob.glob(osp.join(pdir, '*.jpg')):
                cam = self._infer_cam_from_name(osp.basename(img_dir))
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir) + cam)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {cid: label for label, cid in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes), dtype=np.float32)
        raw_count, hair_count = 0, 0
        # parsing hit/miss stats
        parse_raw_found = parse_raw_miss = 0
        parse_hair_found = parse_hair_miss = 0

        def add_images(pdirs_root, is_hair=False):
            nonlocal raw_count, hair_count
            nonlocal parse_raw_found, parse_raw_miss, parse_hair_found, parse_hair_miss

            for pdir in pdirs_root:
                pid = int(osp.basename(pdir))
                label = pid2label[pid]
                for img_dir in glob.glob(osp.join(pdir, '*.jpg')):
                    fname = osp.basename(img_dir)
                    cam = self._infer_cam_from_name(fname)
                    camid = cam2label[cam]
                    if cam in ['A', 'B']:
                        clothes_id = clothes2label[osp.basename(pdir)]
                    else:
                        clothes_id = clothes2label[osp.basename(pdir) + cam]

                    parse_path = None
                    if self.with_parsing:
                        parse_path = self._resolve_parse_path(is_hair, pdir, fname)
                        if parse_path is not None:
                            if is_hair: parse_hair_found += 1
                            else:       parse_raw_found  += 1
                        else:
                            if is_hair: parse_hair_miss  += 1
                            else:       parse_raw_miss   += 1

                    if self.with_parsing:
                        dataset.append((img_dir, label, camid, clothes_id, parse_path))
                    else:
                        dataset.append((img_dir, label, camid, clothes_id))

                    pid2clothes[label, clothes_id] = 1.0
                    if is_hair: hair_count += 1
                    else:       raw_count  += 1

        add_images(raw_pdirs, is_hair=False)
        if hair_pdirs:
            add_images(hair_pdirs, is_hair=True)

        num_imgs = len(dataset)
        stats = {
            "raw": raw_count,
            "hair": hair_count,
            "parse_raw_found": parse_raw_found,
            "parse_raw_miss":  parse_raw_miss,
            "parse_hair_found": parse_hair_found,
            "parse_hair_miss":  parse_hair_miss,
        }
        return dataset, num_pids, num_imgs, num_clothes, pid2clothes, stats

    def _process_dir_test(self, test_path):
        # Build pid list from A
        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []

        for cam in ['A', 'B', 'C']:
            for pdir in glob.glob(osp.join(test_path, cam, '*')):
                pid = int(osp.basename(pdir))
                for img_dir in glob.glob(osp.join(pdir, '*.jpg')):
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        gallery_dataset.append((img_dir, pid, camid, clothes_id))
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        query_dataset_same_clothes.append((img_dir, pid, camid, clothes_id))
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        query_dataset_diff_clothes.append((img_dir, pid, camid, clothes_id))

        # Build gallery index (10-shot)
        pid2imgidx = {}
        for idx, (_, pid, _, _) in enumerate(gallery_dataset):
            pid2imgidx.setdefault(pid, []).append(idx)

        gallery_idx = {}
        random.seed(3)
        for idx in range(10):
            gallery_idx[idx] = [random.choice(pid2imgidx[pid]) for pid in pid2imgidx]

        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return (query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset,
                num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery,
                num_clothes, gallery_idx)
