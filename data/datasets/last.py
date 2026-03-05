import os
import glob
import logging
import numpy as np
import os.path as osp
import re

class LaST(object):
    """ LaST """

    dataset_dir = "last"
    def __init__(self, root='data', **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.hair_dir = osp.join(self.dataset_dir, 'hair')   # 头发图像目录
        self.val_query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.test_query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        self.proc_raw_dir  = osp.join(self.dataset_dir, 'processed')  # 原始图像的解析目录
        self.proc_hair_dir = osp.join(self.dataset_dir, 'processed_hair')  # 头发图像的解析目录

        self._check_before_run()

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(self.train_dir)

        # 处理训练集（混合了 train 和 hair 图像）
        # train, num_train_pids = self._process_dir(self.train_dir, None, pid2label=pid2label, clothes2label=clothes2label, relabel=True, is_train=True)
        train, num_train_pids = self._process_dir(self.train_dir, self.hair_dir, pid2label=pid2label, clothes2label=clothes2label, relabel=True, is_train=True)

        # 处理验证集和测试集（只有 raw 图像）
        val_query, num_val_query_pids = self._process_dir(self.val_query_dir, None, relabel=False)
        val_gallery, num_val_gallery_pids = self._process_dir(self.val_gallery_dir, None, relabel=False, recam=len(val_query))
        test_query, num_test_query_pids = self._process_dir(self.test_query_dir, None, relabel=False)
        test_gallery, num_test_gallery_pids = self._process_dir(self.test_gallery_dir, None, relabel=False, recam=len(test_query))

        num_total_pids = num_train_pids + num_val_gallery_pids + num_test_gallery_pids
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> LaST loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset        | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train         | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), len(clothes2label)))
        logger.info("  query(val)    | {:5d} | {:8d} |".format(num_val_query_pids, len(val_query)))
        logger.info("  gallery(val)  | {:5d} | {:8d} |".format(num_val_gallery_pids, len(val_gallery)))
        logger.info("  query         | {:5d} | {:8d} |".format(num_test_query_pids, len(test_query)))
        logger.info("  gallery       | {:5d} | {:8d} |".format(num_test_gallery_pids, len(test_gallery)))
        logger.info("  --------------------------------------------")
        logger.info("  total         | {:5d} | {:8d} | ".format(num_total_pids, num_total_imgs))
        logger.info("  --------------------------------------------")

        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = len(clothes2label)
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """检查数据集是否准备好"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.val_query_dir):
            raise RuntimeError(f"'{self.val_query_dir}' is not available")
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError(f"'{self.val_gallery_dir}' is not available")
        if not osp.exists(self.test_query_dir):
            raise RuntimeError(f"'{self.test_query_dir}' is not available")
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError(f"'{self.test_gallery_dir}' is not available")

    def get_pid2label_and_clothes2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))  # 支持 .jpg 图像
        img_paths.sort()

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes: label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _process_dir(self, dir_path, hair_dir=None, pid2label=None, clothes2label=None, relabel=False, recam=0, is_train=False):

        """处理每个目录，返回 dataset"""
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))  # 支持 .jpg 图像
            hair_paths = glob.glob(osp.join(hair_dir, '*/*.jpg')) if hair_dir else []  # 处理 hair 图像
            img_paths += hair_paths  # 将 hair 图像与 train 图像混合
        img_paths.sort()
        # 如果提供了 hair_dir（头发图像目录），我们需要分别处理头发图像和原始图像
        # img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))  # 支持 .jpg 图像
        # hair_paths = glob.glob(osp.join(hair_dir, '*/*.jpg')) if hair_dir else []  # 处理 hair 图像
        # img_paths += hair_paths  # 将 hair 图像与 train 图像混合

        # img_paths.sort()

        dataset = []
        pid_container = set()
        parse_raw_found = parse_raw_miss = 0
        parse_hair_found = parse_hair_miss = 0
        for ii, img_path in enumerate(img_paths):
            names = osp.basename(img_path).split('.')[0].split('_')
            # print(names)

            # 判断文件名是否以 'h1_', 'h2_', 'h3_' 等前缀开头
            if names[0].startswith('h') and names[0][1].isdigit():
                # 如果是头发图像，去除 'h1_'，只保留数字部分
                hid = names[0]
                names = names[1:]
                pid_str = names[0]  # 去掉 'h' 和数字部分，保留后面的部分（即 `24234745_34573954`）
                # print(pid_str)
                is_hair = True
            else:
                # 否则，提取数字作为 pid
                pid_str = names[0]
                is_hair = False

            pid = int(pid_str) if pid_str.isdigit() else -1  # 如果没有有效数字，设置为 -1
            pid_container.add(pid)

            clothes = names[0] + '_' + names[-1]
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            if relabel and clothes2label is not None:
                clothes_id = clothes2label[clothes]
            else:
                clothes_id = pid

            # 判断图像类型（是否为头发图像）
            is_hair = self.is_hair(img_path, hair_dir)

            # 获取解析图像路径
            parse_path = self._resolve_parse_path(is_hair, osp.dirname(img_path), osp.basename(img_path))

            # 统计解析图像命中情况
            if is_hair:
                if parse_path:
                    parse_hair_found += 1
                else:
                    parse_hair_miss += 1
            else:
                if parse_path:
                    parse_raw_found += 1
                else:
                    parse_raw_miss += 1

            dataset.append((img_path, pid, camid, clothes_id, parse_path))

        num_pids = len(pid_container)

        # 输出统计信息
        logger = logging.getLogger('reid.dataset')
        logger.info(f"RAW parse hit: {parse_raw_found}, RAW parse miss: {parse_raw_miss}")
        logger.info(f"HAIR parse hit: {parse_hair_found}, HAIR parse miss: {parse_hair_miss}")

        return dataset, num_pids

    def is_hair(self, img_path, hair_dir):
        """判断图像是否为头发图像"""
        return hair_dir and img_path.startswith(hair_dir)

    def _resolve_parse_path(self, is_hair, pid_dir, fname):
        """解析路径的辅助函数"""
        proc_root = self.proc_hair_dir if is_hair else self.proc_raw_dir
        parse_fname = self._swap_ext(fname)  # .png 扩展名
        parse_path = osp.join(proc_root, osp.basename(pid_dir), parse_fname)
        return parse_path if osp.exists(parse_path) else None

    def _swap_ext(self, fname, new_ext=".png"):
        """更改文件扩展名"""
        stem, _ = osp.splitext(fname)
        return stem + new_ext
