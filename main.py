# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist
from apex import amp

from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from train_save import train_cal, train_cal_with_memory
from test import test, test_prcc



VID_DATASET = ['ccvid']

import ast

def str2list(v):
    """
    把字符串形式的 list 转换成 Python list[float]
    例如:
    "[0.1,0.3]" -> [0.1, 0.3]
    "0.1,0.3"   -> [0.1, 0.3]
    """
    if isinstance(v, list):
        return v
    try:
        # 尝试用 ast.literal_eval 解析安全的 python 表达式
        val = ast.literal_eval(v)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        else:
            return [float(val)]
    except Exception:
        return [float(x) for x in v.replace('[','').replace(']','').split(',')]

# def parse_option():
#     parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
#     parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
#     # Datasets
#     parser.add_argument('--root', type=str, help="your root path to data directory")
#     parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
#     # Miscs
#     parser.add_argument('--output', type=str, help="your output path to save model and logs")
#     parser.add_argument('--resume', type=str, metavar='PATH')
#     parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
#     parser.add_argument('--eval', action='store_true', help="evaluation only")
#     parser.add_argument('--tag', type=str, help='tag for log file')
#     parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
#     args, unparsed = parser.parse_known_args()
#     if args.dataset in VID_DATASET:
#         config = get_vid_config(args)
#     else:
#         config = get_img_config(args)

#     return config

def parse_option():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss'
    )
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc',
                        help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    # ⚠️ 必须加这一行，否则 torch.distributed.launch 会报错
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

    # ⚠️ opts：用来支持 yacs 参数覆盖（必须放最后）
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()


    # 根据 dataset 选择配置
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    # 覆盖配置
    if args.opts is not None:
        config.merge_from_list(args.opts)
    if args.opts is not None:
        # 去掉可能的 \r
        args.opts = [x.strip() for x in args.opts]
        config.merge_from_list(args.opts)

    return config, args



def main(config):
    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    # Define a matrix pid2clothes with shape (num_pids, num_clothes). 
    # pid2clothes[i, j] = 1 when j-th clothes belongs to i-th identity. Otherwise, pid2clothes[i, j] = 0.
    pid2clothes = torch.from_numpy(dataset.pid2clothes)

    # Build model
    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if config.LOSS.CAL == 'calwithmemory':
            criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        else:
            clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    local_rank = dist.get_rank()
    model = model.cuda(local_rank)
    classifier = classifier.cuda(local_rank)
    if config.LOSS.CAL == 'calwithmemory':
        criterion_adv = criterion_adv.cuda(local_rank)
    else:
        clothes_classifier = clothes_classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)

    if config.TRAIN.AMP:
        [model, classifier], optimizer = amp.initialize([model, classifier], optimizer, opt_level="O1")
        if config.LOSS.CAL != 'calwithmemory':
            clothes_classifier, optimizer_cc = amp.initialize(clothes_classifier, optimizer_cc, opt_level="O1")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    if config.LOSS.CAL != 'calwithmemory':
        clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier, device_ids=[local_rank], output_device=local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        if config.LOSS.CAL == 'calwithmemory':
            train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
                criterion_adv, optimizer, trainloader, pid2clothes)
        else:
            train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
                criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset, save_att = True)
            else:
                rank1 = test(config, model, queryloader, galleryloader, dataset, save_att = True)
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            model_state_dict = model.module.state_dict()
            classifier_state_dict = classifier.module.state_dict()
            if config.LOSS.CAL == 'calwithmemory':
                clothes_classifier_state_dict = criterion_adv.state_dict()
            else:
                clothes_classifier_state_dict = clothes_classifier.module.state_dict()
            if local_rank == 0:
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    'classifier_state_dict': classifier_state_dict,
                    'clothes_classifier_state_dict': clothes_classifier_state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    

if __name__ == '__main__':
    # config = parse_option()
    config, args = parse_option()
    # Set GPU
    print("opts:", args.opts)
    print("ERASE_KEEP_RANGE:", config.LOSS.ERASE_KEEP_RANGE)
    print("ADV_WEIGHT:", config.LOSS.ADV_WEIGHT)
    print("ATT_WEIGHT:", config.LOSS.ATT_WEIGHT)
    print("ERASE_PROB:", config.LOSS.ERASE_PROB)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.txt')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.ltxt')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config)
# import os
# import sys
# import time
# import datetime
# import argparse
# import logging
# import os.path as osp
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch import distributed as dist

# from configs.default_img import get_img_config
# from configs.default_vid import get_vid_config
# from data import build_dataloader
# from models import build_model
# from losses import build_losses
# from tools.utils import save_checkpoint, set_seed, get_logger
# from train import train_cal, train_cal_with_memory
# from test import test, test_prcc

# VID_DATASET = ['ccvid']

# import ast

# def str2list(v):
#     """
#     把字符串形式的 list 转换成 Python list[float]
#     例如:
#     "[0.1,0.3]" -> [0.1, 0.3]
#     "0.1,0.3"   -> [0.1, 0.3]
#     """
#     if isinstance(v, list):
#         return v
#     try:
#         val = ast.literal_eval(v)
#         if isinstance(val, (list, tuple)):
#             return [float(x) for x in val]
#         else:
#             return [float(val)]
#     except Exception:
#         return [float(x) for x in v.replace('[','').replace(']','').split(',')]

# def parse_option():
#     parser = argparse.ArgumentParser(
#         description='Train clothes-changing re-id model with clothes-based adversarial loss'
#     )
#     parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
#     # Datasets
#     parser.add_argument('--root', type=str, help="your root path to data directory")
#     parser.add_argument('--dataset', type=str, default='ltcc',
#                         help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
#     # Miscs
#     parser.add_argument('--output', type=str, help="your output path to save model and logs")
#     parser.add_argument('--resume', type=str, metavar='PATH')
#     parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
#     parser.add_argument('--eval', action='store_true', help="evaluation only")
#     parser.add_argument('--tag', type=str, help='tag for log file')
#     parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

#     # 分布式必需
#     parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

#     # yacs 覆盖（必须放最后）
#     parser.add_argument(
#         "opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER
#     )

#     args = parser.parse_args()

#     # 根据 dataset 选择配置
#     if args.dataset in VID_DATASET:
#         config = get_vid_config(args)
#     else:
#         config = get_img_config(args)

#     # 覆盖配置（去掉可能的 \r）
#     if args.opts is not None:
#         args.opts = [x.strip() for x in args.opts]
#         config.merge_from_list(args.opts)

#     return config, args


# def main(config, args):
#     # Build dataloader
#     if config.DATA.DATASET == 'prcc':
#         trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)
#     else:
#         trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)

#     # pid2clothes: (num_pids, num_clothes)
#     pid2clothes = torch.from_numpy(dataset.pid2clothes)

#     # Build model & losses
#     model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
#     criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)

#     # Optimizers
#     parameters = list(model.parameters()) + list(classifier.parameters())
#     if config.TRAIN.OPTIMIZER.NAME == 'adam':
#         optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR,
#                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
#         optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR,
#                                   weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
#     elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
#         optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR,
#                                 weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
#         optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR,
#                                    weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
#     elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
#         optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9,
#                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
#         optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9,
#                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
#     else:
#         raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))

#     scheduler = lr_scheduler.MultiStepLR(
#         optimizer,
#         milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
#         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE
#     )

#     # Resume
#     start_epoch = config.TRAIN.START_EPOCH
#     if config.MODEL.RESUME:
#         logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
#         checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
#         model.load_state_dict(checkpoint['model_state_dict'])
#         classifier.load_state_dict(checkpoint['classifier_state_dict'])
#         if config.LOSS.CAL == 'calwithmemory':
#             criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
#         else:
#             clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
#         start_epoch = checkpoint['epoch']

#     # ==== rank & device ====
#     # 优先使用环境变量 LOCAL_RANK；回退到 args.local_rank 或 dist.get_rank()
#     try:
#         local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
#     except Exception:
#         local_rank = dist.get_rank()
#     torch.cuda.set_device(local_rank)

#     # to cuda
#     model = model.cuda(local_rank)
#     classifier = classifier.cuda(local_rank)
#     if config.LOSS.CAL == 'calwithmemory':
#         criterion_adv = criterion_adv.cuda(local_rank)
#     else:
#         clothes_classifier = clothes_classifier.cuda(local_rank)

#     # DDP wrap
#     model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
#     classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
#     if config.LOSS.CAL != 'calwithmemory':
#         clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier, device_ids=[local_rank],
#                                                                  output_device=local_rank)

#     if config.EVAL_MODE:
#         logger.info("Evaluate only")
#         with torch.no_grad():
#             if config.DATA.DATASET == 'prcc':
#                 test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
#             else:
#                 test(config, model, queryloader, galleryloader, dataset)
#         return

#     start_time = time.time()
#     train_time = 0
#     best_rank1 = -np.inf
#     best_epoch = 0
#     logger.info("==> Start training")

#     for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
#         train_sampler.set_epoch(epoch)
#         start_train_time = time.time()

#         if config.LOSS.CAL == 'calwithmemory':
#             train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair,
#                                   criterion_adv, optimizer, trainloader, pid2clothes)
#         else:
#             train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair,
#                       criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)
#         train_time += round(time.time() - start_train_time)

#         # eval & save
#         if ((epoch + 1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and
#             (epoch + 1) % config.TEST.EVAL_STEP == 0) or ((epoch + 1) == config.TRAIN.MAX_EPOCH):
#             logger.info("==> Test")
#             torch.cuda.empty_cache()
#             if config.DATA.DATASET == 'prcc':
#                 rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset, save_att=True)
#             else:
#                 rank1 = test(config, model, queryloader, galleryloader, dataset, save_att=True)
#             torch.cuda.empty_cache()

#             is_best = rank1 > best_rank1
#             if is_best:
#                 best_rank1 = rank1
#                 best_epoch = epoch + 1

#             model_state_dict = model.module.state_dict()
#             classifier_state_dict = classifier.module.state_dict()
#             if config.LOSS.CAL == 'calwithmemory':
#                 clothes_classifier_state_dict = criterion_adv.state_dict()
#             else:
#                 clothes_classifier_state_dict = clothes_classifier.module.state_dict()
#             if local_rank == 0:
#                 save_checkpoint({
#                     'model_state_dict': model_state_dict,
#                     'classifier_state_dict': classifier_state_dict,
#                     'clothes_classifier_state_dict': clothes_classifier_state_dict,
#                     'rank1': rank1,
#                     'epoch': epoch,
#                 }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

#         scheduler.step()

#     logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

#     elapsed = round(time.time() - start_time)
#     elapsed = str(datetime.timedelta(seconds=elapsed))
#     train_time = str(datetime.timedelta(seconds=train_time))
#     logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


# if __name__ == '__main__':
#     config, args = parse_option()

#     print("opts:", args.opts)
#     print("ERASE_KEEP_RANGE:", config.LOSS.ERASE_KEEP_RANGE)
#     print("ADV_WEIGHT:", config.LOSS.ADV_WEIGHT)
#     print("ATT_WEIGHT:", config.LOSS.ATT_WEIGHT)
#     print("ERASE_PROB:", config.LOSS.ERASE_PROB)

#     # Set GPU (必须在任何 CUDA/Dist 初始化前)
#     os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

#     # Init dist
#     dist.init_process_group(backend="nccl", init_method='env://')

#     # Set random seed
#     # 这里用 dist.get_rank() 参与到 seed 中（亦可用 LOCAL_RANK）
#     local_rank_for_seed = dist.get_rank()
#     set_seed(config.SEED + local_rank_for_seed)

#     # get logger
#     if not config.EVAL_MODE:
#         output_file = osp.join(config.OUTPUT, 'log_train_.txt')
#     else:
#         output_file = osp.join(config.OUTPUT, 'log_test.ltxt')
#     logger = get_logger(output_file, local_rank_for_seed, 'reid')
#     logger.info("Config:\n-----------------------------------------")
#     logger.info(config)
#     logger.info("-----------------------------------------")

#     main(config, args)
