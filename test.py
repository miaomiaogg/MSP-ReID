import time
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
import cv2

VID_DATASET = ['ccvid']


def concat_all_gather(tensors, num_total_examples):
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        outputs.append(output[:num_total_examples])
    return outputs


# ---------- 兼容 batch 格式 ----------
def _unpack_eval_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError(f"Unexpected eval batch format: {type(batch)}, len={len(batch) if isinstance(batch,(list,tuple)) else 'NA'}")


# ---------- 保存注意力 ----------
def save_attention_map(model, imgs, save_dir, prefix="att", max_save=10):
    """
    从 model.fmap 提取注意力，叠加到原图，保存到 save_dir
    """
    if not hasattr(model, "fmap") or model.fmap is None:
        return
    print(f"[DEBUG] fmap shape: {inner_model.fmap.shape}, save_dir={save_dir}")
    fmap = model.fmap.detach().cpu()  # [B,C,H,W]
    B, C, H, W = fmap.shape
    att = fmap.pow(2).mean(1)  # [B,H,W]
    att = att - att.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    att = att / (att.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-5)

    os.makedirs(save_dir, exist_ok=True)

    imgs_np = (imgs.cpu().numpy().transpose(0,2,3,1) * 255).astype(np.uint8)

    for i in range(min(B, max_save)):
        heatmap = cv2.applyColorMap(np.uint8(att[i].numpy()*255), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (imgs_np[i].shape[1], imgs_np[i].shape[0]))
        overlay = cv2.addWeighted(imgs_np[i], 0.5, heatmap, 0.5, 0)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{i}.jpg"), overlay[:,:,::-1])


@torch.no_grad()
def extract_img_feature(model, dataloader, save_att=False, save_dir="att_vis"):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, batch in enumerate(dataloader):
        imgs, batch_pids, batch_camids, batch_clothes_ids = _unpack_eval_batch(batch)

        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(non_blocking=True), flip_imgs.cuda(non_blocking=True)

        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features = batch_features + batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        # 可视化注意力
        if save_att and batch_idx < 5:  # 只存前几个 batch
            save_attention_map(model, imgs, save_dir, prefix=f"batch{batch_idx}")

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids


@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length):
    logger = logging.getLogger('reid.test')
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, batch in enumerate(dataloader):
        vids, batch_pids, batch_camids, batch_clothes_ids = _unpack_eval_batch(batch)
        if (batch_idx + 1) % 200 == 0:
            logger.info("{}/{}".format(batch_idx + 1, len(dataloader)))
        vids = vids.cuda(non_blocking=True)
        batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0]: idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids


def test(config, model, queryloader, galleryloader, dataset, save_att=False):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()

    if config.DATA.DATASET in VID_DATASET:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader,
                                                                  dataset.query_vid2clip_index,
                                                                  len(dataset.recombined_query))
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader,
                                                                  dataset.gallery_vid2clip_index,
                                                                  len(dataset.recombined_gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(model, queryloader, save_att, save_dir="att_vis/query")
        gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(model, galleryloader, save_att, save_dir="att_vis/gallery")
        torch.cuda.empty_cache()
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()

    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    if config.DATA.DATASET in ['last', 'deepchange', 'vcclothes_sc', 'vcclothes_cc']:
        return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    logger.info("Results (Same Clothes) top1:{:.1%} mAP:{:.1%}".format(cmc[0], mAP))

    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    logger.info("Results (Clothes Changing) top1:{:.1%} mAP:{:.1%}".format(cmc[0], mAP))

    return cmc[0]

def test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset, save_att=False):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()

    # 提取特征（加注意力可视化）
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(
        model, queryloader_same, save_att, save_dir="att_vis/query_same")
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(
        model, queryloader_diff, save_att, save_dir="att_vis/query_diff")
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(
        model, galleryloader, save_att, save_dir="att_vis/gallery")

    # 多卡 all_gather
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather(
        [qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather(
        [qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather(
        [gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set (same clothes), shape={}".format(qsf.shape))
    logger.info("Extracted features for query set (diff clothes), shape={}".format(qdf.shape))
    logger.info("Extracted features for gallery set, shape={}".format(gf.shape))
    logger.info('Feature extraction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 距离矩阵
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()

    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()

    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    # same clothes
    logger.info("Computing CMC/mAP for same clothes")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results [Same Clothes] top1:{:.1%} top5:{:.1%} mAP:{:.1%}".format(cmc[0], cmc[4], mAP))

    # diff clothes
    logger.info("Computing CMC/mAP for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results [Clothes Changing] top1:{:.1%} top5:{:.1%} mAP:{:.1%}".format(cmc[0], cmc[4], mAP))

    return cmc[0]
