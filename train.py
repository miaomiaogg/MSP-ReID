import time
import logging
import random
import torch
import numpy as np
from PIL import Image
from apex import amp

# from apex import amp
from tools.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms.functional as tvF
import torch.nn.functional as F


# ==========================
# 批处理兼容：4元组 / 5元组
# ==========================
def _unpack_train_batch(batch):
    if len(batch) == 4:
        imgs, pids, camids, clothes_ids = batch
        parse_paths = None
        masks = None
    elif len(batch) == 5:
        imgs, pids, camids, clothes_ids, parse_paths = batch
        masks = None
    elif len(batch) == 8:
        imgs, pids, camids, clothes_ids, M_hair, M_face, M_limbs, M_clothes = batch
        parse_paths = None
        masks = (M_hair, M_face, M_limbs, M_clothes)
    else:
        raise ValueError(f"Unexpected train batch size: {len(batch)} (expected 4/5/8)")
    return imgs, pids, camids, clothes_ids, parse_paths, masks


# ==========================
# 解析图 / 掩码 工具（用于 ERASE）
# ==========================
LIP_CLOTHES = [5, 6, 7, 9, 10, 12]  # 如需更广可含: 3,8,11,18,19

def load_index_mask(parse_path):
    if not parse_path:
        return None
    ext = parse_path.split('.')[-1].lower()
    try:
        if ext == 'png':
            return np.array(Image.open(parse_path), dtype=np.uint8)
        elif ext == 'npy':
            arr = np.load(parse_path)
            return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        else:
            return None
    except Exception:
        return None

def binmask(index_mask, cls_list):
    if index_mask is None: return None
    return np.isin(index_mask, np.array(cls_list, dtype=np.uint8))

def resize_mask_to_tensor(bin_mask_np, H, W):
    if bin_mask_np is None:
        return torch.zeros((1, H, W), dtype=torch.float32)
    m = Image.fromarray((bin_mask_np.astype(np.uint8) * 255), mode='L').resize((W, H), Image.NEAREST)
    t = tvF.to_tensor(m)
    return (t > 0.5).float()


# ==========================
# ERASE 视图
# ==========================
def make_erase_images(imgs, parse_paths, keep_ratio_range=(0.1, 0.3), dilate_px=1):
    B, C, H, W = imgs.shape
    device = imgs.device
    dtype  = imgs.dtype

    op_masks = []
    for i in range(B):
        pmask = load_index_mask(parse_paths[i]) if (parse_paths is not None) else None
        m_clothes_np = binmask(pmask, LIP_CLOTHES)
        M_clothes = resize_mask_to_tensor(m_clothes_np, H, W).to(device)

        if dilate_px > 0 and torch.count_nonzero(M_clothes) > 0:
            for _ in range(dilate_px):
                M_clothes = F.max_pool2d(M_clothes, kernel_size=3, stride=1, padding=1)
                M_clothes = (M_clothes > 0.5).float()

        keep_ratio = float(np.random.uniform(*keep_ratio_range))
        if keep_ratio > 0 and torch.count_nonzero(M_clothes) > 0:
            rand = torch.rand_like(M_clothes)
            keep_mask = (rand < keep_ratio).float() * M_clothes
            op_mask = (M_clothes - keep_mask).clamp(min=0.0, max=1.0)
        else:
            op_mask = M_clothes

        op_masks.append(op_mask)

    op_masks = torch.stack(op_masks, dim=0).to(dtype=dtype)
    op_masks_3 = op_masks.expand(B, C, H, W)
    erased = imgs * (1.0 - op_masks_3)
    return erased


# ==========================
# 注意力约束损失
# ==========================
def build_attention_loss(features, M_face, M_limbs, M_hair, lambda_neg=1.0):
    """
    features: [B,C,H,W]
    M_face, M_limbs, M_hair: [B,1,H,W] float
    """
    eps = 1e-6
    A = features.pow(2).mean(1, keepdim=True)  # [B,1,H,W]
    A = (A - A.amin(dim=(2,3), keepdim=True)) / \
        (A.amax(dim=(2,3), keepdim=True) - A.amin(dim=(2,3), keepdim=True) + eps)

    M_pos = torch.clamp(M_face + M_limbs, 0, 1)
    M_neg = M_hair

    L_pos = 1.0 - ((A * M_pos).sum(dim=(2,3)) / (M_pos.sum(dim=(2,3)) + eps)).mean()
    L_neg = ((A * M_neg).sum(dim=(2,3)) / (M_neg.sum(dim=(2,3)) + eps)).mean()

    return L_pos + lambda_neg * L_neg


# =========================================
# 主训练：带衣服判别器 / ERASE + 注意力约束 + 梯度累计
# =========================================
def train_cal(config, epoch, model, classifier, clothes_classifier,
              criterion_cla, criterion_pair, criterion_clothes, criterion_adv,
              optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    meter_cla, meter_pair, meter_clo, meter_adv = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    meter_acc, meter_clo_acc = AverageMeter(), AverageMeter()
    meter_time, meter_data = AverageMeter(), AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    erase_prob       = getattr(config.LOSS, "ERASE_PROB", 0.5)
    erase_keep_range = tuple(getattr(config.LOSS, "ERASE_KEEP_RANGE", (0.1, 0.3)))
    erase_dilate_px  = int(getattr(config.LOSS, "ERASE_DILATE_PX", 1))
    adv_weight       = getattr(config.LOSS, "ADV_WEIGHT", 1.0)
    att_weight       = getattr(config.LOSS, "ATT_WEIGHT", 1.0)

    logger.info(f"[ERASE] keep_ratio_range={erase_keep_range}, dilate_px={erase_dilate_px}, erase_prob={erase_prob}")
    logger.info(f"[LOSS WEIGHTS] pair={config.LOSS.PAIR_LOSS_WEIGHT}, adv={adv_weight}, att={att_weight}")

    end = time.time()

    for batch_idx, batch in enumerate(trainloader):
        imgs, pids, camids, clothes_ids, parse_paths, masks = _unpack_train_batch(batch)
        apply_erase = (parse_paths is not None) and (random.random() < erase_prob)

        pos_mask_np = pid2clothes[pids]
        imgs, pids, clothes_ids = imgs.cuda(), pids.cuda(), clothes_ids.cuda()
        pos_mask = torch.as_tensor(pos_mask_np, dtype=torch.float32, device=imgs.device)

        if apply_erase:
            erased_imgs = make_erase_images(
                imgs, parse_paths,
                keep_ratio_range=erase_keep_range,
                dilate_px=erase_dilate_px
            )
        else:
            erased_imgs = None

        meter_data.update(time.time() - end)

        # 前向
        B = imgs.size(0)
        if apply_erase:
            inputs_cat = torch.cat([imgs, erased_imgs], dim=0)
            feats_cat  = model(inputs_cat)
            outs_cat   = classifier(feats_cat)
            features_raw, features_erase = feats_cat[:B], feats_cat[B:]
            outputs_raw,  outputs_erase  = outs_cat[:B],  outs_cat[B:]
        else:
            features_raw = model(imgs)
            outputs_raw  = classifier(features_raw)
            features_erase, outputs_erase = None, None

        with torch.no_grad():
            preds = outputs_raw.detach().argmax(1)

        # 衣服分类器（判别器）分支
        pred_clothes_raw = clothes_classifier(features_raw.detach())
        clothes_loss = criterion_clothes(pred_clothes_raw, clothes_ids)

        new_pred_clothes = clothes_classifier(features_raw)
        with torch.no_grad():
            clothes_preds = new_pred_clothes.detach().argmax(1)

        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask) \
            if epoch >= config.TRAIN.START_EPOCH_ADV else torch.tensor(0.0, device=imgs.device)

        # ID 与 pair loss
        cla_loss_raw  = criterion_cla(outputs_raw,   pids)
        pair_loss_raw = criterion_pair(features_raw, pids)

        if apply_erase:
            cla_loss_erase  = criterion_cla(outputs_erase,  pids)
            pair_loss_erase = criterion_pair(features_erase, pids)
            cla_loss  = 0.5 * (cla_loss_raw  + cla_loss_erase)
            pair_loss = 0.5 * (pair_loss_raw + pair_loss_erase)
        else:
            cla_loss  = cla_loss_raw
            pair_loss = pair_loss_raw

        # 注意力损失
        if masks is not None:
            M_hair, M_face, M_limbs, _ = [m.cuda() for m in masks]
            att_loss = build_attention_loss(features_raw, M_face, M_limbs, M_hair)
        else:
            att_loss = torch.tensor(0.0, device=imgs.device)

        # 总损失（主干）
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss_main = cla_loss \
                      + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss \
                      + adv_weight * adv_loss \
                      + att_weight * att_loss
        else:
            loss_main = cla_loss \
                      + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss \
                      + att_weight * att_loss

        # ========== 不用梯度累计：每 iter 更新 ==========
        optimizer.zero_grad()
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()

        if config.TRAIN.AMP:
            with amp.scale_loss(loss_main, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_main.backward()

        if epoch >= config.TRAIN.START_EPOCH_CC:
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()

        optimizer.step()
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.step()

        # 统计
        with torch.no_grad():
            meter_acc.update((preds == pids).float().mean(), pids.size(0))
            meter_clo_acc.update((clothes_preds == clothes_ids).float().mean(), clothes_ids.size(0))

        meter_cla.update(cla_loss.item(), pids.size(0))
        meter_pair.update(pair_loss.item(), pids.size(0))
        meter_clo.update(clothes_loss.item(), clothes_ids.size(0))
        meter_adv.update(adv_loss.item(), clothes_ids.size(0))

        meter_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{t.sum:.1f}s '
                'Data:{d.sum:.1f}s '
                'ClaLoss:{cla.avg:.4f} '
                'PairLoss:{pair.avg:.4f} '
                'CloLoss:{clo.avg:.4f} '
                'AdvLoss:{adv.avg:.4f} '
                'Acc:{acc.avg:.2%} '
                'CloAcc:{cloacc.avg:.2%} '.format(
                    epoch + 1,
                    t=meter_time, d=meter_data,
                    cla=meter_cla, pair=meter_pair, clo=meter_clo, adv=meter_adv,
                    acc=meter_acc, cloacc=meter_clo_acc))


def train_cal_with_memory(config, epoch, model, classifier,
                          criterion_cla, criterion_pair, criterion_adv,
                          optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    meter_cla, meter_pair, meter_adv, meter_acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    meter_time, meter_data = AverageMeter(), AverageMeter()

    model.train()
    classifier.train()

    erase_prob       = getattr(config.LOSS, "ERASE_PROB", 0.5)
    erase_keep_range = tuple(getattr(config.LOSS, "ERASE_KEEP_RANGE", (0.1, 0.3)))
    erase_dilate_px  = int(getattr(config.LOSS, "ERASE_DILATE_PX", 1))
    adv_weight       = getattr(config.LOSS, "ADV_WEIGHT", 1.0)
    att_weight       = getattr(config.LOSS, "ATT_WEIGHT", 1.0)

    logger.info(f"[ERASE] keep_ratio_range={erase_keep_range}, dilate_px={erase_dilate_px}, erase_prob={erase_prob}")
    logger.info(f"[LOSS WEIGHTS] pair={config.LOSS.PAIR_LOSS_WEIGHT}, adv={adv_weight}, att={att_weight}")

    end = time.time()

    for batch_idx, batch in enumerate(trainloader):
        imgs, pids, camids, clothes_ids, parse_paths, masks = _unpack_train_batch(batch)
        apply_erase = (parse_paths is not None) and (random.random() < erase_prob)

        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids = imgs.cuda(), pids.cuda(), clothes_ids.cuda()
        pos_mask = torch.as_tensor(pos_mask, dtype=torch.float32, device=imgs.device)

        if apply_erase:
            erased_imgs = make_erase_images(
                imgs, parse_paths,
                keep_ratio_range=erase_keep_range,
                dilate_px=erase_dilate_px
            )
        else:
            erased_imgs = None

        meter_data.update(time.time() - end)

        # RAW
        features_raw = model(imgs)
        outputs_raw  = classifier(features_raw)
        with torch.no_grad():
            preds = outputs_raw.detach().argmax(1)

        cla_loss_raw  = criterion_cla(outputs_raw,   pids)
        pair_loss_raw = criterion_pair(features_raw, pids)

        adv_loss = criterion_adv(features_raw, clothes_ids, pos_mask) \
            if epoch >= config.TRAIN.START_EPOCH_ADV else torch.tensor(0.0, device=imgs.device)

        # ERASE 分支（只影响 loss 的平均）
        if apply_erase:
            features_erase = model(erased_imgs)
            outputs_erase  = classifier(features_erase)
            cla_loss_erase  = criterion_cla(outputs_erase, pids)
            pair_loss_erase = criterion_pair(features_erase, pids)

            cla_loss  = 0.5 * (cla_loss_raw  + cla_loss_erase)
            pair_loss = 0.5 * (pair_loss_raw + pair_loss_erase)
        else:
            cla_loss  = cla_loss_raw
            pair_loss = pair_loss_raw

        # 注意力约束
        if masks is not None:
            M_hair, M_face, M_limbs, _ = [m.cuda() for m in masks]
            att_loss = build_attention_loss(features_raw, M_face, M_limbs, M_hair)
        else:
            att_loss = torch.tensor(0.0, device=imgs.device)

        # 总损失
        loss = cla_loss \
             + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss \
             + adv_weight * adv_loss \
             + att_weight * att_loss

        # ========== 不用梯度累计：每 iter 更新 ==========
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # 统计
        with torch.no_grad():
            meter_acc.update((preds == pids).float().mean(), pids.size(0))
        meter_cla.update(cla_loss.item(), pids.size(0))
        meter_pair.update(pair_loss.item(), pids.size(0))
        meter_adv.update(adv_loss.item(), clothes_ids.size(0))

        meter_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{t.sum:.1f}s '
                'Data:{d.sum:.1f}s '
                'ClaLoss:{cla.avg:.4f} '
                'PairLoss:{pair.avg:.4f} '
                'AdvLoss:{adv.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                    epoch + 1,
                    t=meter_time, d=meter_data,
                    cla=meter_cla, pair=meter_pair, adv=meter_adv, acc=meter_acc))
