import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed to avoid IOError under heavy IO."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    return accimage_loader if get_image_backend() == 'accimage' else pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    return accimage_loader(path) if get_image_backend() == 'accimage' else pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def _unpack_sample(entry, index=None, expecting_sequence=False):
    """
    Unify dataset entry unpack. Returns:
      (paths_or_path, pid, camid, clothes_id, parse_path)

    Supports:
      - tuple/list: (img(s), pid, camid, clothes_id[, parse_path, ...])  # >=4 items
      - dict: {'img'/'img_path'/'img_paths'/'paths', 'pid', 'camid'?, 'clothes_id'?, 'parse_path'?}
    """
    parse_path = None

    if isinstance(entry, (list, tuple)):
        if len(entry) < 4:
            raise ValueError(f"Bad sample at index {index}: expected at least 4 items, got {len(entry)}")
        paths_or_path, pid, camid, clothes_id = entry[:4]
        if len(entry) >= 5:
            parse_path = entry[4]

    elif isinstance(entry, dict):
        if expecting_sequence:
            paths_or_path = entry.get('img_paths') or entry.get('paths') or entry.get('imgs')
        else:
            paths_or_path = entry.get('img') or entry.get('img_path') or entry.get('path')
        if paths_or_path is None:
            raise ValueError(f"Bad sample dict at index {index}: missing image path(s)")
        pid        = entry['pid']
        camid      = entry.get('camid', 0)
        clothes_id = entry.get('clothes_id', 0)
        parse_path = entry.get('parse_path', None)
    else:
        raise TypeError(f"Unsupported sample type at index {index}: {type(entry)}")

    return paths_or_path, pid, camid, clothes_id, parse_path


class ImageDataset(Dataset):
    """Image Person ReID Dataset (passes parse_path downstream)."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # unpack: image path + labels + parse path
        entry = self.dataset[index]
        img_path, pid, camid, clothes_id, parse_path = _unpack_sample(entry, index, expecting_sequence=False)

        # read image
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        # 在 __getitem__ 里，return 之前加入这行：
        parse_path = parse_path or ""   # None -> ""
        return img, pid, camid, clothes_id, parse_path


        # # return 5-tuple (parse_path is str or None)
        # return img, pid, camid, clothes_id, parse_path


class VideoDataset(Dataset):
    """
    Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W

    Returns:
      cloth_changing=True: (clip, pid, camid, clothes_id)
      cloth_changing=False: (clip, pid, camid)
    """
    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        entry = self.dataset[index]
        img_paths, pid, camid, clothes_id, _ = _unpack_sample(entry, index, expecting_sequence=True)

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # T x C x H x W -> C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid
