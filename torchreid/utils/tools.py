from __future__ import division, print_function, absolute_import
import os
import sys
import json
import time
import errno
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch

from torchreid.data.datasets.keypoints_to_masks import kp_img_to_kp_bbox
from torchreid.utils.constants import bn_correspondants

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'set_random_seed', 'download_url', 'read_image', 'read_masks', 'collect_env_info', 'perc'
]


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


@lru_cache(maxsize=None)
def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


@lru_cache(maxsize=None)
def read_masks(masks_path):
    """Reads part-based masks information from path.

    Args:
        path (str): path to an image. Part-based masks information is stored in a .npy file with image name as prefix

    Returns:
        Numpy array of size N x H x W where N is the number of part-based masks
    """

    got_masks = False
    if not osp.exists(masks_path):
        raise IOError('Masks file"{}" does not exist'.format(masks_path))
    while not got_masks:
        try:
            masks = np.load(masks_path)
            masks = np.transpose(masks, (1, 2, 0))
            got_masks = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(masks_path)
            )
    return masks


kp_files = {
}


@lru_cache(maxsize=None)
def read_keypoints(kp_path, bbox_ltwh=None):
    got_kp = False
    if not osp.exists(kp_path):
        print('Warning: Keypoints file "{}" does not exist. Using default empty keypoints.'.format(kp_path))
        # 返回默认的空关键点数组，形状为 [1, 17, 3]，全部填充为0
        return np.zeros((1, 17, 3))

    while not got_kp:
        try:
            target_found = False
            sample_name = Path(kp_path).stem.split('.')[0]
            revert = True if sample_name in kp_files else False  # mark occluder as target and vice-versa for experimental purposes
            with open(kp_path) as json_file:
                skeletons = json.load(json_file)
                if isinstance(skeletons, dict):
                    skeletons["is_target"] = True
                    skeletons = [skeletons]
                if skeletons is None or len(skeletons) == 0:
                    # print('Keypoints file"{}" is empty'.format(kp_path))  # FIXME a lot for Market
                    keypoints_xyc = np.zeros((1, 17, 3))
                else:
                    target_kp = None
                    negative_kp = []
                    for i, skel in enumerate(skeletons):
                        # If the dataset was not annotated with "is_target" information, the first skeleton is taken as the target in an arbitrary fashion
                        # the only impacted dataset is Partial_REID
                        # Occluded-PoseTrack and Occluded_Duke are annotated with "is_target"
                        # Market and Occluded_REID don't have "is_target" information, but they mainly have a single skeleton per image.
                        # TODO should select the skeleton with the head closest to the top center part of the image as the target, similarly to the heuristic used to generate the human parsing labels
                        if "is_target" not in skel:
                            skel["is_target"] = False if i != 0 else True
                        kps_xyc = np.array(skel["keypoints"])
                        if len(kps_xyc.shape) == 1:
                            kps_xyc = kps_xyc.reshape((-1, 3))
                        if bbox_ltwh is not None:
                            kps_xyc = kp_img_to_kp_bbox(kps_xyc, bbox_ltwh)
                        if skel["is_target"]:
                            target_kp = kps_xyc
                        else:
                            negative_kp.append(kps_xyc)

                    # 如果没有找到目标关键点，创建一个空的目标关键点
                    if target_kp is None:
                        print(
                            'Warning: No target keypoint found in "{}". Using default empty keypoints.'.format(kp_path))
                        target_kp = np.zeros((17, 3))

                    # first skeleton is the reid target by convention
                    keypoints_xyc = np.stack([target_kp] + negative_kp) if negative_kp else np.expand_dims(target_kp,
                                                                                                           axis=0)
                got_kp = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(kp_path)
            )
        except Exception as e:
            print('Error "{}" when reading keypoints file "{}". Using default empty keypoints.'.format(str(e), kp_path))
            return np.zeros((1, 17, 3))

    return keypoints_xyc


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def perc(val, decimals=2):
    return np.around(val * 100, decimals)


def extract_test_embeddings(model_output, test_embeddings):
    """
    统一从模型 forward 输出里抽取 embeddings、visibility、masks
    """

    # ✅ 训练时 forward 返回 7 元组
    # (embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks, id_cls_scores)
    # ✅ 测试时 forward 返回 6 元组
    # (embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks)

    if len(model_output) == 7:
        embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks, id_cls_scores = model_output
    elif len(model_output) == 6:
        embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks = model_output
        id_cls_scores = None
    else:
        raise ValueError(f"Unexpected model_output length: {len(model_output)}")

    embeddings_list = []
    visibility_scores_list = []
    embeddings_masks_list = []

    from torchreid.utils.constants import GLOBAL, PARTS, bn_correspondants

    for test_emb in test_embeddings:
        # ✅ 修正拼写，确保不会再出现 'globl'
        if test_emb == "globl":
            test_emb = GLOBAL

        embds = embeddings[test_emb]
        embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))

        # bn_global 对应 global
        if test_emb in bn_correspondants:
            lookup_key = bn_correspondants[test_emb]
        else:
            lookup_key = test_emb

        if lookup_key in visibility_scores:
            vis_scores = visibility_scores[lookup_key]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
        else:
            # 没有 visibility 信息时补零
            N = embds.shape[0]
            P = embds.shape[1] if embds.dim() == 3 else 1
            visibility_scores_list.append(torch.ones((N, P), device=embds.device))

        if lookup_key in masks:
            pt_masks = masks[lookup_key]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))
        else:
            embeddings_masks_list.append(torch.zeros((embds.shape[0], 1, spatial_features.size(2), spatial_features.size(3)),
                                                     device=embds.device))

    assert len(embeddings) != 0

    embeddings = torch.cat(embeddings_list, dim=1)          # [N, P+2, D]
    visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
    embeddings_masks = torch.cat(embeddings_masks_list, dim=1)    # [N, P+2, Hf, Wf]

    return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores
