from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
from collections import defaultdict

from ..dataset import ImageDataset


class _NightReIDBase(ImageDataset):
    """NightReID dataset base (3 cams, no grouping).

    Expect directory layout under dataset_dir (subset-specific):
      - bounding_box_train/
      - query/
      - bounding_box_test/
      - masks/pifpaf_maskrcnn_filtering/train|query|gallery/*.npy
      - keypoints/train|query|gallery/*.json
    """

    _junk_pids = [0, -1]
    masks_base_dir = 'masks'
    cam_num = 3  # 3 cams, no grouping
    train_dir = 'bounding_box_train'
    query_dir = 'query'
    gallery_dir = 'bounding_box_test'

    # name -> (parts_num, has_background, suffix)
    masks_dirs = {
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        return _NightReIDBase.masks_dirs.get(masks_dir, None)

    def __init__(self, root='', masks_dir=None, **kwargs):
        cfg = kwargs.get('config', None)
        try:
            # 可选：优先从配置获取 keypoints 目录别名（与旧工程兼容），但我们首先查找 dataset_dir/keypoints/<split>
            self.kp_dir = getattr(cfg.model.nape.keypoints, 'kp_dir', 'keypoints') if cfg else 'keypoints'
        except Exception:
            self.kp_dir = 'keypoints'

        # 限流参数
        self.sample_limit = getattr(cfg.data, 'sample_limit', None) if cfg else None
        self.query_limit = getattr(cfg.data, 'query_limit', None) if cfg else None
        self.gallery_limit = getattr(cfg.data, 'gallery_limit', None) if cfg else None
        self.keep_qg_ratio = getattr(cfg.data, 'keep_qg_ratio', False) if cfg else False

        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)

        self.check_before_run([self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir])

        train = self.process_dir(self.train_dir, relabel=True)
        query_full = self.process_dir(self.query_dir, relabel=False)
        gallery_full = self.process_dir(self.gallery_dir, relabel=False)

        print(f"[NightReID] 数据集统计: train={len(train)}, query={len(query_full)}, gallery={len(gallery_full)}")

        # 确保 query 和 gallery 的身份集合重叠
        query, gallery = self.ensure_query_gallery_match(query_full, gallery_full)

        # 限流
        q_lim = self.query_limit if (self.query_limit and self.query_limit > 0) else 0
        g_lim = self.gallery_limit if (self.gallery_limit and self.gallery_limit > 0) else 0
        if q_lim or g_lim:
            query, gallery = self._limit_with_ratio(query, gallery, q_lim, g_lim)
        elif self.sample_limit:
            query, gallery = self._limit_splits_with_pid_overlap(query, gallery)
        else:
            pass

        super(_NightReIDBase, self).__init__(train, query, gallery, **kwargs)

    # -------------------------------
    # 处理图片路径（支持 000001_c1_000001.jpg 与 Market 格式）
    # -------------------------------
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.png')))
        img_paths.sort()

        # Night 格式: 000001_c1_000001.jpg -> pid=000001, cam=1
        pattern_night = re.compile(r'([\-\d]+)_c(\d+)_\d+')
        # Market 格式: 0001_c1s1_000451_03.jpg -> pid=0001, cam=1
        pattern_market = re.compile(r'([\-\d]+)_c(\d)s\d_')

        pid_container = set()
        for img_path in img_paths:
            basename = osp.basename(img_path)
            m = pattern_night.search(basename) or pattern_market.search(basename)
            if not m:
                continue
            try:
                pid = int(m.group(1))
                if pid in self._junk_pids:
                    continue
                pid_container.add(pid)
            except Exception:
                continue

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        data = []
        for img_path in img_paths:
            basename = osp.basename(img_path)
            m = pattern_night.search(basename) or pattern_market.search(basename)
            if not m:
                continue
            try:
                pid = int(m.group(1))
                camid_str = m.group(2)
                camid = int(camid_str) - 1  # 映射为 0,1,2
                if pid in self._junk_pids:
                    continue
                if relabel:
                    pid = pid2label[pid]

                data.append({
                    'img_path': img_path,
                    'pid': pid,
                    'masks_path': self.infer_masks_path_custom(img_path, dir_path),
                    'kp_path': self.infer_kp_path_custom(img_path, dir_path),
                    'camid': camid,
                })
            except Exception:
                continue

        print(f"Processed {len(data)} samples from {dir_path}")
        return data

    # -------------------------------
    # 限流逻辑
    # -------------------------------
    def _limit_with_ratio(self, query_all, gallery_all, query_limit, gallery_limit):
        q_sorted = sorted(query_all, key=lambda x: x['img_path'])
        g_sorted = sorted(gallery_all, key=lambda x: x['img_path'])

        g_pids = set([s['pid'] for s in g_sorted])
        q_candidates = [s for s in q_sorted if s['pid'] in g_pids]

        limited_q = q_candidates[:min(query_limit, len(q_candidates))] if query_limit else q_candidates
        required_pids = set([s['pid'] for s in limited_q])

        g_by_pid = defaultdict(list)
        for s in g_sorted:
            g_by_pid[s['pid']].append(s)

        limited_g, chosen = [], set()
        for pid in required_pids:
            if g_by_pid[pid]:
                s = g_by_pid[pid][0]
                limited_g.append(s)
                chosen.add(s['img_path'])

        for s in g_sorted:
            if gallery_limit and len(limited_g) >= gallery_limit:
                break
            if s['img_path'] not in chosen:
                limited_g.append(s)
                chosen.add(s['img_path'])

        print(f"[NightReID] 实际采样的 query={len(limited_q)} IDs={len(set([s['pid'] for s in limited_q]))}")
        print(f"[NightReID] 实际采样的 gallery={len(limited_g)} IDs={len(set([s['pid'] for s in limited_g]))}")
        return limited_q, limited_g

    def _limit_splits_with_pid_overlap(self, query, gallery):
        q_by_pid, g_by_pid = defaultdict(list), defaultdict(list)
        for s in query:
            q_by_pid[s['pid']].append(s)
        for s in gallery:
            g_by_pid[s['pid']].append(s)

        common_pids = [p for p in q_by_pid if p in g_by_pid]
        if not common_pids:
            return query, gallery

        limited_q, limited_g = [], []
        for pid in common_pids:
            limited_q.extend(q_by_pid[pid][:self.sample_limit])
            limited_g.extend(g_by_pid[pid][:self.sample_limit])
        return limited_q[:self.sample_limit], limited_g[:self.sample_limit]

    # -------------------------------
    # masks/kp 路径推断（优先新结构 keypoints/<split>，兼容 external_annotation/<kp_dir>/<split>）
    # -------------------------------
    def _split_name_from_dir(self, dir_path):
        if 'bounding_box_train' in dir_path:
            return 'train'
        if 'bounding_box_test' in dir_path:
            return 'gallery'
        if 'query' in dir_path:
            return 'query'
        return osp.basename(dir_path)

    def infer_masks_path_custom(self, img_path, dir_path):
        img_name = osp.splitext(osp.basename(img_path))[0]
        subdir = self._split_name_from_dir(dir_path)
        base_dir = osp.join(self.dataset_dir, self.masks_base_dir, self.masks_dir, subdir)
        return osp.join(base_dir, img_name + (self.masks_suffix or '.npy'))

    def infer_kp_path_custom(self, img_path, dir_path):
        img_name = osp.splitext(osp.basename(img_path))[0]
        subdir = self._split_name_from_dir(dir_path)

        # 优先新结构：dataset_dir/keypoints/<split>
        bases = [
            osp.join(self.dataset_dir, 'keypoints', subdir),
            # 兼容旧结构：dataset_dir/external_annotation/<kp_dir>/<split>
            osp.join(self.dataset_dir, 'external_annotation', self.kp_dir, subdir),
        ]
        candidates = [
            img_name + '.jpg_keypoints.json',
            img_name + '_keypoints.json',
            img_name + '.json',
        ]
        for base in bases:
            for fname in candidates:
                cand = osp.join(base, fname)
                if osp.exists(cand):
                    return cand
        # fallback：返回新结构下的默认路径
        return osp.join(self.dataset_dir, 'keypoints', subdir, img_name + '_keypoints.json')

    # -------------------------------
    # 确保查询集和图库集的身份重叠
    # -------------------------------
    def ensure_query_gallery_match(self, query, gallery):
        query_pids = set([item['pid'] for item in query])
        gallery_pids = set([item['pid'] for item in gallery])
        common_pids = query_pids.intersection(gallery_pids)

        if len(common_pids) == 0:
            print(f"警告: query 和 gallery 中没有共同的 person ID!")
            return query, gallery

        filtered_query = [item for item in query if item['pid'] in common_pids]
        filtered_gallery = [item for item in gallery if item['pid'] in common_pids]

        print(f"已限制 query 为 {len(filtered_query)} 样本，ID数 {len(set([x['pid'] for x in filtered_query]))}")
        print(f"已限制 gallery 为 {len(filtered_gallery)} 样本，ID数 {len(set([x['pid'] for x in filtered_gallery]))}")
        return filtered_query, filtered_gallery


class NightReID_1000(_NightReIDBase):
    """NightReID subset 1000"""
    dataset_dir = 'dataset_nightreid/output/1000'


class NightReID_528(_NightReIDBase):
    """NightReID subset 528"""
    dataset_dir = 'dataset_nightreid/output/528'
