from __future__ import division, print_function, absolute_import

import re
import glob
import os
import os.path as osp
import warnings
from collections import defaultdict

from ..dataset import ImageDataset


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'Market-1501-v15.09.15'
    masks_base_dir = 'masks'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    cam_num = 6
    train_dir = 'bounding_box_train'
    query_dir = 'query'
    gallery_dir = 'bounding_box_test'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in Market1501.masks_dirs:
            return None
        else:
            return Market1501.masks_dirs[masks_dir]

    def __init__(self, root='', market1501_500k=False, masks_dir=None, **kwargs):
        self.kp_dir = kwargs['config'].model.nape.keypoints.kp_dir
        # Optional per-split sample limit for quick testing
        self.sample_limit = None
        try:
            cfg = kwargs.get('config', None)
            if cfg is not None and hasattr(cfg, 'data') and hasattr(cfg.data, 'sample_limit') and cfg.data.sample_limit:
                self.sample_limit = int(cfg.data.sample_limit)
        except Exception:
            # fall back to no limit if config does not define it
            self.sample_limit = None
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir

        # allow alternative directory structure
        if not osp.isdir(self.dataset_dir):
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)
        self.extra_gallery_dir = osp.join(self.dataset_dir, 'images')
        self.market1501_500k = market1501_500k
        # cache: base_dir -> { base_lower: actual_filename }
        self._kp_index = {}

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)

        # 当开启样本限流时，为确保评估合法性（所有 query 身份需在 gallery 出现），
        # 先构建完整的 query/gallery，然后基于身份交集做限流选择。
        if self.sample_limit is not None and self.sample_limit > 0:
            _tmp_limit = self.sample_limit
            # 暂时禁用限流以获取完整列表
            self.sample_limit = None
            query_full = self.process_dir(self.query_dir, relabel=False)
            gallery_full = self.process_dir(self.gallery_dir, relabel=False)
            if self.market1501_500k:
                gallery_full += self.process_dir(self.extra_gallery_dir, relabel=False)
            # 还原限流
            self.sample_limit = _tmp_limit
            # 基于身份交集进行限流
            query, gallery = self._limit_splits_with_pid_overlap(query_full, gallery_full)
        else:
            query = self.process_dir(self.query_dir, relabel=False)
            gallery = self.process_dir(self.gallery_dir, relabel=False)
            if self.market1501_500k:
                gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def _build_kp_index(self, base_dir):
        """Build once an index mapping base filename -> actual json filename with preference order.
        Preference: .jpg_keypoints.json > _keypoints.json > .json
        """
        index = {}
        pref_rank = {'.jpg_keypoints.json': 0, '_keypoints.json': 1, '.json': 2}
        try:
            entries = os.listdir(base_dir)
        except Exception:
            self._kp_index[base_dir] = index
            return
        for e in entries:
            el = e.lower()
            base = None
            suf = None
            if el.endswith('.jpg_keypoints.json'):
                base = el[:-len('.jpg_keypoints.json')]
                suf = '.jpg_keypoints.json'
            elif el.endswith('_keypoints.json'):
                base = el[:-len('_keypoints.json')]
                suf = '_keypoints.json'
            elif el.endswith('.json'):
                base = el[:-len('.json')]
                suf = '.json'
            else:
                continue
            if base is None:
                continue
            if base not in index:
                index[base] = e
            else:
                old = index[base]
                oldl = old.lower()
                if oldl.endswith('.jpg_keypoints.json'):
                    old_suf = '.jpg_keypoints.json'
                elif oldl.endswith('_keypoints.json'):
                    old_suf = '_keypoints.json'
                elif oldl.endswith('.json'):
                    old_suf = '.json'
                else:
                    old_suf = '.json'
                if pref_rank[suf] < pref_rank[old_suf]:
                    index[base] = e
        self._kp_index[base_dir] = index

    def infer_kp_path_market1501(self, img_path, dir_path):
        img_name = osp.basename(img_path)
        base_name = osp.splitext(img_name)[0]
        kp_subdir = osp.basename(dir_path)
        base_dir = osp.join(
            self.dataset_dir,
            'external_annotation',
            self.kp_dir,
            kp_subdir
        )
        # 使用缓存索引进行 O(1) 查找，避免每张图像多次磁盘访问
        if base_dir not in self._kp_index:
            self._build_kp_index(base_dir)
        idx = self._kp_index.get(base_dir, {})
        found = idx.get(base_name.lower())
        if found is not None:
            return osp.join(base_dir, found)
        # 回退到最常见的 .jpg_keypoints.json（可能不存在，但由 read_keypoints 统一处理）
        return osp.join(base_dir, base_name + '.jpg_keypoints.json')

    def infer_masks_path(self, img_path, masks_dir, masks_suffix):
        img_name = osp.basename(img_path)
        base_name = osp.splitext(img_name)[0]
        masks_subdir = osp.basename(osp.dirname(img_path))
        base_dir = osp.join(
            self.dataset_dir,
            self.masks_base_dir,
            masks_dir,
            masks_subdir
        )
        cand_path_masks = osp.join(base_dir, base_name + masks_suffix)
        if osp.exists(cand_path_masks):
            return cand_path_masks
        # fallback to plain .npy if no *_keypoints.npy exists
        return osp.join(base_dir, base_name + '.npy')

    def process_dir(self, dir_path, relabel=False):
        # 1) Gather and sort all jpgs for stability across runs/OS
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([\-\d]+)_c(\d)')

        # 2) Filter out junk images first (pid == -1)
        valid_img_paths = []
        for img_path in img_paths:
            m = pattern.search(img_path)
            if m is None:
                continue
            pid, _ = map(int, m.groups())
            if pid == -1:
                continue  # ignore junk images
            valid_img_paths.append(img_path)

        # 3) Apply optional per-split sample limit AFTER filtering
        if self.sample_limit is not None and self.sample_limit > 0:
            valid_img_paths = valid_img_paths[: self.sample_limit]

        # 4) Build pid container from the actually used images
        pid_container = set()
        for img_path in valid_img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # 5) Build final samples
        data = []
        for img_path in valid_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            masks_path = self.infer_masks_path(img_path, self.masks_dir, self.masks_suffix)
            kp_path = self.infer_kp_path_market1501(img_path, dir_path)
            data.append({
                'img_path': img_path,
                'pid': pid,
                'masks_path': masks_path,
                'camid': camid,
                'kp_path': kp_path,
            })
        return data

    def _limit_splits_with_pid_overlap(self, query, gallery):
        """在样本限流下，确保被选中的 query 身份在 gallery 中均存在。

        策略：
        - 计算 query/gallery 的身份交集。
        - query 仅从交集中顺序选取样本，直到达到 sample_limit 或用尽。
        - gallery 先保证每个所需身份至少选 1 张，再在这些身份中补齐到 sample_limit（若可）。
        - 若交集为空，则返回原始列表（不做限流），以避免评估异常。
        """
        limit = self.sample_limit if (self.sample_limit is not None and self.sample_limit > 0) else 0
        if not limit:
            return query, gallery

        # 按路径排序，确保可复现实验
        query_sorted = sorted(query, key=lambda x: x['img_path'])
        gallery_sorted = sorted(gallery, key=lambda x: x['img_path'])

        q_by_pid = defaultdict(list)
        g_by_pid = defaultdict(list)
        for s in query_sorted:
            q_by_pid[s['pid']].append(s)
        for s in gallery_sorted:
            g_by_pid[s['pid']].append(s)

        common_pids = [pid for pid in q_by_pid.keys() if pid in g_by_pid]
        if not common_pids:
            # 无交集（极少见，表示数据异常或路径推断失败），直接返回原始以避免断言
            return query, gallery

        # 1) 构建限流后的 query（仅从交集身份采样）
        limited_q = []
        for pid in common_pids:
            for s in q_by_pid[pid]:
                if len(limited_q) >= limit:
                    break
                limited_q.append(s)
            if len(limited_q) >= limit:
                break

        # 2) 构建限流后的 gallery，先覆盖所有 required_pids 再补齐
        required_pids = {s['pid'] for s in limited_q}
        limited_g = []
        # 先保证每个 required_pid 至少一张
        for pid in sorted(required_pids):
            lst = g_by_pid.get(pid, [])
            if lst:
                limited_g.append(lst[0])
        # 再在 required_pids 范围内补齐到上限
        if len(limited_g) < limit:
            for pid in sorted(required_pids):
                # 从第 2 张开始补
                for s in g_by_pid.get(pid, [])[1:]:
                    if len(limited_g) >= limit:
                        break
                    limited_g.append(s)
                if len(limited_g) >= limit:
                    break

        return limited_q, limited_g
