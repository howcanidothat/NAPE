from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
from collections import defaultdict

from ..dataset import ImageDataset


class Night600_en(ImageDataset):
    """Night600_en dataset (3 cams, no grouping)."""

    _junk_pids = [0, -1]
    dataset_dir = 'Night600_en'
    masks_base_dir = 'masks'
    cam_num = 3  # 3 cams, no grouping
    train_dir = 'bounding_box_train'
    query_dir = 'query'
    gallery_dir = 'bounding_box_test'

    masks_dirs = {
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        """返回 masks 配置 (parts_num, has_background, suffix)"""
        return Night600_en.masks_dirs.get(masks_dir, None)

    def __init__(self, root='', masks_dir=None, **kwargs):
        cfg = kwargs.get('config', None)
        try:
            self.kp_dir = cfg.model.kpr.keypoints.kp_dir
        except Exception:
            raise RuntimeError("Config 缺少 'model.kpr.keypoints.kp_dir'")

        # 限流参数
        self.sample_limit = getattr(cfg.data, "sample_limit", None) if cfg else None
        self.query_limit = getattr(cfg.data, "query_limit", None) if cfg else None
        self.gallery_limit = getattr(cfg.data, "gallery_limit", None) if cfg else None
        self.keep_qg_ratio = getattr(cfg.data, "keep_qg_ratio", False) if cfg else False

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

        print(f"数据集统计: train={len(train)}, query={len(query_full)}, gallery={len(gallery_full)}")

        # 确保query和gallery集的身份重叠
        query, gallery = self.ensure_query_gallery_match(query_full, gallery_full)

        # 限流
        q_lim = self.query_limit if (self.query_limit and self.query_limit > 0) else 0
        g_lim = self.gallery_limit if (self.gallery_limit and self.gallery_limit > 0) else 0
        if q_lim or g_lim:
            query, gallery = self._limit_with_ratio(query, gallery, q_lim, g_lim)
        elif self.sample_limit:
            query, gallery = self._limit_splits_with_pid_overlap(query, gallery)
        else:
            # 保留 ensure_query_gallery_match 过滤后的 query/gallery
            pass

        super(Night600_en, self).__init__(train, query, gallery, **kwargs)

    # -------------------------------
    # 核心：处理图片路径 (不分组，直接 camid-1)
    # -------------------------------
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.png')))
        img_paths.sort()

        pattern = re.compile(r'([-\d]+)_c(\d)s\d')

        pid_container = set()
        for img_path in img_paths:
            try:
                pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
                if pid == -1: continue
                pid_container.add(pid)
            except:
                continue

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            try:
                pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
                if pid == -1: continue

                # 不分组，直接映射 c1→0, c2→1, c3→2
                camid = camid - 1

                if relabel:
                    pid = pid2label[pid]

                data.append({
                    'img_path': img_path,
                    'pid': pid,
                    'masks_path': self.infer_masks_path_custom(img_path, dir_path),
                    'kp_path': self.infer_kp_path_custom(img_path, dir_path),
                    'camid': camid,
                })
            except:
                continue

        # 打印处理后的数据
        print(f"Processed {len(data)} samples")
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
            if gallery_limit and len(limited_g) >= gallery_limit: break
            if s['img_path'] not in chosen:
                limited_g.append(s)
                chosen.add(s['img_path'])

        print(f"[Night600_en] 实际采样的 query={len(limited_q)} IDs={len(set([s['pid'] for s in limited_q]))}")
        print(f"[Night600_en] 实际采样的 gallery={len(limited_g)} IDs={len(set([s['pid'] for s in limited_g]))}")
        return limited_q, limited_g

    def _limit_splits_with_pid_overlap(self, query, gallery):
        q_by_pid, g_by_pid = defaultdict(list), defaultdict(list)
        for s in query: q_by_pid[s['pid']].append(s)
        for s in gallery: g_by_pid[s['pid']].append(s)

        common_pids = [p for p in q_by_pid if p in g_by_pid]
        if not common_pids: return query, gallery

        limited_q, limited_g = [], []
        for pid in common_pids:
            limited_q.extend(q_by_pid[pid][:self.sample_limit])
            limited_g.extend(g_by_pid[pid][:self.sample_limit])
        return limited_q[:self.sample_limit], limited_g[:self.sample_limit]

    # -------------------------------
    # masks/kp 路径推断
    # -------------------------------
    def infer_masks_path_custom(self, img_path, dir_path):
        img_name = osp.splitext(osp.basename(img_path))[0]
        if 'bounding_box_train' in dir_path: subdir = 'train'
        elif 'bounding_box_test' in dir_path: subdir = 'gallery'
        elif 'query' in dir_path: subdir = 'query'
        else: subdir = osp.basename(dir_path)
        base_dir = osp.join(self.dataset_dir, self.masks_base_dir, self.masks_dir, subdir)
        return osp.join(base_dir, img_name + (self.masks_suffix or '.npy'))

    def infer_kp_path_custom(self, img_path, dir_path):
        img_name = osp.splitext(osp.basename(img_path))[0]
        if 'bounding_box_train' in dir_path: subdir = 'train'
        elif 'bounding_box_test' in dir_path: subdir = 'gallery'
        elif 'query' in dir_path: subdir = 'query'
        else: subdir = osp.basename(dir_path)
        base_dir = osp.join(self.dataset_dir, 'external_annotation', self.kp_dir, subdir)
        for fname in [img_name + '.jpg_keypoints.json', img_name + '_keypoints.json', img_name + '.json']:
            cand = osp.join(base_dir, fname)
            if osp.exists(cand): return cand
        return osp.join(base_dir, img_name + '_keypoints.json')

    # -------------------------------
    # 确保查询集和图库集的身份重叠
    # -------------------------------
    def ensure_query_gallery_match(self, query, gallery):
        """确保query和gallery中有匹配的person ID"""
        query_pids = set([item['pid'] for item in query])
        gallery_pids = set([item['pid'] for item in gallery])

        # 打印调试信息：检查 query 和 gallery 中的 PID
        print(f"Query PIDs: {query_pids}")
        print(f"Gallery PIDs: {gallery_pids}")

        # 找到共同的person ID
        common_pids = query_pids.intersection(gallery_pids)
        print(f"Common PIDs: {common_pids}")

        if len(common_pids) == 0:
            print(f"警告: query和gallery中没有共同的person ID! query_pids: {query_pids}, gallery_pids: {gallery_pids}")
            return query, gallery

        # 过滤query和gallery，只保留共同的person ID
        filtered_query = [item for item in query if item['pid'] in common_pids]
        filtered_gallery = [item for item in gallery if item['pid'] in common_pids]

        print(
            f"已限制 query 数据集为 {len(filtered_query)} 个样本，包含 {len(set([item['pid'] for item in filtered_query]))} 个唯一ID")
        print(f"已限制 gallery 数据集为 {len(filtered_gallery)} 个样本，确保包含所有查询ID")

        return filtered_query, filtered_gallery
