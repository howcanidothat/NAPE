from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class Night600_noen(ImageDataset):
    """Night600_en dataset.

    Custom dataset for night-time person re-identification.

    Dataset statistics:
        - identities: 600
        - images: train + query + gallery (to be specified based on your dataset)
    """
    _junk_pids = [0, -1]
    dataset_dir = 'Night600_noen'
    masks_base_dir = 'masks'
    cam_num = 6  # adjust based on your dataset
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
        if masks_dir not in Night600_noen.masks_dirs:
            return None
        else:
            return Night600_noen.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.kp_dir = kwargs['config'].model.kpr.keypoints.kp_dir
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # Check if dataset directory exists
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError(f"Dataset directory not found: {self.dataset_dir}")

        self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        # ✅ 修改后的打印信息
        print(f"数据集统计: train={len(train)}, query={len(query)}, gallery={len(gallery)}")

        # 样本限制功能已禁用，始终使用完整数据集
        print("使用完整数据集进行测试（样本限制已禁用）...")

        # 确保query和gallery有匹配的ID
        query, gallery = self.ensure_query_gallery_match(query, gallery)

        super(Night600_noen, self).__init__(train, query, gallery, **kwargs)

    def ensure_query_gallery_match(self, query, gallery):
        """确保query和gallery中有匹配的person ID"""
        query_pids = set([item['pid'] for item in query])
        gallery_pids = set([item['pid'] for item in gallery])

        # 找到共同的person ID
        common_pids = query_pids.intersection(gallery_pids)

        if len(common_pids) == 0:
            print("警告: query和gallery中没有共同的person ID!")
            return query, gallery

        # 过滤query和gallery，只保留共同的person ID
        filtered_query = [item for item in query if item['pid'] in common_pids]
        filtered_gallery = [item for item in gallery if item['pid'] in common_pids]

        print(
            f"已限制 query 数据集为 {len(filtered_query)} 个样本，包含 {len(set([item['pid'] for item in filtered_query]))} 个唯一ID")
        print(f"已限制 gallery 数据集为 {len(filtered_gallery)} 个样本，确保包含所有查询ID")

        # 检查是否还有query ID在gallery中没有对应样本
        final_query_pids = set([item['pid'] for item in filtered_query])
        final_gallery_pids = set([item['pid'] for item in filtered_gallery])
        missing_pids = final_query_pids - final_gallery_pids

        if missing_pids:
            print(f"警告: 仍有 {len(missing_pids)} 个查询ID在图库中没有对应样本!")

        return filtered_query, filtered_gallery

    def limit_samples_with_matching(self, query_all, gallery_all, limit_samples):
        """智能限制样本数量，确保query和gallery都正好为limit_samples个样本且包含相同的person ID"""
        # 获取所有person ID
        query_pids = set([item['pid'] for item in query_all])
        gallery_pids = set([item['pid'] for item in gallery_all])
        common_pids = query_pids.intersection(gallery_pids)

        if len(common_pids) == 0:
            print("警告: query和gallery中没有共同的person ID!")
            return query_all[:limit_samples], gallery_all[:limit_samples]

        # 按person ID分组
        query_by_pid = {}
        gallery_by_pid = {}

        for item in query_all:
            pid = item['pid']
            if pid in common_pids:
                if pid not in query_by_pid:
                    query_by_pid[pid] = []
                query_by_pid[pid].append(item)

        for item in gallery_all:
            pid = item['pid']
            if pid in common_pids:
                if pid not in gallery_by_pid:
                    gallery_by_pid[pid] = []
                gallery_by_pid[pid].append(item)

        # 过滤出在两个集合中都有样本的person ID
        valid_pids = []
        for pid in common_pids:
            if pid in query_by_pid and pid in gallery_by_pid:
                valid_pids.append(pid)

        if len(valid_pids) == 0:
            print("警告: 没有在query和gallery中都存在的person ID!")
            return query_all[:limit_samples], gallery_all[:limit_samples]

        # 计算每个ID应该取多少样本
        samples_per_pid = max(1, limit_samples // len(valid_pids))

        limited_query = []
        limited_gallery = []

        # 第一轮：为每个有效的person ID分配基本样本数
        for pid in valid_pids:
            # 从query中取样本
            pid_query_samples = query_by_pid[pid][:samples_per_pid]
            limited_query.extend(pid_query_samples)

            # 从gallery中取相同数量的样本
            pid_gallery_samples = gallery_by_pid[pid][:samples_per_pid]
            limited_gallery.extend(pid_gallery_samples)

        # 第二轮：如果还没达到limit_samples，继续添加样本
        pid_index = 0
        while len(limited_query) < limit_samples and len(limited_gallery) < limit_samples:
            pid = valid_pids[pid_index % len(valid_pids)]

            # 计算当前这个ID已经有多少样本了
            current_query_count = sum(1 for item in limited_query if item['pid'] == pid)
            current_gallery_count = sum(1 for item in limited_gallery if item['pid'] == pid)

            # 如果还有更多样本可以添加
            if (current_query_count < len(query_by_pid[pid]) and
                    current_gallery_count < len(gallery_by_pid[pid]) and
                    len(limited_query) < limit_samples and
                    len(limited_gallery) < limit_samples):
                # 添加一个query样本
                limited_query.append(query_by_pid[pid][current_query_count])
                # 添加一个gallery样本
                limited_gallery.append(gallery_by_pid[pid][current_gallery_count])

            pid_index += 1

            # 防止无限循环：如果所有ID都已经用完了可用样本
            if pid_index > len(valid_pids) * 10:  # 最多循环10轮
                break

        # 确保两个数据集都正好是limit_samples个样本
        limited_query = limited_query[:limit_samples]
        limited_gallery = limited_gallery[:limit_samples]

        # 统计最终结果
        final_query_pids = set([item['pid'] for item in limited_query])
        final_gallery_pids = set([item['pid'] for item in limited_gallery])

        print(f"✅ query 数据集: {len(limited_query)} 个样本，包含 {len(final_query_pids)} 个唯一ID")
        print(f"✅ gallery 数据集: {len(limited_gallery)} 个样本，包含 {len(final_gallery_pids)} 个唯一ID")

        # 验证ID一致性
        if final_query_pids == final_gallery_pids:
            print("✅ query和gallery包含完全相同的person ID")
        else:
            missing_in_gallery = final_query_pids - final_gallery_pids
            missing_in_query = final_gallery_pids - final_query_pids
            if missing_in_gallery:
                print(f"⚠️  gallery中缺少的ID: {missing_in_gallery}")
            if missing_in_query:
                print(f"⚠️  query中缺少的ID: {missing_in_query}")

        return limited_query, limited_gallery

    def limit_samples_flexible(self, query_all, gallery_all, query_limit, gallery_limit):
        """灵活限制query和gallery的样本数量，允许不同的限制数量"""
        # 获取所有person ID
        query_pids = set([item['pid'] for item in query_all])
        gallery_pids = set([item['pid'] for item in gallery_all])
        common_pids = query_pids.intersection(gallery_pids)

        if len(common_pids) == 0:
            print("警告: query和gallery中没有共同的person ID!")
            return query_all[:query_limit], gallery_all[:gallery_limit]

        # 按person ID分组
        query_by_pid = {}
        gallery_by_pid = {}

        for item in query_all:
            pid = item['pid']
            if pid in common_pids:
                if pid not in query_by_pid:
                    query_by_pid[pid] = []
                query_by_pid[pid].append(item)

        for item in gallery_all:
            pid = item['pid']
            if pid in common_pids:
                if pid not in gallery_by_pid:
                    gallery_by_pid[pid] = []
                gallery_by_pid[pid].append(item)

        # 过滤出在两个集合中都有样本的person ID
        valid_pids = []
        for pid in common_pids:
            if pid in query_by_pid and pid in gallery_by_pid:
                valid_pids.append(pid)

        if len(valid_pids) == 0:
            print("警告: 没有在query和gallery中都存在的person ID!")
            return query_all[:query_limit], gallery_all[:gallery_limit]

        # 为query分配样本
        query_samples_per_pid = max(1, query_limit // len(valid_pids))
        limited_query = []

        for pid in valid_pids:
            pid_query_samples = query_by_pid[pid][:query_samples_per_pid]
            limited_query.extend(pid_query_samples)

        # 如果还没达到query_limit，继续添加样本
        pid_index = 0
        while len(limited_query) < query_limit:
            pid = valid_pids[pid_index % len(valid_pids)]
            current_count = sum(1 for item in limited_query if item['pid'] == pid)

            if current_count < len(query_by_pid[pid]):
                limited_query.append(query_by_pid[pid][current_count])

            pid_index += 1
            if pid_index > len(valid_pids) * 10:  # 防止无限循环
                break

        # 为gallery分配样本 - 确保包含所有query中的person ID
        query_pids_final = set([item['pid'] for item in limited_query[:query_limit]])
        limited_gallery = []

        # 首先确保每个query ID在gallery中至少有一个样本
        for pid in query_pids_final:
            if pid in gallery_by_pid and len(gallery_by_pid[pid]) > 0:
                limited_gallery.append(gallery_by_pid[pid][0])

        # 然后添加更多gallery样本直到达到限制
        gallery_samples_per_pid = max(1, (gallery_limit - len(limited_gallery)) // len(query_pids_final))

        for pid in query_pids_final:
            if pid in gallery_by_pid:
                # 跳过已经添加的第一个样本
                additional_samples = gallery_by_pid[pid][1:gallery_samples_per_pid + 1]
                limited_gallery.extend(additional_samples)

        # 如果还没达到gallery_limit，继续添加样本
        pid_index = 0
        valid_gallery_pids = list(query_pids_final)
        while len(limited_gallery) < gallery_limit:
            pid = valid_gallery_pids[pid_index % len(valid_gallery_pids)]
            current_count = sum(1 for item in limited_gallery if item['pid'] == pid)

            if pid in gallery_by_pid and current_count < len(gallery_by_pid[pid]):
                limited_gallery.append(gallery_by_pid[pid][current_count])

            pid_index += 1
            if pid_index > len(valid_gallery_pids) * 20:  # 防止无限循环
                break

        # 确保不超过限制
        limited_query = limited_query[:query_limit]
        limited_gallery = limited_gallery[:gallery_limit]

        # 统计最终结果
        final_query_pids = set([item['pid'] for item in limited_query])
        final_gallery_pids = set([item['pid'] for item in limited_gallery])

        print(f"✅ query 数据集: {len(limited_query)} 个样本，包含 {len(final_query_pids)} 个唯一ID")
        print(f"✅ gallery 数据集: {len(limited_gallery)} 个样本，包含 {len(final_gallery_pids)} 个唯一ID")

        # 验证所有query ID都在gallery中有对应
        missing_in_gallery = final_query_pids - final_gallery_pids
        if len(missing_in_gallery) == 0:
            print("✅ 所有query ID都在gallery中有对应样本")
        else:
            print(f"⚠️  gallery中缺少的query ID: {missing_in_gallery}")

        return limited_query, limited_gallery

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.extend(glob.glob(osp.join(dir_path, '*.png')))

        img_paths.sort()

        # Night600_en uses format: personID_c1s1_frameID_00.jpg
        pattern = re.compile(r'([-\d]+)_c(\d)s\d')

        pid_container = set()
        for img_path in img_paths:
            try:
                pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
                if pid == -1:
                    continue
                pid_container.add(pid)
            except:
                print(f"Warning: Could not parse filename {osp.basename(img_path)}")
                continue

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            try:
                pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
                if pid == -1:
                    continue

                # ✅ 保留真实的摄像头编号
                camid = camid - 1  # 从0开始计数 (c1 -> 0, c2 -> 1 ...)

                if relabel:
                    pid = pid2label[pid]

                masks_path = self.infer_masks_path_custom(img_path, dir_path)
                kp_path = self.infer_kp_path_custom(img_path, dir_path)

                data.append({
                    'img_path': img_path,
                    'pid': pid,
                    'masks_path': masks_path,
                    'camid': camid,
                    'kp_path': kp_path,
                })
            except:
                continue

        return data

    def infer_masks_path_custom(self, img_path, dir_path):
        """Custom masks path inference for Night600_en dataset"""
        img_name = osp.basename(img_path)
        base_name = osp.splitext(img_name)[0]

        # Map directory names
        if 'bounding_box_train' in dir_path:
            masks_subdir = 'train'
        elif 'bounding_box_test' in dir_path:
            masks_subdir = 'gallery'
        elif 'query' in dir_path:
            masks_subdir = 'query'
        else:
            masks_subdir = osp.basename(dir_path)

        masks_path = osp.join(
            self.dataset_dir,
            self.masks_base_dir,
            self.masks_dir,
            masks_subdir,
            base_name + self.masks_suffix
        )
        return masks_path

    def infer_kp_path_custom(self, img_path, dir_path):
        """Custom keypoints path inference for Night600_en dataset"""
        img_name = osp.basename(img_path)
        base_name = osp.splitext(img_name)[0]

        # Map directory names
        if 'bounding_box_train' in dir_path:
            kp_subdir = 'train'
        elif 'bounding_box_test' in dir_path:
            kp_subdir = 'gallery'
        elif 'query' in dir_path:
            kp_subdir = 'query'
        else:
            kp_subdir = osp.basename(dir_path)

        kp_path = osp.join(
            self.dataset_dir,
            'external_annotation',
            self.kp_dir,
            kp_subdir,
            base_name + '.json'
        )
        return kp_path