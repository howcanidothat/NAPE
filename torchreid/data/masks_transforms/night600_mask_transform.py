from __future__ import division, print_function, absolute_import

from torchreid.data.masks_transforms.mask_transform import MaskGroupingTransform

# 为Night600数据集的6通道mask创建简单的transform
class Night600SixChannelMask(MaskGroupingTransform):
    """
    直接使用6个通道的mask，不进行重新组合
    假设6个通道分别代表：头部、躯干、左臂、右臂、左腿、右腿
    """
    parts_grouping = {
        "part_0": [0],  # 第1个通道
        "part_1": [1],  # 第2个通道  
        "part_2": [2],  # 第3个通道
        "part_3": [3],  # 第4个通道
        "part_4": [4],  # 第5个通道
        "part_5": [5],  # 第6个通道
    }
    
    # 创建简单的parts_map，直接映射到索引
    parts_map = {i: i for i in range(6)}
    
    def __init__(self):
        super().__init__(self.parts_grouping, self.parts_map)