import torch
from torch.utils.data import Dataset
import numpy as np

class RadarDataset(Dataset):
    def __init__(self, data_path):
        """
        初始化数据集
        :param data_path: 处理后的雷达数据保存路径，如'E:/test/input_raw_data2011-7.npy'
        """
        self.data = np.load(data_path, allow_pickle=True)  # 加载预处理后的雷达数据[6](@ref)
        self.total_samples = self.data.shape[1] 

    def __len__(self):
        """
        返回数据集总样本数
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本索引
        :return: 雷达数据（40,256,256）
        """
        radar_data = self.data[:, idx, :, :] 
        
        # 转换为Tensor
        radar_data = torch.from_numpy(radar_data).float()
        
        return radar_data.unsqueeze(1)/70

# 使用示例
if __name__ == '__main__':
    dataset = RadarDataset(
        data_path='/Users/lwd/Documents/BFU/paper/Data/北工大处理的数据/train_data.npy'
    )
    
    # 查看第一个样本
    radar_batch = dataset[0]
    print(f"Sample Shape: {radar_batch.shape}")