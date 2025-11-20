import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataset import RadarDataset
import numpy as np
import random
# from 带约束可学习小波变换的WaST import WaST_level1, HighFocalFrequencyLoss
# from 多尺度时空_WaST import WaST_level1,HighFocalFrequencyLoss
# from ablation_wast import WaST_Ablation as WaST_level1
# from 带约束可学习小波变换_多尺度时空_WaST import HighFocalFrequencyLoss

# from wast_copy import  WaST_level1,HighFocalFrequencyLoss
# from wast_csr import WaST_level1, HighFocalFrequencyLoss
# from 带约束可学习小波变换_多尺度时空_WaST import WaST_level1, HighFocalFrequencyLoss
# from 多尺度时空_WaST import WaST_level1, HighFocalFrequencyLoss

#------------------最终消融实验-----------------------------------------------————————————

# 最终模型，多尺度时空，DCN
# from ablation_wast_final import WaST_Ablation as WaST_level1
from DD_Net import DD_Net, HighFocalFrequencyLoss
# from 带约束可学习小波变换_多尺度时空_WaST import HighFocalFrequencyLoss

# 原始wast
# from wast import WaST_level1, HighFocalFrequencyLoss

# wast_dcn
# from wast_dcn import WaST_level1, HighFocalFrequencyLoss

# wast_多尺度时空超图
# from ablation_wast import WaST_Ablation as WaST_level1
# from 带约束可学习小波变换_多尺度时空_WaST import HighFocalFrequencyLoss








 

from dataset import RadarDataset

from metrics import compute_metrics  # 导入计算指标的函数

def main():
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=1)

    bs = 8
    height = 64
    width = 64

    # 构建配置及模型（确保配置与训练时一致）
    # 定义模型参数
    in_shape = (20, 1, 64, 64)  # 输入数据的形状，格式为 (帧数, 通道数, 高度, 宽度)
    encoder_dim = 20  # 编码器的维度
    block_list = [2, 8, 2]  # 每个阶段的块数
    drop_path_rate = 0.1  # DropPath 率
    mlp_ratio = 4.  # MLP 层的扩展比例

    # 初始化模型
    # model = WaST_level1(in_shape=in_shape, encoder_dim=encoder_dim, block_list=block_list, 
    #                     drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio)
    model = DD_Net(in_shape=in_shape, encoder_dim=encoder_dim, block_list=block_list, 
                        # --- 在这里配置消融实验 ---
                        attention_mode='node',   # 'node' or 'hyperedge'
                        batching_mode='broadcast',  # 'broadcast' or 'standard
                        use_dcn_for_ablation=True,
                        # --- 消融实验配置结束 ---
                        drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio)

    model = model.to(accelerator.device)

    # 加载保存的 PredRNN 模型权重（请检查路径和文件名是否正确）
    model.load_state_dict(torch.load('./wast_多尺度时空cube_dcn.pt', map_location=accelerator.device))  # 'wast_多尺度时空cube_dcn'，'wast_基于多尺度时空cube'，'wast_dcn'，'wast'
    model.eval()

    # 构造测试数据集和数据加载器
    test_dataset = RadarDataset('data/test_data_new.npy')
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                             num_workers=8, drop_last=False)

    # 使用 Accelerator 准备模型和数据加载器
    model, test_loader = accelerator.prepare(model, test_loader)

    # 进行测试，计算平均损失和指标
    avg_loss, final_metrics = test_epoch(model, test_loader, accelerator)
    accelerator.print(f"Test Loss: {avg_loss:.4f}")
    accelerator.print("Overall Average Metrics:")
    accelerator.print(final_metrics)

def test_epoch(model, dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    metrics_list = []  # 用于存储每个 batch 的指标

    hffl = HighFocalFrequencyLoss()
    
    with torch.no_grad():
        for data in dataloader:
            # 假设 batch_data 的 shape 为 [batch, 40, 1, 64, 64]

            input = data[:, :20]
            target = data[:, 20:]
            output = model(input)

            loss = F.mse_loss(output, target) + hffl(output, target, reshape = True)

            total_loss += loss.item()


            print("Before slicing, pred:", output.shape, "target:", target.shape)
            
            
            # 去掉 channel 维度，变为 [batch, num_frames, height, width]
            output = output.squeeze(2)
            target = target.squeeze(2)

            
            print("After slicing, pred:", output.shape, "target:", target.shape)
    
            
            # 直接对整个 batch 计算指标
            batch_metrics = compute_metrics(output * 70, target * 70, threshold=10)
            metrics_list.append(batch_metrics)
    
    avg_loss = total_loss / len(dataloader)
    
    # 直接对所有 batch 的指标取平均
    final_metrics = {}
    for key in metrics_list[0].keys():
        final_metrics[key] = sum(batch_metric[key] for batch_metric in metrics_list) / len(metrics_list)
        
    return avg_loss, final_metrics

if __name__ == "__main__":
    main()
