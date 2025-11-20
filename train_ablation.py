import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
# from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器

from dataset import RadarDataset
from tqdm import tqdm

from accelerate import Accelerator
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random


#------------------最终消融实验-----------------------------------------------————————————

# 最终模型，多尺度时空，DCN
from DD_Net import DD_Net, HighFocalFrequencyLoss


# 原始wast
# from wast import WaST_level1, HighFocalFrequencyLoss

# wast_dcn
# from wast_dcn import WaST_level1, HighFocalFrequencyLoss

# wast_多尺度时空超图
# from ablation_wast import WaST_Ablation as WaST_level1
# from 带约束可学习小波变换_多尺度时空_WaST import HighFocalFrequencyLoss







def main():
    # 设置随机种子
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=1)

    save_model_path = './'
    max_epochs = 100
    bs = 4
    height = 64
    width = 64
    input_num_frames = 20
    output_num_frames = 20

    data_type = torch.float32
    device = accelerator.device



    # 创建数据集和数据加载器

    train_dataset = RadarDataset('data/train_data_new.npy')
    train_dataloader = DataLoader(train_dataset, batch_size=bs, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    test_dataset = RadarDataset('data/test_data_new.npy') 
    test_dataloader = DataLoader(test_dataset, batch_size=5, num_workers=8, pin_memory=True)

    # 定义模型参数
    in_shape = (20, 1, 64, 64)  # 输入数据的形状，格式为 (帧数, 通道数, 高度, 宽度)
    encoder_dim = 20  # 编码器的维度
    block_list = [2, 8, 2]  # 每个阶段的块数
    drop_path_rate = 0.1  # DropPath 率
    mlp_ratio = 4.  # MLP 层的扩展比例

    # 初始化模型
    model = DD_Net(in_shape=in_shape, encoder_dim=encoder_dim, block_list=block_list, 
                        # --- 在这里配置消融实验 ---
                        attention_mode='node',
                        batching_mode='broadcast',
                        use_dcn_for_ablation=True,
                        # --- 消融实验配置结束 ---
                        drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio)
    
    # model.load_state_dict(torch.load('./WaST_Causal.pt', map_location=accelerator.device))

    
    model = model.to(device)

    # 定义损失函数和优化器
    hffl = HighFocalFrequencyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    writer = SummaryWriter(log_dir='tf-logs/wast')

    def test_epoch(model, dataloader):
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            # 添加测试进度条
            test_pbar = tqdm(dataloader, desc="测试中", leave=False, unit="batch")
            for test_data in test_pbar:
                test_input = test_data[:, :input_num_frames]
                test_target = test_data[:, input_num_frames:input_num_frames+output_num_frames]
                test_output = model(test_input)

                test_loss = F.mse_loss(test_output, test_target) + hffl(test_output, test_target, reshape = True)

                total_test_loss += test_loss.item()

        return total_test_loss / len(dataloader)


    
    # 早停机制参数
    start_epoch = 200    # 第50轮开始监测早停
    patience = 10  # 如果验证集损失在连续5个epoch内没有改善，则停止训练
    best_loss = float('inf')
    counter = 0

    # 添加epoch级别的进度条
    epoch_pbar = tqdm(range(max_epochs), desc="训练进度", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        # 添加step级别的进度条
        step_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        
        for step, data in enumerate(step_pbar):
            # data shape: [batch, 40, 1, 64, 64]
            input = data[:, :input_num_frames]
            target = data[:, input_num_frames:input_num_frames+output_num_frames]
            output = model(input)

            loss = F.mse_loss(output, target) + hffl(output, target, reshape = True)

            # # --- 新的损失函数计算方式 ---
            # # 1. 定义一个权重图
            # weight_map = torch.ones_like(target).to(device)
            
            # # 2. 找到目标值超过阈值的像素点 (假设您的数据经过了/70的归一化)
            # threshold_normalized = 10.0 / 70.0
            # important_pixels_mask = (target > threshold_normalized)
            
            # # 3. 为这些重要像素点赋予更高的权重
            # weight_map[important_pixels_mask] = 5.0  # 权重设为10，可以调整
            
            # # 4. 计算加权的MSE
            # # 首先计算每个像素的loss，但不求平均
            # per_pixel_loss = F.mse_loss(output, target, reduction='none')
            # # 然后用权重图进行加权
            # weighted_mse = (per_pixel_loss * weight_map).mean()
            
            # # 5. 与HFFL组合
            # loss = weighted_mse + hffl(output, target, reshape=True)
            # # --- 损失函数修改结束 ---

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            writer.add_scalar('Loss/train_per_step', loss.item(), epoch * len(train_dataloader) + step)

            # 更新step进度条显示
            step_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_dataloader) 
        test_loss = test_epoch(model, test_dataloader)
        writer.add_scalar('Loss/train_avg', avg_loss, epoch+1)
        writer.add_scalar('Loss/test', test_loss, epoch+1)
        
        # 更新epoch进度条显示
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_loss:.6f}',
            'Test Loss': f'{test_loss:.6f}',
            'Best Loss': f'{best_loss:.6f}'
        })

        # 早停机制
        if test_loss <= best_loss:
            best_loss = test_loss
            counter = 0
            # 保存最佳模型
            convlstm_save_path = os.path.join(save_model_path, f'wast.pt')
            torch.save(model.state_dict(), convlstm_save_path)
            epoch_pbar.write(f"最佳模型已保存至 {save_model_path}")
        else:
            # 仅在epoch>=50时开始计数
            if epoch >= start_epoch:
                counter += 1
                epoch_pbar.write(f"验证损失未改善，计数器 {counter}/{patience}")

        # 检查是否需要早停
        if epoch >= start_epoch and counter >= patience:
            epoch_pbar.write(f"早停触发于第 {epoch+1} 轮")
            break

    epoch_pbar.close()

    writer.close()

if __name__ == "__main__":
    main()