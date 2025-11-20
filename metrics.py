import torch
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
from skimage.metrics import structural_similarity as ssim 

def calculate_mse(pred, true):
    """计算均方误差（MSE）"""
    return F.mse_loss(pred, true)

def calculate_mae(pred, true):
    """计算平均绝对误差（MAE）"""
    return F.l1_loss(pred, true)

def calculate_psnr(pred, true, data_range=70.0):
    mse = F.mse_loss(pred.float(), true.float())
    # 使用的是一个固定的、理论上的数据最大动态范围 (data_range)
    return 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))

def calculate_ssim(pred, true):
    """计算结构相似性指数（SSIM）"""
    pred = pred.cpu()
    true = true.cpu()
    return ssim(pred.numpy(), true.numpy(), data_range=70)


def calculate_csi(tp, fp, fn):
    """计算CSI指标"""
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0

def calculate_far(tp, fp):
    """计算FAR指标"""
    denominator = tp + fp
    return fp / denominator if denominator > 0 else 0

def calculate_pod(tp, fn):
    """计算POD指标"""
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0

def compute_metrics(pred_images, true_images, threshold=10):
    """
    计算整个批次图像序列的CSI、FAR、POD、MSE、MAE、PSNR和SSIM指标。
    
    参数:
    pred_images: [batch_size, num_frames, height, width] 的预测图像Tensor。
    true_images: [batch_size, num_frames, height, width] 的真实图像Tensor。
    threshold: 用于二值化图像的阈值。
    
    返回:
    metrics: 包含平均CSI、FAR、POD、MSE、MAE、PSNR和SSIM的字典。
    """
    batch_size, num_frames, height, width = pred_images.shape

    # 初始化指标总和
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    TN_sum = 0
    MSE_sum = 0
    MAE_sum = 0
    PSNR_sum = 0
    SSIM_sum = 0
    total_frames = batch_size * num_frames

    # 遍历每帧图像
    for b in range(batch_size):
        for t in range(num_frames):
            pred_image = pred_images[b, t]
            true_image = true_images[b, t]

            # 二值化预测和真实图像
            pred_binary = (pred_image >= threshold).float()
            true_binary = (true_image >= threshold).float()

            # 计算TP, FP, FN, TN
            TP = torch.sum((pred_binary == 1) & (true_binary == 1))
            FP = torch.sum((pred_binary == 1) & (true_binary == 0))
            FN = torch.sum((pred_binary == 0) & (true_binary == 1))
            TN = torch.sum((pred_binary == 0) & (true_binary == 0))

            # 累加到总和中
            TP_sum += TP
            FP_sum += FP
            FN_sum += FN
            TN_sum += TN

            # 计算MSE, MAE, PSNR, SSIM
            MSE = calculate_mse(pred_image, true_image)
            MAE = calculate_mae(pred_image, true_image)
            PSNR = calculate_psnr(pred_image, true_image)
            SSIM = calculate_ssim(pred_image, true_image)

            MSE_sum += MSE
            MAE_sum += MAE
            PSNR_sum += PSNR
            SSIM_sum += SSIM

    # 计算平均指标
    CSI_avg = calculate_csi(TP_sum, FP_sum, FN_sum)
    FAR_avg = calculate_far(TP_sum, FP_sum)
    POD_avg = calculate_pod(TP_sum, FN_sum)
    MSE_avg = MSE_sum / total_frames
    MAE_avg = MAE_sum / total_frames
    PSNR_avg = PSNR_sum / total_frames
    SSIM_avg = SSIM_sum / total_frames

    metrics = {
        'CSI': CSI_avg,
        'FAR': FAR_avg,
        'POD': POD_avg,
        'MSE': MSE_avg.item(),
        'MAE': MAE_avg.item(),
        'PSNR': PSNR_avg.item(),
        'SSIM': SSIM_avg.item()
    }

    return metrics
