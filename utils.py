import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_image_auroc(image_scores, image_labels):
    """
    计算图片级 AUROC
    """
    try:
        # 清理异常值，防止偶尔由于模型导致出现 NaN/Inf 从而报错
        image_scores = np.nan_to_num(np.array(image_scores))
        return roc_auc_score(image_labels, image_scores)
    except ValueError:
        # 当标签中只包含一类（如全为0 或全为1）时，计算会报错，返回0.0处理
        return 0.0

def compute_pixel_auroc(pixel_scores, pixel_masks):
    """
    计算像素级 AUROC
    """
    pixel_scores = pixel_scores.flatten()
    pixel_masks = pixel_masks.flatten()
    
    # 确立安全的二值掩码 (0或1)，即使原图是 255 插值后的偶尔浮点数
    pixel_masks = (pixel_masks > 0.5).astype(int)
    
    # 清理偶尔产生的 NaN / Inf 异常特征得分
    pixel_scores = np.nan_to_num(pixel_scores)
    
    # 防止显存或内存不足，如果数据量极其巨大（如多类全量混训验证），可以取下采样
    if len(pixel_scores) > 100000000: # 超过1亿像素点（约1500张图）时随机下采样计算
        np.random.seed(42)
        indices = np.random.choice(len(pixel_scores), 100000000, replace=False)
        pixel_scores = pixel_scores[indices]
        pixel_masks = pixel_masks[indices]
        
    try:
        return roc_auc_score(pixel_masks, pixel_scores)
    except ValueError:
        return 0.0

def compute_pro(anomaly_maps, ground_truth_maps, max_step=200):
    """
    计算 Per-Region Overlap (PRO) (简化版)
    针对工业缺陷检测的常用评估指标
    """
    # 实际应用中PRO计算较慢且复杂，这里提供API预留和简单的像素阈值遍历代替
    # 推荐接入 msflow 或 PromptAD/utils/metrics.py 中的成熟PRO算法
    # 此处作为占位返回 0.0 以作示例
    return 0.0

def normalize_anomaly_map(anomaly_map):
    """
    Min-Max 标准化异常热力图
    """
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    return (anomaly_map - min_val) / (max_val - min_val + 1e-8)

def save_anomaly_heatmap(img_path, anomaly_map, save_dir):
    """
    将原始图片、热力图、叠加图并排保存，便于肉眼观察缺陷定位效果。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 提取文件名和异常分数地图
    base_name = os.path.basename(img_path)
    score_map = normalize_anomaly_map(anomaly_map)
    
    # 2. 读取原图
    orig_img = plt.imread(img_path)
    # 如果原图是 [0, 255] 之间，需要归一化到 [0, 1] 以便配合 matplotlib
    if orig_img.max() > 1.0:
        orig_img = orig_img / 255.0
        
    # 如果网络输出的尺度和原图不同 (通常都会将图resize为256)，需要用插值将图调整大小
    # 为了直观这里直接画图时利用 imshow 的自动缩放处理。但通常可以resize保持一致。
    
    # 将热力图转换到色图
    cmap = plt.get_cmap('jet')
    heatmap = cmap(score_map)
    heatmap = heatmap[:, :, :3] # 截取RGB通道，丢弃Alpha
    
    # 将原图与热力图按比例叠加 (0.5透明度)
    # 由于原图和热力图分辨率可能不同，直接在这借助 plt 画出并排的图片最好
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(score_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Anomaly Heatmap')
    axes[1].axis('off')

    axes[2].imshow(orig_img)
    axes[2].imshow(score_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    # 根据文件夹结构保存 (如果不同标签的话)
    parent_dir = os.path.basename(os.path.dirname(img_path))
    class_save_dir = os.path.join(save_dir, parent_dir)
    os.makedirs(class_save_dir, exist_ok=True)
    
    save_path = os.path.join(class_save_dir, base_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

