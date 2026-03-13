import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time

from dataset import GenADDataset
from model import PerStepGuidedDiffFlowAD
from model_idea2_ode import ProbabilityFlowODE_AD
from model_idea3_cascade import FeatureCascadedAD
from utils import compute_image_auroc, compute_pixel_auroc, save_anomaly_heatmap

def parse_args():
    parser = argparse.ArgumentParser(description='DiffFlow for Anomaly Detection - Test Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件所在路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='需评估的模型权重(.pt)完整路径')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger(log_dir, category):
    os.makedirs(log_dir, exist_ok=True)
    save_cat = category if isinstance(category, str) else "multi"
    log_file = os.path.join(log_dir, f"test_{save_cat}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('DiffFlow_Test')
    logger.setLevel(logging.INFO)
    
    # 文件处理器 (写入到文件)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def test():
    args = parse_args()
    config = load_config(args.config)
    
    # 解析配置
    c_data = config.get('dataset', {})
    c_model = config.get('model', {})
    c_train = config.get('train', {})
    c_test = config.get('test', {})

    dataset_path = c_data.get('dataset_path', '/root/autodl-tmp/mvtec_anomaly_detection')
    category = c_data.get('category', 'hazelnut')
    dataset_type = c_data.get('dataset_type', 'mvtec')
    meta_file = c_data.get('meta_file', None)
    
    batch_size = c_train.get('batch_size', 8)
    log_dir = c_train.get('log_dir', './logs')
    device = c_train.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    n_steps = c_model.get('n_steps', 50)
    guidance_scale = c_model.get('guidance_scale', 0.1)
    model_type = c_model.get('model_type', 'idea1')

    save_heatmaps = c_test.get('save_heatmaps', False)
    heatmap_dir = c_test.get('heatmap_dir', './heatmaps')
    save_cat_name = category if isinstance(category, str) else "multi_class"
    if save_heatmaps:
        heatmap_dir = os.path.join(heatmap_dir, save_cat_name)
        os.makedirs(heatmap_dir, exist_ok=True)

    logger = setup_logger(log_dir, category)

    logger.info(f"=== 启动 DiffFlow 独立测试 ===")
    logger.info(f"Dataset: {dataset_path} [{dataset_type}] | Category: {category} | Device: {device} | Model: {model_type}")
    logger.info(f"Loading checkpoint: {args.checkpoint}")

    # 1. 准备数据(只加载测试集)
    test_dataset = GenADDataset(dataset_path, category, dataset_type=dataset_type, is_train=False, meta_file=meta_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. 初始化模型并加载权重
    if model_type == 'idea1':
        model = PerStepGuidedDiffFlowAD(channels=3, guidance_scale=guidance_scale, n_steps=n_steps).to(device)
    elif model_type == 'idea2':
        model = ProbabilityFlowODE_AD(channels=3).to(device)
        model.n_steps = n_steps
    elif model_type == 'idea3':
        model = FeatureCascadedAD().to(device)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"找不到指定的权重文件: {args.checkpoint}")
        return
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_img_scores = []
    all_img_labels = []
    all_pixel_scores = []
    all_pixel_masks = []

    logger.info(f"开始测试，总批次数: {len(test_loader)}...")
    
    # 测试集前向推理，依赖 Flow 梯度所以手动保持局部梯度可计算
    pbar_test = tqdm(test_loader, desc=f"Testing {category}")
    for imgs, labels, masks, img_paths in pbar_test:
        imgs = imgs.to(device)
        
        # 调用引导推理过程 (内部具有 grad 计算)
        anomaly_maps, _ = model.forward_test(imgs)
        
        # 处理为numpy
        anomaly_maps_np = anomaly_maps.cpu().detach().numpy()
        masks_np = masks.numpy()
        
        # 图像级别的得分可以用热力图最大值代表
        image_scores = anomaly_maps_np.reshape(anomaly_maps_np.shape[0], -1).max(axis=1)
        
        # 如果开启了保存热力图，遍历 batch 内所有样本
        if save_heatmaps:
            for i in range(len(img_paths)):
                save_anomaly_heatmap(img_paths[i], anomaly_maps_np[i].squeeze(), heatmap_dir)

        all_img_scores.extend(image_scores)
        all_img_labels.extend(labels.numpy())
        
        all_pixel_scores.append(anomaly_maps_np)
        all_pixel_masks.append(masks_np)

    # 计算全局指标
    all_pixel_scores = np.concatenate(all_pixel_scores, axis=0)
    all_pixel_masks = np.concatenate(all_pixel_masks, axis=0)
    
    img_auroc = compute_image_auroc(all_img_scores, all_img_labels)
    
    try:
        pix_auroc = compute_pixel_auroc(all_pixel_scores, all_pixel_masks)
    except ValueError:
        pix_auroc = 0.0 
        logger.warning("数据集中可能全部为正常样本/无效掩码，无法计算Pixel AUROC")
        
    logger.info("=" * 40)
    logger.info(f"最终测试结果 - 类别: {category}")
    logger.info(f"Image AUROC: {img_auroc:.4f}")
    logger.info(f"Pixel AUROC: {pix_auroc:.4f}")
    logger.info("=" * 40)

if __name__ == '__main__':
    test()