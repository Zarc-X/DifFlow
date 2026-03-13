import os
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time

from dataset import GenADDataset
from model import PerStepGuidedDiffFlowAD
from model_idea2_ode import ProbabilityFlowODE_AD
from model_idea3_cascade import FeatureCascadedAD
from utils import compute_image_auroc, compute_pixel_auroc, normalize_anomaly_map

def parse_args():
    parser = argparse.ArgumentParser(description='DiffFlow for Anomaly Detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件所在路径')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger(log_dir, category):
    os.makedirs(log_dir, exist_ok=True)
    save_cat = category if isinstance(category, str) else "multi"
    log_file = os.path.join(log_dir, f"train_{save_cat}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('DiffFlow')
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

def train():
    args = parse_args()
    config = load_config(args.config)
    
    # 解析配置
    c_data = config.get('dataset', {})
    c_model = config.get('model', {})
    c_train = config.get('train', {})

    dataset_path = c_data.get('dataset_path', '/root/autodl-tmp/mvtec_anomaly_detection')
    category = c_data.get('category', 'hazelnut')
    dataset_type = c_data.get('dataset_type', 'mvtec')
    meta_file = c_data.get('meta_file', None)
    
    epochs = c_train.get('epochs', 100)
    batch_size = c_train.get('batch_size', 8)
    lr = c_train.get('lr', 2e-4)
    lambda_flow = c_train.get('lambda_flow', 1.0)
    save_dir = c_train.get('save_dir', './checkpoints')
    log_dir = c_train.get('log_dir', './logs')
    device = c_train.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    n_steps = c_model.get('n_steps', 50)
    guidance_scale = c_model.get('guidance_scale', 0.1)
    model_type = c_model.get('model_type', 'idea1')

    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_dir, category)

    logger.info(f"=== 启动 DiffFlow 训练 ===")
    logger.info(f"Dataset: {dataset_path} [{dataset_type}] | Category: {category} | Device: {device} | Model: {model_type}")

    # 对于展示目的，如果 category 是列表或者 all，将其简写保存
    save_cat_name = category if isinstance(category, str) else "multi_class"
    if save_cat_name == 'all': save_cat_name = "all_class"

    # 1. 准备数据
    train_dataset = GenADDataset(dataset_path, category, dataset_type=dataset_type, is_train=True, meta_file=meta_file)
    test_dataset = GenADDataset(dataset_path, category, dataset_type=dataset_type, is_train=False, meta_file=meta_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. 初始化模型与优化器
    if model_type == 'idea1':
        model = PerStepGuidedDiffFlowAD(channels=3, guidance_scale=guidance_scale, n_steps=n_steps).to(device)
    elif model_type == 'idea2':
        model = ProbabilityFlowODE_AD(channels=3).to(device)
        model.n_steps = n_steps
    elif model_type == 'idea3':
        model = FeatureCascadedAD().to(device)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_img_auroc = 0.0

    # 3. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_diff = 0
        total_loss_flow = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            # 获取训练Loss: Diffusion(MSE) 和 Flow(-LogP)
            loss_diff, loss_flow = model.forward_train(imgs)
            loss = loss_diff + lambda_flow * loss_flow
            
            loss.backward()
            optimizer.step()
            
            total_loss_diff += loss_diff.item()
            total_loss_flow += loss_flow.item()
            
            pbar.set_postfix({'L_Diff': total_loss_diff/len(train_loader), 'L_Flow': total_loss_flow/len(train_loader)})

        # 4. 评估验证循环 (每 5 个 Epoch 验证一次以节省时间)
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            all_img_scores = []
            all_img_labels = []
            all_pixel_scores = []
            all_pixel_masks = []

            logger.info("开始测试集评估...")
            with torch.no_grad(): # 测试阶段依然利用反向传播机制获取梯度的技巧，由于之前代码显式要求 requires_grad_，这里要开启grad计算引导
                # 因此这里我们临时覆盖 no_grad 限制给模型内的 forward_test 开启所需梯度
                pass 
            
            # 【重要】为了使 Flow 向图像求偏导生效，测试时不能将包裹在外层的 with torch.no_grad() 写死
            # 下方采取手动管理 grad
            pbar_test = tqdm(test_loader, desc=f"Epoch {epoch} [Test]")
            for imgs, labels, masks, img_paths in pbar_test:
                imgs = imgs.to(device)
                
                # 调用引导推理过程 (内部具有 grad 计算)
                anomaly_maps, _ = model.forward_test(imgs)
                
                # 处理为numpy
                anomaly_maps_np = anomaly_maps.cpu().detach().numpy()
                masks_np = masks.numpy()
                
                # 图像级别的得分可以简单取热力图最大值或均值
                image_scores = anomaly_maps_np.reshape(anomaly_maps_np.shape[0], -1).max(axis=1)
                
                all_img_scores.extend(image_scores)
                all_img_labels.extend(labels.numpy())
                
                all_pixel_scores.append(anomaly_maps_np)
                all_pixel_masks.append(masks_np)

            # 计算 指标
            all_pixel_scores = np.concatenate(all_pixel_scores, axis=0)
            all_pixel_masks = np.concatenate(all_pixel_masks, axis=0)
            
            img_auroc = compute_image_auroc(all_img_scores, all_img_labels)
            
            # 使用简单的像素 AUROC (如果显存或内存爆炸，工业界通常做下采样后计算)
            # 在某些纯无掩码的数据集上可能异常，可以增加异常捕获
            try:
                pix_auroc = compute_pixel_auroc(all_pixel_scores, all_pixel_masks)
            except ValueError:
                pix_auroc = 0.0 # 当只传入纯净样本无掩码时会抛出
                
            logger.info(f"[Epoch {epoch} Results] Image AUROC: {img_auroc:.4f} | Pixel AUROC: {pix_auroc:.4f}")

            if img_auroc > best_img_auroc:
                best_img_auroc = img_auroc
                save_path = os.path.join(save_dir, f'difflow_best_{save_cat_name}.pt')
                torch.save(model.state_dict(), save_path)
                logger.info(f"已保存最佳模型至: {save_path}")

if __name__ == '__main__':
    train()
