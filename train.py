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
from models.idea1_heuristic import PerStepGuidedDiffFlowAD
from models.idea2_ode import ProbabilityFlowODE_AD
from models.idea3_cascade import FeatureCascadedAD
from models.idea4_simple import SimpleGuidedDiffFlowAD
from models.idea5_msflow_diffusion import DiffusionMSFlowAD
from utils import compute_image_auroc, compute_pixel_auroc, compute_best_f1, compute_average_precision, normalize_anomaly_map

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
    log_file = os.path.join(log_dir, f"train_{save_cat}.log")
    
    # 避免多个类别训练时日志处理器重复累加，每次清空之前的handlers
    logger = logging.getLogger('DiffFlow')
    if logger.hasHandlers():
        logger.handlers.clear()
        
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
    
    # 防止日志向上传递到 root logger 造成可能的双重打印
    logger.propagate = False 
    
    return logger

def train_category(args, config, category_name, run_timestamp):
    # 解析配置
    c_data = config.get('dataset', {})
    c_model = config.get('model', {})
    c_train = config.get('train', {})

    dataset_path = c_data.get('dataset_path', '/root/autodl-tmp/mvtec_anomaly_detection')
    category = category_name  # 使用传入的单一类别
    dataset_type = c_data.get('dataset_type', 'mvtec')
    meta_file = c_data.get('meta_file', None)
    
    epochs = c_train.get('epochs', 100)
    batch_size = c_train.get('batch_size', 8)
    lr = c_train.get('lr', 2e-4)
    lambda_flow = c_train.get('lambda_flow', 1.0)
    device = c_train.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    n_steps = c_model.get('n_steps', 50)
    guidance_scale = c_model.get('guidance_scale', 0.1)
    model_type = c_model.get('model_type', 'idea1')

    save_cat_name = category if isinstance(category, str) else "multi_class"

    # 新的结果层级结构: ./results/{run_timestamp}/{category}/
    results_base_dir = c_train.get('results_dir', './results')
    run_dir = os.path.join(results_base_dir, f"run_{run_timestamp}", save_cat_name)
    
    save_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存本次训练的配置参数
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    logger = setup_logger(log_dir, category)

    logger.info(f"=== 启动 DiffFlow 训练 ===")
    logger.info(f"Dataset: {dataset_path} [{dataset_type}] | Category: {category} | Device: {device} | Model: {model_type}")
    logger.info(f"Results will be saved to: {run_dir}")

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
    elif model_type == 'idea4':
        model = SimpleGuidedDiffFlowAD(channels=3, guidance_scale=guidance_scale, n_steps=n_steps).to(device)
    elif config['model']['model_type'] == 'idea5':
        from models.idea5_msflow_diffusion import DiffusionMSFlowAD
        model = DiffusionMSFlowAD(channels=3, n_steps=config['model'].get('n_steps', 1000)).to(device)
    elif config['model']['model_type'] == 'idea7':
        from models.idea7_diffusion_flow import DiffusionFlowAD
        model = DiffusionFlowAD(channels=3, n_steps=config['model'].get('n_steps', 1000)).to(device)
    elif config['model']['model_type'] in ['idea6', 'idea7']:
        from models.idea6_decoupled import DecoupledDiffFlowAD
        model = DecoupledDiffFlowAD(channels=3, n_steps=config['model'].get('n_steps', 1000)).to(device)
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
            if config['model']['model_type'] in ['idea6', 'idea7']:
                loss_diff, loss_flow = model.forward_train(imgs, epoch=epoch)
            else:
                loss_diff, loss_flow = model.forward_train(imgs)
            loss = loss_diff + lambda_flow * loss_flow
            
            loss.backward()
            optimizer.step()
            
            total_loss_diff += loss_diff.item()
            total_loss_flow += loss_flow.item()
            
            pbar.set_postfix({'L_Diff': total_loss_diff/len(train_loader), 'L_Flow': total_loss_flow/len(train_loader)})

        # 4. 评估验证循环 (每 5 个 Epoch 验证一次以节省时间)
        if epoch % 10 == 0 or epoch == epochs:
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
            
            # --- 加入工业界标准 PyTorch 测速 (CUDA Event) ---
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = []
            
            pbar_test = tqdm(test_loader, desc=f"Epoch {epoch} [Test]")
            for step, (imgs, labels, masks, img_paths) in enumerate(pbar_test):
                imgs = imgs.to(device)
                
                # 开始记录推理时间（GPU计算起步）
                starter.record()
                
                # 核心推理代码（这部分耗时将直接反映给用户）
                with torch.no_grad(): # 确认idea7等纯前向架构不需要推理回传，可以直接使用no_grad加速
                    anomaly_maps, recon_imgs = model.forward_test(imgs)
                
                # 记录推理结束并等待所有 CUDA 核心执行流完成
                ender.record()
                torch.cuda.synchronize()
                
                # 第一步 (step == 0) GPU 会有加载核函数的 Warm-up 延迟，不计入测速
                if step > 0:
                    curr_time_ms = starter.elapsed_time(ender)
                    # 将时间均摊到 batch size 的那张图片上
                    timings.append(curr_time_ms / imgs.shape[0])
                
                # 处理为numpy
                anomaly_maps_np = anomaly_maps.cpu().detach().numpy()
                recon_imgs_np = recon_imgs.cpu().detach().numpy() if recon_imgs is not None else None
                masks_np = masks.numpy()
                
                # 【防污染核心屏障】：消除卷层边缘 Padding 的畸高噪点并防止其干扰 Pixel 评价以及在滤波时向内部渗透
                for b in range(anomaly_maps_np.shape[0]):
                    valid_min = anomaly_maps_np[b, :, 16:-16, 16:-16].min()
                    anomaly_maps_np[b, :, :16, :] = valid_min
                    anomaly_maps_np[b, :, -16:, :] = valid_min
                    anomaly_maps_np[b, :, :, :16] = valid_min
                    anomaly_maps_np[b, :, :, -16:] = valid_min
                
                # 【平滑处理】：为异常预测图添加高斯平滑滤波（工业缺陷检测标准流程）。
                # MSFlow、FastFlow 等必须依靠平滑来消除由 CNN 产生的孤立极高噪点，
                # 否则会导致后续取 max 作为 image score 时，正常图片也被误判为极高分。
                from scipy.ndimage import gaussian_filter
                for i in range(anomaly_maps_np.shape[0]):
                    # 动态选择后处理模糊强度：idea4 为像素级高频重构误差，只需极小的 sigma 去噪；其他特征级模型需要强平滑
                    if config['model']['model_type'] == 'idea4':
                        anomaly_maps_np[i, 0] = gaussian_filter(anomaly_maps_np[i, 0], sigma=1)
                    else:
                        anomaly_maps_np[i, 0] = gaussian_filter(anomaly_maps_np[i, 0], sigma=4)
                    
                    # 在最后一个 Epoch，保存热力图用于可视化验证
                    if epoch == epochs:
                        from utils import save_anomaly_heatmap
                        hm_save_dir = os.path.join(run_dir, 'heatmaps')
                        save_anomaly_heatmap(img_paths[i], anomaly_maps_np[i, 0], hm_save_dir, recon_img=recon_imgs_np[i] if recon_imgs_np is not None else None)
                
                # 图像级别的得分直接取绝对最大值 max() 会受到卷积网络边缘 Padding 产生的畸高噪点影响。
                # 之前使用 sort 取 Top-K 依然无法避开全部受污染的边缘像素。
                # 【最终修复】：我们通过裁剪掉边缘 16 个像素（Padding 影响区），然后再取中心有效区域的全局最大值，从而获得真正有区分度的 Image Score。
                valid_maps = anomaly_maps_np[:, :, 16:-16, 16:-16]
                image_scores = valid_maps.reshape(valid_maps.shape[0], -1).max(axis=1)
                
                all_img_scores.extend(image_scores)
                all_img_labels.extend(labels.numpy())
                
                all_pixel_scores.append(anomaly_maps_np)
                all_pixel_masks.append(masks_np)

            # 计算 指标
            all_pixel_scores = np.concatenate(all_pixel_scores, axis=0)
            all_pixel_masks = np.concatenate(all_pixel_masks, axis=0)
            
            img_auroc = compute_image_auroc(all_img_scores, all_img_labels)
            img_f1 = compute_best_f1(all_img_labels, all_img_scores)
            img_ap = compute_average_precision(all_img_labels, all_img_scores)
            
            # 使用简单的像素 AUROC (如果显存或内存爆炸，工业界通常做下采样后计算)
            # 在某些纯无掩码的数据集上可能异常，可以增加异常捕获
            try:
                pix_auroc = compute_pixel_auroc(all_pixel_scores, all_pixel_masks)
                
                # 计算像素级最佳 F1 时同样需要做安全下采样
                pm_flat = (all_pixel_masks.flatten() > 0.5).astype(int)
                ps_flat = all_pixel_scores.flatten()
                if len(ps_flat) > 100000000:
                    np.random.seed(42)
                    indices = np.random.choice(len(ps_flat), 100000000, replace=False)
                    ps_flat = ps_flat[indices]
                    pm_flat = pm_flat[indices]
                pix_f1 = compute_best_f1(pm_flat, ps_flat)
                pix_ap = compute_average_precision(pm_flat, ps_flat)
            except ValueError:
                pix_auroc = 0.0 # 当只传入纯净样本无掩码时会抛出
                pix_f1 = 0.0
                pix_ap = 0.0
                
            # 给出最终算力性能日志
            avg_ms = np.mean(timings) if len(timings) > 0 else 0
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            logger.info(
                f"[Epoch {epoch} Results] Img AUROC: {img_auroc:.4f}, Img AP: {img_ap:.4f}, Img F1: {img_f1:.4f} | "
                f"Pix AUROC: {pix_auroc:.4f}, Pix AP: {pix_ap:.4f}, Pix F1: {pix_f1:.4f} | "
                f"Speed: {avg_ms:.2f} ms/img ({fps:.1f} FPS)"
            )

            if img_auroc > best_img_auroc:
                best_img_auroc = img_auroc
                save_path = os.path.join(save_dir, f'difflow_best_{save_cat_name}.pt')
                torch.save(model.state_dict(), save_path)
                logger.info(f"已保存最佳模型至: {save_path}")
                
            if epoch == epochs:
                save_path_last = os.path.join(save_dir, f'difflow_last_{save_cat_name}.pt')
                torch.save(model.state_dict(), save_path_last)
                logger.info(f"已保存最终 Epoch 模型至: {save_path_last}")

    # ==========================
    # 防止多类别循环训练时导致 OOM
    # 释放当前类别的模型及数据缓存
    # ==========================
    try:
        del model, optimizer, train_loader, test_loader
    except NameError:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 每次运行 train.py 生成一个统一时间的运行批次标识
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    c_data = config.get('dataset', {})
    categories = c_data.get('category', 'hazelnut')
    
    if isinstance(categories, list):
        for cat in categories:
            print(f"\n{'='*50}\n>>> 开始独立类别模型训练: {cat} <<<\n{'='*50}")
            train_category(args, config, cat, run_timestamp)
    elif categories == 'all':
        dataset_path = c_data.get('dataset_path', '/root/autodl-tmp/mvtec_anomaly_detection')
        if os.path.exists(dataset_path):
            all_cats = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            for cat in all_cats:
                print(f"\n{'='*50}\n>>> 开始独立类别模型训练: {cat} <<<\n{'='*50}")
                train_category(args, config, cat, run_timestamp)
        else:
            print(f"数据集目录不存在: {dataset_path}")
    else:
        print(f"\n{'='*50}\n>>> 开始独立类别模型训练: {categories} <<<\n{'='*50}")
        train_category(args, config, categories, run_timestamp)

if __name__ == '__main__':
    main()
