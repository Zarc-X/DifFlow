# Guided DiffFlow for Anomaly Detection (difflow)

`difflow` 是一套专注于工业表面缺陷检测（Anomaly Detection, AD）的创新型生成式框架。该仓库包含了一种全新的前沿探索机制：结合 **Diffusion 扩散模型**的高频重建能力与 **Normalizing Flow 流模型**的精确概率测度能力。

本项目的核心理论是通过**每步交替结合（Per-step Guidance）**，即利用前向加噪破坏异常结构后，在 U-Net 反向降噪的每一步中，显式调用 Flow 模型的似然偏导数（Likelihood Gradient）对当前分布进行“强引力修偏”，利用这个引力梯度的累积量作为精准的像素级异常得分。

---

## 📂 项目目录结构

```text
difflow/
│
├── dataset.py    # 通用异常检测数据集构建器 (包含差异化解析)
├── model.py      # 模型核心构建模块 (UNet, Flow及其融合层)
├── train.py      # 模型端到端训练与验证主入口 (含超参数解析)
├── utils.py      # 运算工具函数 (含AUROC计算、归一化等)
└── README.md     # 当前说明文档
```

---

## 🧠 模型结构介绍

本项目目前采用的是 **PerStepGuidedDiffFlowAD** 双塔融合架构 (详见 `model.py`):

1.  **DiffUNet分支 (结构修复器)**
    *   基础的去噪网络，输入带有时间步 $t$ 条件的带噪图片。
    *   **职责**：负责拟合马尔可夫链的步长推演，在测试阶段对带瑕图提供基础的去噪残差预测。

2.  **FlowLikelihoodEstimator分支 (概率巡查官)**
    *   本质为 Normalizing Flow，通过训练在无异常数据上建立平滑的高斯隐向量空间表征。
    *   **职责**：输出对单个样本或特征的精确对数似然 $\log P(x)$。

3.  **核心融合逻辑 (测试阶段推断)**
    *   加入微量噪声并迭代 $t \rightarrow t-1$ 步进。
    *   **在每一步**都获取 Flow 针对于当前图像的“纠偏力度”（即：使概率密度增大的梯度向量）：$\nabla_{x_t} \log P_{flow}(x_t)$。
    *   异常区域将被 Flow 持续打低分并产生极大的梯度引导（阻力）。这股连续积分的阻力最终合并作为输出的 `Anomaly Score Map`。

---

## 🚀 操作指南

### 1. 依赖准备
项目使用了极其原生的 PyTorch，请确保已安装以下依赖（与现有的 `msflow` 或 `realnet` 环境通常是兼容兼容的）：
```bash
pip install torch torchvision numpy scikit-learn tqdm pillow
```

### 2. 开始训练与验证
训练程序 `train.py` 使用了 `argparse` 进行统一传参。它能够一键式处理训练并在设定的 Epoch 后执行评测。

#### 场景 1: 使用 MVTec AD 数据集
```bash
python train.py \
    --dataset_type mvtec \
    --dataset_path "/path/to/your/mvtec" \
    --category "hazelnut" \
    --epochs 100 \
    --batch_size 8 \
    --lr 2e-4
```

#### 场景 2: 使用 VisA 数据集
对于多级文件夹嵌套和标签处理有所不同的 VisA：
```bash
python train.py \
    --dataset_type visa \
    --dataset_path "/path/to/your/VisA/1cls" \
    --category "candle"
```

#### 场景 3: 使用 Real-IAD 数据集
可以直接挂载该数据集所使用的 Jsonl 元数据清单文件：
```bash
python train.py \
    --dataset_type real_iad \
    --meta_file "/path/to/real-iad/train_meta.jsonl"
```

### 3. 可调关键超参数 (`train.py`)
*   `--lambda_flow`: (默认 1.0) 训练时控制 Flow `LogLikelihood` Loss 与 Diffusion `MSE` Loss 之间的平衡权重。
*   `--guidance_scale`: (默认 0.1) **核心测试参数**。控制推理阶段 Flow 梯度引力对降噪强度的物理约束能力。
*   `--n_steps`: (默认 50) 总体去噪步数。如果发现生成速度过慢，可调小该阈值使用大步长更新。

---

## 📝 进阶开发提醒 (TODOs)
*   **网络扩容**：当前 `model.py` 的 `DiffUNet` 以及 `FlowLikelihoodEstimator` 由极简卷积组代替。在正式对比 benchmark 之前，**强烈建议**将其替换为你在工作区 `msflow/models/flow_models.py` 中现成封装的 Freia (RealNVP 组件) 以及 `realnet` 里的标准多头注意力 UNet。
*   **计算效能**：测试阶段需要对输入图片求偏导数 (`requires_grad_()`) 且保留计算图切片，当增大 batch size 时会显著占用显存。
*   **指标评估**：`utils.py` 只留空了 PRO（Per-Region Overlap）分数的 API。如有刚需可接入 `PromptAD` 的评价脚本进行联合评估。
