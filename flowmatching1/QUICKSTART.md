# CFM Pipeline 快速启动

## 项目已创建！✓

项目位置: `~/flowmatching`

### 目录结构

```
flowmatching/
├── data/                          # 数据生成
│   └── synthetic_data_generator.py
├── models/                        # 模型核心
│   ├── stage1_featuremap.py       # Stage 1: 流形提取
│   ├── stage2_cfm.py              # Stage 2: CFM训练（核心）
│   ├── stage3_attribution.py      # Stage 3: 基因归因
│   └── components/
│       ├── bezier.py              # 三点Bézier曲线
│       └── losses.py              # 损失函数 (L_flow, L_geom, etc.)
├── validation/
│   └── metrics.py                 # 验证指标 (MMD, Wasserstein-2)
├── experiments/
│   ├── config.yaml                # 超参数配置
│   └── run_pipeline.py            # 主runner
├── requirements.txt               # 依赖
└── README.md                      # 详细文档
```

### 第一步：安装依赖

```bash
cd ~/flowmatching
pip install -r requirements.txt
```

### 第二步：运行完整pipeline

```bash
cd experiments
python run_pipeline.py --config config.yaml
```

这会自动执行：
- **Stage 0**: 生成合成数据
- **Stage 1**: 提取状态标签和转移分数
- **Stage 2**: 训练Bézier三锚点CFM模型
- **Stage 3**: 计算基因归因（核心驱动、动力学驱动、静态标记）
- **Stage 4**: 验证

### 第三步：测试单个模块

```bash
# 测试数据生成
python -m data.synthetic_data_generator

# 测试Bézier曲线
python -m models.components.bezier

# 测试损失函数
python -m models.components.losses

# 测试CFM模型
python -m models.stage2_cfm

# 测试Stage 1
python -m models.stage1_featuremap

# 测试归因
python -m models.stage3_attribution

# 测试指标
python -m validation.metrics
```

## 设计特点 ✨

### 1. **完全模块化**
每个Stage都是独立的，可以单独修改而不影响其他部分：
- `stage1_featuremap.py` ← 流形提取
- `stage2_cfm.py` ← CFM核心（最重要）
- `stage3_attribution.py` ← 后处理

### 2. **配置驱动**
所有超参数在 `config.yaml` 中，无需改代码：
```yaml
stage2:
  lambda_geom: 0.1       # L_geom权重
  lambda_context: 0.01   # L_context权重
  lambda_smooth: 0.001   # L_smooth权重
  learning_rate: 0.001
  n_epochs: 50
```

### 3. **三点Bézier CFM**
关键创新：
```
x(t) = (1-t)² x₀ + 2t(1-t) x_T + t² x₁
v(t) = 2(1-t)(x_T - x₀) + 2t(x₁ - x_T)
```
支持非单调转移（如先上升后下降的基因）

### 4. **四个损失函数**
```
L_total = L_flow + λ₁·L_geom + λ₂·L_context + λ₃·L_smooth
```
- `L_flow`: 预测速度 vs 目标Bézier速度
- `L_geom`: 在稳定区域惩罚运动
- `L_context`: 跨drug/cell-line一致性
- `L_smooth`: Jacobian正则化

### 5. **三路基因分解**
```
- G_flow ∩ G_DGV    → 核心驱动（同时被kinetics和DE捕捉）
- G_flow \ G_DGV    → 动力学驱动（新发现！）
- G_DGV \ G_flow    → 静态标记（已知DE基因）
```

## 接下来该做什么？

### 优先级 P1（这周）
1. **安装依赖** `pip install -r requirements.txt`
2. **运行toy pipeline** `python run_pipeline.py`
3. **检查loss曲线** 是否正常收敛

### 优先级 P2（下周）
1. **完善L_context** - 目前用简单方差最小化，需要因子分解：
   ```
   v_θ = v_shared(x,t) + Σ_c α_c · v_context_c(x,t)
   ```

2. **微调超参数** - 在config.yaml中调整λ₁,λ₂,λ₃

3. **消融实验** - 验证三点Bézier vs 二点线性确实有提升

### 优先级 P3（可选）
- 加载真实数据 (h5ad格式)
- 与scVelo对比
- 外部验证 (CRISPR screen, L1000)

## 项目理念

- **独立模块** → 易于调试和修改
- **配置驱动** → 易于复现和分享
- **逐阶段开发** → 先toy然后真实数据
- **完整pipeline** → 一键运行所有stages

---

**现在可以开始了！** 🚀

问题？检查README.md或运行各个模块的内部测试。
