# p3audit — 论文三代码库

**Do physiological shortcut audits work for EEG decoders? A semi-synthetic ground-truth test with a natural counterfactual anchor**

本代码库按《论文三详细写作大纲 v8.1 合并版》实现：半合成生成器、四个核心模型与 LaBraM 补充、编码/依赖三类审计指标、真值与信度、H1–H5 统计、第一至第三批预实验（P1–P12），以及 PhysioNet-MI / SHU-MI 真实审计。所有数值设定集中在 `configs/default.yaml`，其中仍待预实验确定的值登记在 `tbd_registry`。

---

## 1. 安装与自检

```bash
pip install -e .[dev]           # numpy scipy scikit-learn pandas pyyaml mne torch statsmodels
pytest -q                       # 18 个单元测试（线性叠加精确性、LEACE、IG 完备性、耦合、访问守卫等）
p3audit tbd                     # 列出全部 TBD 值及其来源预实验
```

合成数据全流程试跑（不需要下载数据，CPU 几十分钟，数字无意义，只验证代码路径）：

```bash
p3audit --config configs/dryrun.yaml --synthetic 14 prepare --stage all
p3audit --config configs/dryrun.yaml --synthetic 14 pilot P2      # 其余同理：P6 P1a sensitivity P7 P3 P4 P8 P9
p3audit --config configs/dryrun.yaml --synthetic 14 run-units --max-epochs 3
p3audit --config configs/dryrun.yaml --synthetic 14 real-audit --max-epochs 2
p3audit --config configs/dryrun.yaml --synthetic 14 analyse
```

## 2. 真实数据的执行顺序（对应第 7 节时间表）

physionet.org 在云端可能无法访问（大纲第 7 节外部依赖），请在本地或已开白名单的机器上运行；`mne.datasets.eegbci` 会按受试者与 run 下载到 `paths.data_root`。

| 阶段 | 命令 | 产出 / 需要回填的配置 |
|---|---|---|
| 第 1 个月 · 第一批 | `scripts/run_first_batch.sh`（split → P2 → P6 → components → P1a → sensitivity → P7；P3 并行） | `results/pilots/*.json`；回填 `background.*`、`saccade.amplitude_uv`、`task_component.reg` 等 |
| | `p3audit prepare --stage cache` | 每名受试者的基函数缓存（见 §4.1） |
| 第 2 个月 · 第二批 | `pilot P4` → `pilot P5`（仅 P4 失败时）→ `pilot P8` → `pilot P9` → `pilot P10a --npz ...` → `pilot P12` | 回填 `s_star/s_low/s_high`、`a_star`、`ig.windows_per_unit`、SESOI 与各阈值 |
| 预注册 | 把已定值写入 `confirmed:` → `p3audit freeze` | `results/prereg_snapshot.json`（配置哈希、代码哈希、划分、预实验看过的数据清单） |
| 第 3–6 个月 | `p3audit --confirmatory run-units [--shard i/n]` | `results/units/<uid>.json` |
| 第 6–8 个月 | `p3audit --confirmatory real-audit`，`shu-audit` | `results/real/`、`results/shu/` |
| 分析 | `p3audit analyse` | `results/analysis/H1…H5.json` 与 CSV |

`--confirmatory` 在任何 TBD 值尚未列入 `confirmed` 时拒绝运行。`run-units` 支持 `--module/--model/--confound/--shard` 过滤，结果按单元落盘、可断点续跑。

## 3. 大纲章节 → 代码

| 大纲 | 模块 |
|---|---|
| 3.1 数据与排除规则（只读元数据） | `data/sources.py`（`RunSource.header` 不读信号）、`data/exclusion.py`（R1–R4 可执行规则） |
| 3.1 "测试受试者信号不被读取" | `utils/access.py`：每次读信号都登记用途；预实验用途只允许预实验受试者，生成器用途只允许 run 1/2（Alpha）与 3/7/11（mu），日志进入预注册快照 |
| 3.2 受试者划分 | `data/splits.py` |
| 3.3 四条信号流 | `data/streams.py` |
| 3.4.1 背景窗口、伪标签 | `data/windows.py` |
| 3.4.2 任务信号（mu 成分调制） | `generator/components.py`（逐受试者 GED + 纳入检查，含 20–40 Hz 不单调上升）、`generator/inject.py` |
| 3.4.3 后部 Alpha、重叠、拓扑离散度 | `generator/components.py`、`generator/overlap.py`、`generator/cache.py`（σ_α） |
| 3.4.4 扫视阶跃 + 尖峰电位 | `generator/saccade_ica.py`（拆半 ICA、置换零分布、失败规则）、`generator/inject.py` |
| 3.4.5 耦合、析因、失配、组模板对照 | `generator/design.py`、`generator/semisynth.py` |
| 3.4.6 成对测试集与真值、拆半 | `metrics/truth.py`、`data/windows.half_assignment` |
| 3.4.7 标定 | `experiments/calibration.py` |
| 3.5 模型 | `models/convnets.py`（EEGNet-8,2、ShallowConvNet、CSOANet 占位、原始特征线性参照）、`models/foundation.py`（CBraMod/LaBraM 适配器）、`training/trainer.py` |
| 3.6.1 编码（探针，主口径 + 受试者内口径 + PCA） | `metrics/probe.py` |
| 3.6.2 依赖：残差擦除 / SRI + HEOG 回归 / IG | `metrics/erasure.py`、`metrics/intervention.py`、`metrics/attribution.py` |
| 3.7 自然反事实、灵敏度曲线、真实审计 | `saccades/detect.py`、`saccades/sensitivity.py`、`experiments/real_audit.py`、`experiments/shu_audit.py` |
| 3.8 统计 | `stats/inference.py`、`stats/suffstats.py`、`stats/power.py` |
| 第 2 节 H1–H5 判定 | `experiments/analysis.py` |
| 第 6 节预实验 | `experiments/pilots.py`、`experiments/calibration.py` |
| 6.3 预注册前事项 | `p3audit tbd`、`p3audit freeze` |

## 4. 关键实现决定

### 4.1 注入在原始域进行，按线性叠加精确实现

四条信号流（重采样、FIR、陷波、平均参考）都是线性且在 4 样本网格上平移等变的算子，所以 `stream(x + Δ) = stream(x) + stream(Δ)`。缓存阶段对每个背景窗口在 160 Hz 原始域计算单位注入 Δ（任务 mu 左/右、Alpha、扫视左/右，以及失配与组模板版本），分别通过模型输入流，得到"基函数"；任一实验条件都是

`X = X0 + (−s)·B_mu[y] + (exp(kσ_α/2) − 1)·z·B_alpha + amp·z·B_sacc`

这与"先注入、再滤波和重参考"在数学上完全相同（`tests/test_core.py::test_stream_linearity_exact` 验证误差 < 1e-5），同时避免 4 秒窗口上的滤波边缘失真，并让 p、s、幅度、测试集类型变化时无需重新滤波。窗口带 1 秒填充，干预（带阻）在填充后的窗口上做再裁剪。

代价是磁盘：4 秒窗口、全部附加基函数时约 30 GB（`cache_dtype: float16` 可减半）。

### 4.2 每个单元保存逐受试者充分统计量

所有 BA 类量（真值、ΔBA_keep、SRI、ΔBA_reg、探针 D）都按受试者存混淆计数，IG 按受试者存相关性求和，并各自分全体 / 半集 0 / 半集 1。因此分层 bootstrap（先种子、后受试者）、拆半信度、交叉拟合（真值与指标在不同半集上计算后互换）都能在分析阶段精确重算，不需要重跑模型。

## 5. 需要你确认的设计问题（实现时发现，按重要性排序）

1. **H3 的重叠调节变量在"单元"层面没有变异。** 大纲把"单元的平均 |corr(a_mu, a_α)|"作为连续调节变量放进 H3 模型，但所有单元共用同一批 34 名测试受试者，单元平均重叠在所有单元间是同一个常数，模型无法估计其斜率。代码改为在"受试者 × 单元"层面进入（逐受试者的真值与指标，从计数恢复），见 `analysis.analyse_h3` 的 `overlap_moderation`。大纲 2、3.4.3、4.4 的表述需要相应修改。
2. **回归量污染扫描在计算量表里记为 16 次训练，但它只改变指标，不改变模型输入。** 当前实现是指标层面的：`r = AF7−AF8 + c·(sd_r/sd_m)·(C3−C4)`，c ∈ {0, 0.1, 0.25}，不额外训练。如果你设想的是"生成器让 HEOG 导联混入任务信号、模型在其上训练"，需要新增一个基函数和 16 次训练。
3. **扫视失配的含义。** 实现为：训练与验证窗口使用主模板池，失配单元的测试窗口改用留出的预实验模板（`saccade.mismatch_pool_n = 4`）并叠加偶极方向旋转与额/颞权重比扰动；Alpha 峰频 ±2 Hz 失配则训练与测试都偏移。这样 HEOG 回归系数（在训练窗口上估计）面对的是分布外的扫视拓扑，正是检验其循环性的条件。
4. **扫视的 z 定义。** 默认 `mode: presence`（z = 1 注入一次右向扫视，z = 0 不注入），与"注入指示"的字面意思一致；`mode: direction`（z = 1 右、z = 0 左）已实现，更接近真实左右拳任务中方向与标签耦合的情形。二者对依赖指标的难度不同，建议预注册前定下。
5. **P8 与标定不使用测试受试者。** P8 以 10 名验证受试者为评估集、从非预实验训练受试者中划出早停集；P4/P5 同样在验证受试者上评估。这满足"标定只使用训练与验证受试者"，但验证受试者在标定中既用于早停又用于评估的问题因此被消除了，代价是标定用的训练集略小。
6. **H1 的探针在每个测试幅度上分别拟合**（在训练受试者的同幅度嵌入上拟合、冻结后用于测试受试者），而不是在 a\* 上拟合一次再用于所有幅度。后者会让低幅度下的检出阈值偏高。
7. **组模板对照的空间滤波器**：取预实验受试者归一化 w 的平均，再缩放使 wᵀa_group = 1。大纲只规定了组平均 a。
8. **HEOG 回归系数的估计方式需要预注册。** AF7−AF8 不是真正的 EOG 导联，在全部样本上做普通最小二乘时，系数几乎完全由 EEG 背景协方差决定，漏掉大部分眼动场（合成数据上 F7 的系数约 0.01，理想值约 0.57）。默认改为"扫视锁定"估计：在回归量上检测阶跃，用前后 100 ms 的阶跃向量估计传播系数；OLS 保留为敏感性结果（`heog["ols"]`）。即便如此，合成数据上 ΔBA_reg 仍明显低于真值（见 §6.1），这可能正是论文要报告的失效模式之一，也可能部分来自合成模板的噪声，需要在真实预实验上看。
9. **IG 的 delta 频段包含直流。** 4 秒窗口内的扫视阶跃，大部分能量落在 0–0.5 Hz 的 FFT 频点上。若 delta 从 0.5 Hz 开始，这部分相关性会落进"其余"频段，扫视的 IG 份额反而随 p 下降。现已设为 [0, 4] Hz。
10. **计算量与表 7 的差异。** `p3audit grid` 当前列出 358 次半合成训练，表 7 对应部分约 373 次；差异来自 LaBraM 的范围（配置 `design.supplementary`，默认仅 Alpha 混淆、p 序列 15 次 + 析因 5 个单元 × 2 种子）与去重后的中心单元。请按你的预算调整该配置。

## 6. 必须由你补上或核实的部分

- **CSOANet**：`models/convnets.py::CSOANetPlaceholder` 只是占位，重现了大纲写明的三支 72/200/776 ms 感受野，替换为已发表实现时保持 `features()` 与 `head` 接口不变即可。
- **CBraMod / LaBraM**：`models/foundation.py` 按官方仓库的布局从 `foundation.<name>.repo_path` 导入并加载 `checkpoint`；输入单位（µV × 0.01）、类名与 `forward_features` 签名均标为 [待核实]。未配置时只能用 `--fallback-standin` 跑通流程，这样产生的结果在分析阶段会被自动剔除。
- **原始参考方式**（`physionet.original_reference`）：尖峰电位模板换算参考时需要，目前为空，代码会跳过换算。
- **尖峰电位模板**（`inject.spike_template`）：按 Keren 等（2010）的定性描述参数化，是一项假设。
- **扫视检测器**：采用匹配阶跃统计量 d(t) = 后 100 ms 均值 − 前 100 ms 均值，阈值 max(k·MAD, 最小幅度)；在合成数据上比速度阈值稳健得多。k 与最小幅度须在预实验受试者上确定（TBD）。
- **SHU-MI 与 EEGEyeNet 加载器**（`data/external.py`）：按公开说明编写，文件名、键名、通道表由 P11、P10a 核实。
- **specparam**：`metrics/spectral.aperiodic_exponent` 是对数-对数直线拟合的简化版；若预注册写 specparam，安装 `pip install -e .[spectral]` 后替换。

### 6.1 合成数据上的已验证内容

- 全流程已在 14 名合成受试者上跑通：准备阶段、第一批全部预实验、P4、P9、56 个半合成单元（0 失败）、真实审计 5 折 + 偏差对照模型、H1–H5 分析。
- 指标恢复检查（`scripts/sanity_metric_recovery.py`，EEGNet，扫视混淆，s = 0.3）：

| p | ΔBA_neu | ΔBA_rev | ΔBA_keep（擦除） | IG 份额 | ΔBA_reg（扫视锁定 / OLS） |
|---|---|---|---|---|---|
| 0.5 | −0.004 | −0.012 | −0.012 | 0.084 | −0.020 / −0.016 |
| 0.9 | 0.401 | 0.802 | 0.258 | — | 0.060 / 0.020 |
| 1.0 | 0.484 | 0.980 | 0.000（不可识别） | 0.209 | 0.040 / 0.000 |

  真值落在理论上限（p − 0.5，2p − 1）附近；残差擦除在 p < 1 时跟随真值、在 p = 1 时归零，与 3.6.2 表中"p = 1 时不可识别"一致；IG 份额随依赖上升；HEOG 回归明显低估（见 §5 第 8 条）。
- 合成数据中，真实审计的自然反事实分析检出了人为放入的扫视捷径：一致试次对无扫视试次的 GLMM 系数约 2.4（logit）。
- 这些只是代码正确性的检查，合成信号的参数并不真实，不代表真实数据上的结论。

## 7. 结果文件

```
results/
  exclusion.csv  split.json  signal_access_log.jsonl  prereg_snapshot.json
  components/sub-XXX.json        # 每名受试者的 w、a、纳入检查、重叠
  saccade_pool*.npz/json         # P6 模板池
  pilots/P2.json … P12.json      # 预实验报告
  units/<model>__<cell>__seed<k>.json
  real/  shu/  analysis/
cache/sub-XXX/{X0,alpha,mu_L,mu_R,sacc_R,sacc_L,...}.npy + meta.json
```
