# 论文图 ↔ 画图代码对照

论文：`LLMAT-paper/NeurIPS/neurips_2026.tex`（Overleaf `8f2324c` + 本地修改）。代码：本仓库 `LLMAT/`。图号和页码来自本地编译的 PDF（48 页）。

- 代码路径相对于仓库根目录。脚本用相对路径保存，需在仓库根目录运行（`python -m plotting.plot_gap`）。
- 大多数 `plotting/` 脚本的 `__main__` 只启用了一个函数，`f_models` / `finetuning` 等参数是手改的：画对应的图前先改 `__main__`。
- 脚本输出到 `results/figs/...`，再手动复制进 `LLMAT-paper/figures/...`。


## Introduction

### Fig. 1 (p.2) `fig:bo`

> Overview of LLMAT: Binary classifiers on the LLM features recursively partition the candidate space to a tree …

tex 第 366 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `BO.png.pdf` | — | — | 方法示意图，手工绘制，不是代码生成 |


## Experiments

### Fig. 2 (p.7) `fig:LLMAT_refine_AFs`

> Leaf-node AFs offer more fine-grained candidate suggestions for Levy-1D optimization.

tex 第 708 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `toy_data/root_leaf_acqf_iter_5.png` | [plotting/visualize_toy_bo.py](plotting/visualize_toy_bo.py) | `visualize_root_and_leaf_acqf` | 运行 scripts/run_visualization.sh |

### Fig. 3 (p.7) `fig:t5-chem-mol-finetuning`

> Left: Comparison over baselines on fixed features across all models and datasets.

tex 第 742 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/all_models_datasets_GAP_algorithms_average.pdf` | [plotting/plot_gap_algorithms_avg.py](plotting/plot_gap_algorithms_avg.py) | `plot_average_gap_over_all` | main 中启用；输出 results/figs/algorithms_avg/ |
| `GAPs/t5-base-chem_GAP_finetune_average.pdf` | [plotting/plot_gap.py](plotting/plot_gap.py) | `plot_average_gap` | 输出 results/figs/0907/GAPs/ |
| `Regrets/t5-base-chem_regret_finetune_average.pdf` | [plotting/plot_regret.py](plotting/plot_regret.py) | `plot_average_regret` | 输出 results/figs/0907/Regrets/ |

### Fig. 4 (p.8) `fig:fix-best-y`

> BO with fixed features (best-performing feature per algorithm shown). LLMAT achieves best overall performance,…

tex 第 766 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Bestys/all_datasets_comparison_best_auc.pdf` | [plotting/plot_fixed.py](plotting/plot_fixed.py) | `plot_best_y_all_seaborn_auc3` | auc / auc2 / auc3 三个版本写同一个文件名；main 里留的是 auc3（已注释） |

### Fig. 5 (p.8) `fig:molformer_tree_depth`

> (无 caption)

tex 第 784 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `finetuning_radar_analysis_academic.pdf` | [plotting/plot_diff.py](plotting/plot_diff.py) | `plot_academic_radar_chart` | 由 main 里的 create_radar_plot 调用 |
| `GAPs/molformer_GAP_finetune_average_ablation.pdf` | [plotting/plot_tree_depth_ablation.py](plotting/plot_tree_depth_ablation.py) | `plot_gap_tree_depth_ablation` |  |
| `Regrets/molformer_regret_finetune_average_ablation.pdf` | [plotting/plot_tree_depth_ablation.py](plotting/plot_tree_depth_ablation.py) | `plot_regret_tree_depth_ablation` |  |

### Fig. 6 (p.9) `fig:molformer_clustering`

> Average reduced prediction time percentage, average GAP, and average regret across six datasets for LLM-base c…

tex 第 818 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/molformer_pval_ablation.pdf` | [plotting/plot_cluster_time_ablation.py](plotting/plot_cluster_time_ablation.py) | `plot_average_prediction_time_arcoss_datasets` |  |

### Fig. 7 (p.9) `fig:molformer_algo_ablation_remove`

> Left: Average GAP across six datasets illustrating the impact of individual algorithmic module ablations. Righ…

tex 第 846 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/molformer_GAP_ablation_average.pdf` | [plotting/plot_gap_ablation.py](plotting/plot_gap_ablation.py) | `plot_average_gap_ablation` | plot_gap_ablation_avg.py::plot_average_gap_ablation_per_model 也写同名文件 |
| `ablation/gap_visualization_normalized.png` | [plotting/visualize_clustering_prompt_gaps.py](plotting/visualize_clustering_prompt_gaps.py) | `visualize_clustering_prompt_gaps` | 输出到当前目录 |


## Analysis of LLM Property Ranking

### Fig. 8 (p.16) `fig:llm_cluster`

> General LLMs provide coarse property ranking of redox potentials rather than accurate numeric value for the th…

tex 第 881 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `molecules/mole1.png`<br>`molecules/mole2.png`<br>`molecules/mole3.png` | — | — | 仓库里没找到生成代码（可能是 RDKit 手动导出） |


## Clustering Prompts

### Fig. 9 (p.23) `fig:prompt_ablation`

> Prompt variation's effect on cluster labels.

tex 第 1256 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/agreement_heatmap.pdf`<br>`ablation/cluster_distributions.pdf`<br>`ablation/objectives_by_cluster.pdf` | [benchmarks/MAT/prompt_cluster_ablation.py](benchmarks/MAT/prompt_cluster_ablation.py) | `plot_ablation_results` |  |


## Additional Experimental Results

### Fig. 10 (p.27) `fig:fix-best-y-all`

> BO process for different datasets on features extracted from various foundation models.

tex 第 1530 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Bestys/all_datasets_comparison_all.pdf` | [plotting/plot_fixed.py](plotting/plot_fixed.py) | `plot_best_y_all_seaborn` |  |

### Fig. 11 (p.27) `fig:t5-chem-fixed-train-time`

> Training time comparison on different chemistry datasets with T5-Chem model.

tex 第 1539 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FXredox-mer_t5-base-chem_trace_training_time.pdf` 等 6 个（FXPhotovoltaics, FXkinase, FXlaser, FXphotoswitch, FXredox-mer, FXsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_training_time` | FX=fixed, FT=finetuning（main 中 finetuning 开关） |

### Fig. 12 (p.28) `fig:t5-chem-fixed-pred-time`

> Prediction time comparison on different chemistry datasets with T5-Chem model.

tex 第 1554 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FXredox-mer_t5-base-chem_trace_pred_time.pdf` 等 6 个（FXPhotovoltaics, FXkinase, FXlaser, FXphotoswitch, FXredox-mer, FXsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_prediction_time` | 同上 |

### Fig. 13 (p.28) `fig:t5-chem-fixed-overall-time`

> Overall time comparison on different chemistry datasets with T5-Chem model.

tex 第 1566 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FXredox-mer_t5-base-chem_trace_time.pdf` 等 6 个（FXPhotovoltaics, FXkinase, FXlaser, FXphotoswitch, FXredox-mer, FXsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_time` | 同上 |

### Fig. 14 (p.29) `fig:t5-chem-finetune-his`

> Historical optimums comparison on different chemistry datasets using T5-Chem model.

tex 第 1623 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Bestys/redox-mer_t5-base-chem_finetune.pdf` 等 6 个（kinase, laser, pce, photoswitch, redox-mer, solvation） | [plotting/plot_best_y.py](plotting/plot_best_y.py) | `plot_best_y` | 输出 results/figs/0907/Bestys/ |

### Fig. 15 (p.29) `fig:t5-chem-finetune-gap`

> GAP comparison on different chemistry datasets with T5-Chem model.

tex 第 1637 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `GAPs/redox-mer_t5-base-chem_GAP_finetune.pdf` 等 6 个（Photovoltaics, kinase, laser, photoswitch, redox-mer, solvation） | [plotting/plot_gap.py](plotting/plot_gap.py) | `plot_gap` | 每个数据集一张；pce 在代码里被改名为 Photovoltaics |

### Fig. 16 (p.30) `fig:t5-chem-finetune-regret`

> Regrets comparison on different chemistry datasets using T5-Chem model.

tex 第 1651 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Regrets/redox-mer_t5-base-chem_regret_finetune.pdf` 等 6 个（Photovoltaics, kinase, laser, photoswitch, redox-mer, solvation） | [plotting/plot_regret.py](plotting/plot_regret.py) | `plot_regret` | 每个数据集一张；pce→Photovoltaics |

### Fig. 17 (p.30) `fig:t5-chem-finetune-train-time`

> Training time comparison on different chemistry datasets with T5-Chem model.

tex 第 1689 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FTredox-mer_t5-base-chem_trace_training_time.pdf` 等 6 个（FTPhotovoltaics, FTkinase, FTlaser, FTphotoswitch, FTredox-mer, FTsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_training_time` | FX=fixed, FT=finetuning（main 中 finetuning 开关） |

### Fig. 18 (p.31) `fig:t5-chem-finetune-pred-time`

> Prediction time comparison on different chemistry datasets with T5-Chem model.

tex 第 1701 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FTredox-mer_t5-base-chem_trace_pred_time.pdf` 等 6 个（FTPhotovoltaics, FTkinase, FTlaser, FTphotoswitch, FTredox-mer, FTsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_prediction_time` | 同上 |

### Fig. 19 (p.31) `fig:t5-chem-finetune-overall-time`

> Overall time comparison on different chemistry datasets with T5-Chem model.

tex 第 1713 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FTredox-mer_t5-base-chem_trace_time.pdf` 等 6 个（FTPhotoswitch, FTPhotovoltaics, FTkinase, FTlaser, FTredox-mer, FTsolvation） | [plotting/plot_time.py](plotting/plot_time.py) | `plot_time` | 同上 |

### Fig. 20 (p.31) `fig:t5-chem-time-fix`

> Computation time of different algorithms with fixed T5-Chem features across datasets.

tex 第 1726 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FXtime_comparison_grouped.pdf` | [plotting/plot_time.py](plotting/plot_time.py) | `plot_time_comparison_grouped` | FX/FT 前缀来自 main 里的 finetuning 开关 |

### Fig. 21 (p.32) `fig:t5-chem-time-finetune`

> BO time for different datasets on finetuning T5-Chem model with LoRA.

tex 第 1732 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/FTtime_comparison_grouped.pdf` | [plotting/plot_time.py](plotting/plot_time.py) | `plot_time_comparison_grouped` | FX/FT 前缀来自 main 里的 finetuning 开关 |

### Fig. 22 (p.32) `fig:molformer-finetune-his`

> Historical optimums comparison on different chemistry datasets using Molformer.

tex 第 1745 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Bestys/redox-mer_molformer_finetune.pdf` 等 6 个（kinase, laser, pce, photoswitch, redox-mer, solvation） | [plotting/plot_best_y.py](plotting/plot_best_y.py) | `plot_best_y` | 输出 results/figs/0907/Bestys/ |

### Fig. 23 (p.32) `fig:molformer-regret`

> Regrets comparison on different chemistry datasets using Molformer.

tex 第 1797 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `Regrets/redox-mer_molformer_regret_finetune.pdf` 等 6 个（Photovoltaics, kinase, laser, photoswitch, redox-mer, solvation） | [plotting/plot_regret.py](plotting/plot_regret.py) | `plot_regret` | 每个数据集一张；pce→Photovoltaics |

### Fig. 24 (p.33) `fig:molformer-gap`

> GAPs comparison on different chemistry datasets using Molformer

tex 第 1821 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `GAPs/redox-mer_molformer_GAP_finetune.pdf` 等 6 个（Photovoltaics, kinase, laser, photoswitch, redox-mer, solvation） | [plotting/plot_gap.py](plotting/plot_gap.py) | `plot_gap` | 每个数据集一张；pce 在代码里被改名为 Photovoltaics |

### Fig. 25 (p.33) `fig:molformer`

> Average GAPs and regrets across all the datasets on finetuning and fixed Molformer.

tex 第 1844 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `GAPs/molformer_GAP_finetune_average.pdf` | [plotting/plot_gap.py](plotting/plot_gap.py) | `plot_average_gap` | 输出 results/figs/0907/GAPs/ |
| `Regrets/molformer_regret_finetune_average.pdf` | [plotting/plot_regret.py](plotting/plot_regret.py) | `plot_average_regret` | 输出 results/figs/0907/Regrets/ |

### Fig. 26 (p.33) `fig:t5-chem_clustering_kmeans`

> Distribution of property values for K-means clusters on features extracted from T5-Chem.

tex 第 1855 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `clusters/redox-mer_t5-base-chem-just-smiles-average_kmeans_clusters.pdf` 等 6 个（kinase, laser, pce, photoswitch, redox-mer, solvation） | [benchmarks/MAT/material_benchmarks.py](benchmarks/MAT/material_benchmarks.py) | `MATBench.draw_clusters` | 构造函数里 self.draw_clusters() 被注释（第 146 行），要画需打开；输出 benchmarks/MAT/data/cluster_figs/ |

### Fig. 27 (p.34) `fig:molformer_clustering_kmeans`

> Distribution of property values for K-means clusters on features extracted from Molformer.

tex 第 1866 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `clusters/redox-mer_molformer_kmeans_clusters.pdf` 等 6 个（kinase, laser, pce, photoswitch, redox-mer, solvation） | [benchmarks/MAT/material_benchmarks.py](benchmarks/MAT/material_benchmarks.py) | `MATBench.draw_clusters` | 构造函数里 self.draw_clusters() 被注释（第 146 行），要画需打开；输出 benchmarks/MAT/data/cluster_figs/ |

### Fig. 28 (p.34) `fig:t5-chem_clustering_llms`

> Distribution of property values for ChatGPT-based clustering results.

tex 第 1879 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `clusters/redox-mer_t5-base-chem-just-smiles-average_llms_clusters.pdf` 等 6 个（kinase, laser, pce, photoswitch, redox-mer, solvation） | [benchmarks/MAT/material_benchmarks.py](benchmarks/MAT/material_benchmarks.py) | `MATBench.draw_clusters` | 构造函数里 self.draw_clusters() 被注释（第 146 行），要画需打开；输出 benchmarks/MAT/data/cluster_figs/ |

### Fig. 29 (p.35) `fig:molformer-gap-paval-ablation`

> GAPs' change for different clustering approach and p-values using Molformer.

tex 第 1906 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `pval_ablation/molformer_redox-mer_GAP_finetune_pval_ablation.pdf` 等 6 个（molformer） | [plotting/plot_cluster_time_ablation.py](plotting/plot_cluster_time_ablation.py) | `plot_gap_pval_ablation` |  |

### Fig. 30 (p.35) `fig:molformer-regret-paval-ablation`

> Regrets' change for different clustering approach and p-values with Molformer.

tex 第 1919 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `pval_ablation/molformer_redox-mer_regret_finetune_pval_ablation.pdf` 等 6 个（molformer） | [plotting/plot_cluster_time_ablation.py](plotting/plot_cluster_time_ablation.py) | `plot_regret_pval_ablation` |  |

### Fig. 31 (p.36) `fig:molformer_clustering_all`

> Prediction time reduction percentage, AUC of GAPs, and Regrets across all the datasets for different clusterin…

tex 第 1934 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `time/redox-mer_molformer_pval_ablation.pdf` 等 6 个（kinase, laser, photoswitch, photovoltaics, redox-mer, solvation） | [plotting/plot_cluster_time_ablation.py](plotting/plot_cluster_time_ablation.py) | `plot_average_prediction_time` | 每个数据集一张 |

### Fig. 32 (p.37) `fig:Levey_1D`

> Illustration 1000 samples of Levy-1D function with K-means clustering.

tex 第 1949 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `toy_data/levy_1d_clusters_landscape.png` | — | — | 仓库里没找到生成代码 |

### Fig. 33–34 (p.37) `fig:LLMAT_refine_AFs_iteration1`, `fig:LLMAT_refine_AFs_iteration3`

> Refined AFs in LLMAT enable more efficient BO (Iteration 1). / Refined AFs in LLMAT enable more efficient BO (Iteration 3).

tex 第 1960 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `toy_data/node_details_iter_1.png`<br>`toy_data/node_details_iter_3.png` | [plotting/visualize_toy_bo.py](plotting/visualize_toy_bo.py) | `visualize_node_details` | 运行 scripts/run_visualization.sh（Levy-1D，seed 5） |

### Fig. 35–36 (p.38) `fig:LLMAT_refine_AFs_iteration5`, `fig:LLMAT_refine_AFs_iteration7`

> Refined AFs in LLMAT enable more efficient BO (Iteration 5). / Refined AFs in LLMAT enable more efficient BO (Iteration 7).

tex 第 1975 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `toy_data/node_details_iter_5.png`<br>`toy_data/node_details_iter_7.png` | [plotting/visualize_toy_bo.py](plotting/visualize_toy_bo.py) | `visualize_node_details` | 运行 scripts/run_visualization.sh（Levy-1D，seed 5） |

### Fig. 37 (p.38) `fig:Label Noise Ablation`

> Illustration of label noise's effects on the average regret of LLMAT.

tex 第 2010 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `noise/Label_Noise_LLMAT.png` | — | — | 没找到同名输出；最接近的是 plotting/plot_noise_ablation_levy.py（输出 levy_10d_noise_ablation_*） |

### Fig. 38 (p.38) `fig: Ablation of gamma.`

> Illustration of LLMAT Performance Change w.r.t Different gammas.

tex 第 2020 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/redox_mer_gamma_ablation_combined.pdf` | [plotting/plot_gamma_ablation_redox.py](plotting/plot_gamma_ablation_redox.py) | `plot_combined_comparison` | 输出 results/figs/gamma_ablation/ |

### Fig. 39 (p.39) `fig: llma2.`

> Illustration Performance Comparison of Various Algorithms on Llama2-7b across All Datasets.

tex 第 2028 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/llama-2-7b_GAP_algorithms_average.pdf` | [plotting/plot_gap_algorithms_avg.py](plotting/plot_gap_algorithms_avg.py) | `plot_average_gap_per_model` | main 里改调用并设 f_models |

### Fig. 40 (p.39) `fig:Multi-Redox Pareto Front`

> Illustration of Pareto fronts and  LLMs and K-means clusters on Multi-Redox Data.

tex 第 2038 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `multiobj/redox-mer_multiobj_t5-base-chem-average_llms_pareto.pdf`<br>`multiobj/redox-mer_multiobj_t5-base-chem-average_kmeans_pareto.pdf` | [benchmarks/MAT/multiobj_material_benchmarks.py](benchmarks/MAT/multiobj_material_benchmarks.py) | `draw_pareto_front` | 输出 benchmarks/MAT/data/cluster_figs/ |

### Fig. 41 (p.39) `fig:multi-objective extension`

> Multi-objective optimization for LAPEFT vs. LLMAT on the Redox-mer dataset: Upper & Lower Left: Hypervolume, s…

tex 第 2046 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `multiobj/redox_hv_scalarized_subfigures_comparison.png` | [plotting/plot_redox_comparison.py](plotting/plot_redox_comparison.py) | `plot_hv_and_scalarized_subfigures` |  |
| `multiobj/redox_objectives_comparison.png` | [plotting/plot_redox_comparison.py](plotting/plot_redox_comparison.py) | `plot_objectives_comparison` |  |

### Fig. 42 (p.40) `fig:api_ablation`

> GAP (top) and average regret (bottom) of LLMAT on Redox-mer when LLM-based clustering uses different LLM APIs …

tex 第 2057 行

| 图文件 (`figures/`) | 代码 | 函数 | 备注 |
|---|---|---|---|
| `ablation/gap_visualization_api_ablation.png` | [plotting/visualize_api_ablation_gaps.py](plotting/visualize_api_ablation_gaps.py) | `visualize_api_ablation_gaps` |  |
| `ablation/regret_visualization_api_ablation.png` | [plotting/visualize_api_ablation_gaps.py](plotting/visualize_api_ablation_gaps.py) | `visualize_api_ablation_regret` |  |


## 按脚本汇总

| 脚本 | 用到的图 |
|---|---|
| `benchmarks/MAT/material_benchmarks.py` | Fig. 26, Fig. 27, Fig. 28 |
| `benchmarks/MAT/multiobj_material_benchmarks.py` | Fig. 40 |
| `benchmarks/MAT/prompt_cluster_ablation.py` | Fig. 9 |
| `plot_redox_comparison.py` | Fig. 41 |
| `plotting/plot_best_y.py` | Fig. 14, Fig. 22 |
| `plotting/plot_cluster_time_ablation.py` | Fig. 6, Fig. 29, Fig. 30, Fig. 31 |
| `plotting/plot_diff.py` | Fig. 5 |
| `plotting/plot_fixed.py` | Fig. 4, Fig. 10 |
| `plotting/plot_gamma_ablation_redox.py` | Fig. 38 |
| `plotting/plot_gap.py` | Fig. 3, Fig. 15, Fig. 24, Fig. 25 |
| `plotting/plot_gap_ablation.py` | Fig. 7 |
| `plotting/plot_gap_algorithms_avg.py` | Fig. 3, Fig. 39 |
| `plotting/plot_regret.py` | Fig. 3, Fig. 16, Fig. 23, Fig. 25 |
| `plotting/plot_time.py` | Fig. 11, Fig. 12, Fig. 13, Fig. 17, Fig. 18, Fig. 19, Fig. 20, Fig. 21 |
| `plotting/plot_tree_depth_ablation.py` | Fig. 5 |
| `visualize_api_ablation_gaps.py` | Fig. 42 |
| `visualize_clustering_prompt_gaps.py` | Fig. 7 |
| `visualize_toy_bo.py` | Fig. 2, Fig. 33, Fig. 35 |
| 无代码（—） | Fig. 1, Fig. 8, Fig. 32, Fig. 37 |
