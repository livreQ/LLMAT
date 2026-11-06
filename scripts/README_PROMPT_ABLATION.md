# Redoxmer Prompt Ablation Study

Simple scripts for running ablation study on redoxmer with different LLM clustering prompts.

## Files

- `run_redoxmer_prompt_ablation.py` - Python script that runs single experiment with specific prompt_id
- `run_prompt_ablation.sh` - batch script for running the ablation

## Prompt Variants

The ablation study tests 5 different prompt variants (prompt_id 0-4):

- **0**: Original prompt (baseline)
- **1**: Detailed variant with more chemical reasoning
- **2**: Simplified variant with direct instructions
- **3**: Quantitative variant with structural guidelines
- **4**: Example-based variant from redox expert perspective

See `benchmarks/MAT/prompt_cluster_ablation.py` for full prompt definitions.

## Usage

### Single Experiment

Run a single experiment with specific prompt_id and seed:

```bash
bash scripts/run_prompt_ablation.sh 665 0
```

Where:
- `665` is the seed
- `0` is the prompt_id (0-4)

### Parallel Execution for All Prompts

To run all prompt variants in parallel with one seed:

```bash
for prompt_id in 0 1 2 3 4; do
    bash scripts/run_prompt_ablation.sh 665 $prompt_id
done
```

### Full Ablation Study (All Prompts × All Seeds)

```bash
for seed in 665 1319 7222 7541 8916; do
    for prompt_id in 0 1 2 3 4; do
        bash scripts/run_prompt_ablation.sh $seed $prompt_id
    done
done
```

This submits 25 jobs (5 seeds × 5 prompts).

## Prerequisites

1. **Generate Clustered Datasets First**

Before running BO experiments, generate the LLM-clustered datasets for each prompt:

```python
# In benchmarks/MAT/prompt_material_benchmarks.py __main__ block
for prompt_id in range(5):
    mat_bench = MATBench(
        data_name="redox-mer",
        run_subset_only=False,
        feature_type="molformer",
        finetuning=True,
        iupac=False,
        prompt_type="just-smiles",
        randseed=666,
        clustering_type="llms",
        feature_reduction="average",
        prompt_id=prompt_id,  # This generates dataset for each prompt
    )
```

Or run the existing script:
```bash
python benchmarks/MAT/prompt_material_benchmarks.py
```

This creates datasets at: `benchmarks/MAT/data/dataset/redox-mer/molformer/llms/dataset{prompt_id}.pkl`

2. **Set OpenAI API Key**

The LLM clustering uses GPT-4o. Update API key in `benchmarks/MAT/prompt_cluster.py` line 105.

## How It Works

1. **Dataset Loading**: The `MATBench` class loads the clustered dataset based on `prompt_id`:
   ```python
   dataset_path = DATA_DIR_NAME + "/data/dataset/" + self.dataset_name + "dataset" + str(self.prompt_id) + ".pkl"
   ```

2. **BO Execution**: The script runs standard LLMAT with the loaded dataset:
   - Uses LLM clustering (`clustering_type=llms`)
   - Finetuning enabled (`--finetuning`)
   - Results saved to standard output directories

3. **Results**: Each experiment saves results to:
   ```
   results/redox-mer/molformer/llms/
   ├── trace_best_y_*_llmat_*_{seed}.npy
   ├── trace_y_his_*_llmat_*_{seed}.npy
   └── ...
   ```

## Analyzing Results

After experiments complete, use existing analysis notebooks:

```bash
# Load and compare results from different prompts
jupyter notebook notebook/plot_results_finetune_average-molformer-ablation.ipynb
```

Or create custom analysis:

```python
import numpy as np
import glob

# Load all results
for prompt_id in range(5):
    for seed in [665, 1319, 7222, 7541, 8916]:
        trace_file = f"results/redox-mer/molformer/llms/trace_best_y_*_llmat_*_{seed}.npy"
        traces = glob.glob(trace_file)
        if traces:
            trace = np.load(traces[0])
            print(f"Prompt {prompt_id}, Seed {seed}: Final best = {trace[-1]:.4f}")
```

## Example Workflow

```bash
# 1. Generate datasets (one-time, requires OpenAI API)
python benchmarks/MAT/prompt_material_benchmarks.py

# 2. Submit jobs for all prompts with one seed (for testing)
for prompt_id in 0 1 2 3 4; do
    bash scripts/run_prompt_ablation.sh 665 $prompt_id
done

# 3. Monitor jobs
squeue -u $USER

# 4. Check results
ls results/redox-mer/molformer/llms/

# 5. Full ablation (all seeds × all prompts)
for seed in 665 1319 7222 7541 8916; do
    for prompt_id in 0 1 2 3 4; do
        bash scripts/run_prompt_ablation.sh $seed $prompt_id
    done
done
```

## Notes

- Each job takes ~2 hours with finetuning
- Ensure datasets are pre-generated to avoid API costs during experiments
- The `prompt_id` parameter is passed through to `MATBench` which loads the correct dataset
- Results are comparable since same seeds are used across prompts
