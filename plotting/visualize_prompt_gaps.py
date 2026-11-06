import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_average_gap():
    """
    Visualizes the average GAP metrics for different clustering prompts.
    """
    base_path = "results/redox-mer/t5-base-just-smiles-average/llmat/"
    prompt_dirs = sorted(glob.glob(os.path.join(base_path, "llms_prompt*")))
    
    plt.figure(figsize=(12, 8))

    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        file_pattern = os.path.join(prompt_dir, "trace_best_y_*.npy")
        best_y_files = glob.glob(file_pattern)
        
        if not best_y_files:
            print(f"No 'trace_best_y' files found in {prompt_dir}")
            continue

        all_gaps = []
        for best_y_file in best_y_files:
            try:
                trace = np.load(best_y_file)
                ground_truth = trace[0]
                best_y_values = trace[1:]
                gap = ground_truth - best_y_values
                all_gaps.append(gap)
            except Exception as e:
                print(f"Could not process file {best_y_file}: {e}")
        
        if not all_gaps:
            print(f"No valid GAP data could be calculated for {prompt_name}")
            continue

        # Pad shorter runs to the length of the longest run
        max_len = max(len(gap) for gap in all_gaps)
        padded_gaps = []
        for gap in all_gaps:
            pad_len = max_len - len(gap)
            padded_gaps.append(np.pad(gap, (0, pad_len), 'edge'))

        gaps_array = np.array(padded_gaps)
        mean_gap = np.mean(gaps_array, axis=0)
        std_gap = np.std(gaps_array, axis=0)
        
        iterations = np.arange(len(mean_gap))
        
        plt.plot(iterations, mean_gap, label=prompt_name)
        plt.fill_between(iterations, mean_gap - std_gap, mean_gap + std_gap, alpha=0.2)

    plt.title("Average 1 Metrics Across Different Clustering Prompts")
    plt.xlabel("Iterations")
    plt.ylabel("Average GAP")
    plt.legend()
    plt.grid(True)
    
    output_filename = "gap_visualization.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_average_gap()
