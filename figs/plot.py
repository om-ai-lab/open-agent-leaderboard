import matplotlib.pyplot as plt

# Data - AQuA
aqua_algorithms = ['IO', 'COT', 'SC-COT', 'POT', 'ReAct-Pro*', 
                  'IO', 'COT', 'SC-COT', 'POT', 'ReAct-Pro*']
aqua_llms = ['gpt-3.5-turbo'] * 5 + ['Doubao-lite-32k'] * 5
aqua_costs = [0.0380, 0.0957, 0.6491, 0.1748, 0.4928, 
              0.0058, 0.0066, 0.0409, 0.0147, 0.0446]
aqua_scores = [38.98, 61.02, 67.32, 59.45, 64.57, 
               79.13, 82.68, 83.46, 71.65, 77.56]

# Data - GSM8K
gsm8k_algorithms = aqua_algorithms  # Same algorithm order
gsm8k_llms = aqua_llms  # Same LLM order
gsm8k_costs = [0.3328, 0.6788, 5.0227, 0.6902, 3.4633, 
               0.0354, 0.0557, 0.1533, 0.0576, 0.2513]
gsm8k_scores = [37.83, 78.70, 80.06, 76.88, 74.91, 
                72.02, 89.31, 88.63, 79.61, 85.60]

# Create figure
plt.figure(figsize=(15, 10))  # Increase figure size

# Calculate valid maximum values
valid_costs = [c for c in aqua_costs + gsm8k_costs if c is not None and c > 0]
valid_scores = [s for s in aqua_scores + gsm8k_scores if s > 0]
max_cost = max(valid_costs)
max_score = max(valid_scores)

# Plot lines for each LLM and dataset
llms = ['gpt-3.5-turbo', 'Doubao-lite-32k']
datasets = ['AQuA', 'GSM8K']
colors = ['blue', 'red']
markers = ['o', 's']

for llm_idx, llm in enumerate(llms):
    for dataset_idx, dataset in enumerate(datasets):
        points = []
        for i, (algo, curr_llm) in enumerate(zip(aqua_algorithms, aqua_llms)):
            if curr_llm == llm:
                if dataset == 'AQuA' and aqua_scores[i] > 0 and aqua_costs[i] > 0:
                    points.append((aqua_costs[i], aqua_scores[i], algo))
                elif dataset == 'GSM8K' and gsm8k_scores[i] > 0 and gsm8k_costs[i] is not None and gsm8k_costs[i] > 0:
                    points.append((gsm8k_costs[i], gsm8k_scores[i], algo))
        
        if points:
            # Sort by cost
            points.sort(key=lambda x: x[0])
            costs, scores, algos = zip(*points)
            
            # Plot lines and points
            color = colors[dataset_idx]
            linestyle = '-' if llm == 'gpt-3.5-turbo' else '--'
            label = f'{dataset}-{llm}'
            
            plt.plot(costs, scores, linestyle, color=color, alpha=0.5, label=label, linewidth=2)
            plt.scatter(costs, scores, color=color, marker=markers[llm_idx], s=100)  # Increase point size
            
            # Add labels
            for cost, score, algo in points:
                label = f'{algo}\n({llm})\n{dataset}'
                plt.annotate(label, 
                           (cost, score),
                           xytext=(10, 10),  # Increase text offset
                           textcoords='offset points',
                           fontsize=10,  # Increase font size
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Add legend
plt.legend(title='Dataset-LLM', fontsize=12, title_fontsize=12)

# Add title and labels
plt.title('Score vs Cost', fontsize=14)
plt.xlabel('Cost ($)', fontsize=14)
plt.ylabel('Score', fontsize=14)

# Set tighter display range with log scale
plt.xscale('log')  # Use log scale to expand 0-1 range
plt.xlim(-0.01, max_cost * 1.2)  # Adjust x-axis to include all values, starting from 0.01 to avoid log(0)

# Add vertical line to indicate the 1 cost mark
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

# Set tick font size and format
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format x-axis values as 0.00x

# Save figure
plt.savefig('score_vs_cost11.png', dpi=300, bbox_inches='tight')

# Show figure
plt.show()