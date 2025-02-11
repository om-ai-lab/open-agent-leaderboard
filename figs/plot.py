import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
df = pd.read_csv('figs/overall_results.csv')

# Filter out data with cost 0
df = df[(df['AQuA-Cost($)'] > 0) & (df['gsm8k-Cost($)'] > 0) & (df['MATH-500-Cost($)'] > 0)]

# Calculate average score and average cost for each Algorithm and LLM
df['Avg_Score'] = (df['AQuA-Score'] + df['gsm8k-Score'] + df['MATH-500-Score']) / 3
df['Avg_Cost'] = (df['AQuA-Cost($)'] + df['gsm8k-Cost($)'] + df['MATH-500-Cost($)']) / 3

# Group by Algorithm and LLM and calculate the mean
avg_data = df.groupby(['Algorithm', 'LLM']).agg({'Avg_Score': 'mean', 'Avg_Cost': 'mean'}).reset_index()

# Create plot
plt.figure(figsize=(15, 10))

# Plot averages
colors = ['blue', 'red', 'green', 'purple', 'orange']
for idx, row in avg_data.iterrows():
    plt.scatter(row['Avg_Cost'], row['Avg_Score'], color=colors[idx % len(colors)], s=100, label=f"{row['Algorithm']} - {row['LLM']}")
    plt.annotate(f"{row['Algorithm']}\n{row['LLM']}", 
                 (row['Avg_Cost'], row['Avg_Score']),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

# Remove legend
# plt.legend(title='Algorithm - LLM', fontsize=12, title_fontsize=12)

# Add title and labels
plt.title('Average Score vs Average Cost by Algorithm and LLM', fontsize=14)
plt.xlabel('Average Cost ($)', fontsize=14)
plt.ylabel('Average Score', fontsize=14)

# Set log scale
plt.xscale('log')
plt.xlim(-0.01, avg_data['Avg_Cost'].max() * 1.2)

# Add vertical line to indicate cost of 1
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

# Add auxiliary line
min_cost = avg_data['Avg_Cost'].min()
max_cost = avg_data['Avg_Cost'].max()
min_score = avg_data['Avg_Score'].min()
max_score = avg_data['Avg_Score'].max()

plt.plot([min_cost, max_cost], [max_score, min_score], color='gray', linestyle='--', linewidth=1, label='Ideal Line')

# Set tick font size and format
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Save plot
plt.savefig('./figs/average_score_vs_cost_by_algorithm_llm.png', dpi=300, bbox_inches='tight')

# Show plot
plt.show()