import matplotlib.pyplot as plt
import pandas as pd

# 从 CSV 文件中读取数据
df = pd.read_csv('figs/overall_results.csv')

# 过滤掉成本为0的数据
df = df[(df['AQuA-Cost($)'] > 0) & (df['gsm8k-Cost($)'] > 0)]

# 计算每个 Algorithm 和 LLM 的平均分数和平均成本
df['Avg_Score'] = (df['AQuA-Score'] + df['gsm8k-Score']) / 2
df['Avg_Cost'] = (df['AQuA-Cost($)'] + df['gsm8k-Cost($)']) / 2

# 按 Algorithm 和 LLM 分组并计算平均值
avg_data = df.groupby(['Algorithm', 'LLM']).agg({'Avg_Score': 'mean', 'Avg_Cost': 'mean'}).reset_index()

# 创建图形
plt.figure(figsize=(15, 10))

# 绘制平均值
colors = ['blue', 'red', 'green', 'purple', 'orange']
for idx, row in avg_data.iterrows():
    plt.scatter(row['Avg_Cost'], row['Avg_Score'], color=colors[idx % len(colors)], s=100, label=f"{row['Algorithm']} - {row['LLM']}")
    plt.annotate(f"{row['Algorithm']}\n{row['LLM']}", 
                 (row['Avg_Cost'], row['Avg_Score']),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

# 去掉图例
# plt.legend(title='Algorithm - LLM', fontsize=12, title_fontsize=12)

# 添加标题和标签
plt.title('Average Score vs Average Cost by Algorithm and LLM', fontsize=14)
plt.xlabel('Average Cost ($)', fontsize=14)
plt.ylabel('Average Score', fontsize=14)

# 设置对数刻度
plt.xscale('log')
plt.xlim(-0.01, avg_data['Avg_Cost'].max() * 1.2)

# 添加垂直线以指示1成本标记
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

# 添加辅助线
min_cost = avg_data['Avg_Cost'].min()
max_cost = avg_data['Avg_Cost'].max()
min_score = avg_data['Avg_Score'].min()
max_score = avg_data['Avg_Score'].max()

plt.plot([min_cost, max_cost], [max_score, min_score], color='gray', linestyle='--', linewidth=1, label='Ideal Line')

# 设置刻度字体大小和格式
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# 保存图形
plt.savefig('./figs/average_score_vs_cost_by_algorithm_llm.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()