import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
plt.style.use('default')  # Use default style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Set seaborn style
sns.set_style("whitegrid")

# Read CSV file
df = pd.read_csv('figs/record.csv')

# Print column names for debugging
print("Column names:", df.columns.tolist())

# Data cleaning
df = df.dropna(subset=['Algorithm', 'Dataset', 'LLM'])  # Drop rows with missing values in key columns

# Check the actual name of the 'score' column
score_column = 'Score'  # Use the actual column name

# Convert numeric columns and fill default values
df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0)

def create_operator_comparison(data, dataset_name, save_path):
    """Create a bar chart comparing average performance of operators"""
    plt.figure(figsize=(15, 8))
    
    # Filter dataset and calculate average performance for each operator
    dataset_data = data[data['Dataset'] == dataset_name]

    # Add debug information
    if dataset_data.empty:
        print(f"No data found for dataset: {dataset_name}")
        return
    
    operator_stats = dataset_data.groupby('Algorithm')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    
    # Check if operator_stats is empty
    if operator_stats.empty:
        print(f"No operator statistics available for dataset: {dataset_name}")
        return
    
    # Create bar chart
    ax = plt.gca()
    bars = ax.barh(range(len(operator_stats)), operator_stats['mean'], alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, i, f'{width:.2f}%', va='center', fontsize=10)
    
    plt.title(f'Average Performance Comparison of Different Operators on {dataset_name}')
    plt.xlabel('Score')
    plt.ylabel('Operator')
    plt.yticks(range(len(operator_stats)), operator_stats.index)
    
    # Add grid lines
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_llm_comparison(data, dataset_name, save_path):
    """Create a bar chart comparing average performance of LLMs"""
    plt.figure(figsize=(15, 8))
    
    # Filter dataset and calculate average performance for each LLM
    dataset_data = data[data['Dataset'] == dataset_name]
    llm_stats = dataset_data.groupby('LLM')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    
    # Create bar chart
    ax = plt.gca()
    bars = ax.barh(range(len(llm_stats)), llm_stats['mean'], alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, i, f'{width:.2f}%', va='center', fontsize=10)
    
    plt.title(f'Average Performance Comparison of Different LLMs on {dataset_name}')
    plt.xlabel('Score')
    plt.ylabel('LLM Model')
    plt.yticks(range(len(llm_stats)), llm_stats.index)
    
    # Add grid lines
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def print_dataset_statistics(data, dataset_name):
    """Print dataset statistics"""
    dataset_data = data[data['Dataset'] == dataset_name]
    
    print(f"\n{dataset_name} Dataset Statistics:")
    print("=" * 50)
    
    # Operator statistics
    print("\nOperator Performance:")
    operator_stats = dataset_data.groupby('Algorithm')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for operator, stats in operator_stats.iterrows():
        print(f"\n{operator}:")
        print(f"  Mean: {stats['mean']:.2f}%")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Samples: {int(stats['count'])}")
    
    # LLM statistics
    print("\nLLM Performance:")
    llm_stats = dataset_data.groupby('LLM')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for llm, stats in llm_stats.iterrows():
        print(f"\n{llm}:")
        print(f"  Mean: {stats['mean']:.2f}%")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Samples: {int(stats['count'])}")

def create_operator_llm_comparison(data, dataset_name, save_path):
    """Create a bar chart comparing LLM performance grouped by operator"""
    plt.figure(figsize=(20, 10))
    
    # Filter dataset
    dataset_data = data[data['Dataset'] == dataset_name]
    
    # Get all operators and LLMs
    operators = dataset_data['Algorithm'].unique()
    llms = dataset_data['LLM'].unique()
    
    # Set parameters for grouped bar chart
    x = np.arange(len(operators))
    width = 0.8 / len(llms)  # Adjust bar width
    
    # Create color mapping
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(llms)))
    
    # Draw grouped bar chart
    for i, llm in enumerate(llms):
        # Get scores for each operator under this LLM
        scores = []
        for op in operators:
            score = dataset_data[(dataset_data['Algorithm'] == op) & 
                               (dataset_data['LLM'] == llm)][score_column].values
            if len(score) > 0:
                scores.append(score[0])  # Take actual score
            else:
                scores.append(0)  # If no data, set to 0
        
        # Draw bar chart
        offset = width * i - width * len(llms)/2 + width/2
        bars = plt.bar(x + offset, scores, width, label=llm, alpha=0.8, color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show non-zero values
                plt.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set chart properties
    plt.title(f'LLM Performance Comparison by Operator on {dataset_name}')
    plt.xlabel('Operator')
    plt.ylabel('Score')
    
    # Set x-axis labels
    plt.xticks(x, operators, rotation=45, ha='right')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_llm_operator_comparison(data, dataset_name, save_path):
    """Create a bar chart comparing operator performance grouped by LLM"""
    plt.figure(figsize=(20, 10))
    
    # Filter dataset
    dataset_data = data[data['Dataset'] == dataset_name]
    
    # Get all LLMs and operators
    llms = dataset_data['LLM'].unique()
    operators = dataset_data['Algorithm'].unique()
    
    # Set parameters for grouped bar chart
    x = np.arange(len(llms))
    width = 0.8 / len(operators)  # Adjust bar width
    
    # Create color mapping
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(operators)))
    
    # Draw grouped bar chart
    for i, op in enumerate(operators):
        # Get scores for each LLM under this operator
        scores = []
        for llm in llms:
            score = dataset_data[(dataset_data['LLM'] == llm) & 
                               (dataset_data['Algorithm'] == op)][score_column].values
            if len(score) > 0:
                scores.append(score[0])  # Take actual score
            else:
                scores.append(0)  # If no data, set to 0
        
        # Draw bar chart
        offset = width * i - width * len(operators)/2 + width/2
        bars = plt.bar(x + offset, scores, width, label=op, alpha=0.8, color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show non-zero values
                plt.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set chart properties
    plt.title(f'Operator Performance Comparison by LLM on {dataset_name}')
    plt.xlabel('LLM Model')
    plt.ylabel('Score')
    
    # Set x-axis labels
    plt.xticks(x, llms, rotation=45, ha='right')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_overall_llm_comparison(data, save_path):
    """Create a bar chart comparing average performance of LLMs across all datasets"""
    plt.figure(figsize=(15, 8))
    
    # Calculate average performance for each LLM
    llm_stats = data.groupby('LLM')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    
    # Check if llm_stats is empty
    if llm_stats.empty:
        print("No LLM statistics available for overall comparison.")
        return
    
    # Create bar chart
    ax = plt.gca()
    bars = ax.barh(range(len(llm_stats)), llm_stats['mean'], alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, i, f'{width:.2f}%', va='center', fontsize=10)
    
    plt.title('Average Performance Comparison of Different LLMs Across All Datasets')
    plt.xlabel('Score')
    plt.ylabel('LLM Model')
    plt.yticks(range(len(llm_stats)), llm_stats.index)
    
    # Add grid lines
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_overall_operator_comparison(data, save_path):
    """Create a bar chart comparing average performance of operators across all datasets"""
    plt.figure(figsize=(15, 8))
    
    # Calculate average performance for each operator
    operator_stats = data.groupby('Algorithm')[score_column].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)
    
    # Check if operator_stats is empty
    if operator_stats.empty:
        print("No Operator statistics available for overall comparison.")
        return
    
    # Create bar chart
    ax = plt.gca()
    bars = ax.barh(range(len(operator_stats)), operator_stats['mean'], alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, i, f'{width:.2f}%', va='center', fontsize=10)
    
    plt.title('Average Performance Comparison of Different Operators Across All Datasets')
    plt.xlabel('Score')
    plt.ylabel('Operator')
    plt.yticks(range(len(operator_stats)), operator_stats.index)
    
    # Add grid lines
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# Create output directory
if not os.path.exists('figs'):
    os.makedirs('figs')

# Generate charts for each dataset
datasets = ['gsm8k', 'AQuA']
for dataset in datasets:
    # Generate operator comparison chart for this dataset
    create_operator_comparison(
        df, 
        dataset,
        f'figs/{dataset}_operator_comparison.png'
    )
    
    # Generate LLM comparison chart for this dataset
    create_llm_comparison(
        df,
        dataset,
        f'figs/{dataset}_llm_comparison.png'
    )
    
    # Generate operator-LLM comparison chart for this dataset
    create_operator_llm_comparison(
        df,
        dataset,
        f'figs/{dataset}_operator_llm_comparison.png'
    )
    
    # Generate LLM-operator comparison chart for this dataset
    create_llm_operator_comparison(
        df,
        dataset,
        f'figs/{dataset}_llm_operator_comparison.png'
    )
    
    # Print statistics for this dataset
    print_dataset_statistics(df, dataset)

# Generate overall comparison charts
create_operator_comparison(df, None, 'figs/overall_operator_comparison.png')
create_overall_llm_comparison(df, 'figs/overall_llm_comparison.png')
create_overall_operator_comparison(df, 'figs/overall_operator_comparison.png')

print("\nAnalysis complete! Results saved to figs directory.") 