#!/usr/bin/env python3
"""
Clean, readable visualization of subdomain difficulty vs consistency.
Uses a heatmap-style approach with clear categorization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_llm_results():
    """Load all LLM test results and compute subdomain statistics."""
    # Look for CSV files in current directory
    csv_files = list(Path(".").glob("llm_test_results_*.csv"))
    if not csv_files:
        print("No LLM test result files found in current directory.")
        return pd.DataFrame()
    
    all_results = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Load subdomain mapping
    subdomain_file = Path("merged_all_questions_with_subdomains_renamed.csv")
    if not subdomain_file.exists():
        print("Subdomain mapping file not found.")
        return pd.DataFrame()
    
    subdomains = pd.read_csv(subdomain_file)
    
    # Merge with subdomain info
    merged = combined.merge(subdomains[['Unique_Serial', 'subdomain_name']], left_on='question_id', right_on='Unique_Serial', how='left')
    
    # Filter out rows without subdomain
    merged = merged.dropna(subset=['subdomain_name'])
    
    # Compute statistics per subdomain
    stats = merged.groupby('subdomain_name').agg({
        'is_correct': ['mean', 'std', 'count']
    }).round(3)
    
    stats.columns = ['mean_accuracy', 'std_across_models', 'n_questions']
    stats = stats.reset_index()
    
    # Filter subdomains with at least 5 questions
    stats = stats[stats['n_questions'] >= 5]
    
    return stats

def create_clean_visualization(stats):
    """Create a clean, readable visualization."""
    if stats.empty:
        print("No data to visualize.")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), dpi=150)
    fig.suptitle('Subdomain Performance Analysis', fontsize=24, fontweight='bold', y=0.95)
    
    # 1. Difficulty Distribution (Bar Chart)
    stats_sorted = stats.sort_values('mean_accuracy', ascending=True)
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.6 else 'green' for acc in stats_sorted['mean_accuracy']]
    
    bars = ax1.barh(range(len(stats_sorted)), stats_sorted['mean_accuracy'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(stats_sorted)))
    ax1.set_yticklabels(stats_sorted['subdomain_name'], fontsize=10)
    ax1.set_xlabel('Mean Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Subdomain Difficulty (Lower = Harder)', fontsize=16, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Hard Threshold')
    ax1.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Medium Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Consistency Analysis (Bar Chart)
    consistency_sorted = stats.sort_values('std_across_models', ascending=True)
    colors_cons = ['red' if std > 0.15 else 'orange' if std > 0.1 else 'green' for std in consistency_sorted['std_across_models']]
    
    bars2 = ax2.barh(range(len(consistency_sorted)), consistency_sorted['std_across_models'], color=colors_cons, alpha=0.7)
    ax2.set_yticks(range(len(consistency_sorted)))
    ax2.set_yticklabels(consistency_sorted['subdomain_name'], fontsize=10)
    ax2.set_xlabel('Standard Deviation', fontsize=14, fontweight='bold')
    ax2.set_title('Subdomain Consistency (Lower = More Consistent)', fontsize=16, fontweight='bold')
    ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.7, label='Consistent')
    ax2.axvline(x=0.15, color='orange', linestyle='--', alpha=0.7, label='Inconsistent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quadrant Analysis (Clean Scatter)
    # Create quadrants based on medians
    median_acc = stats['mean_accuracy'].median()
    median_std = stats['std_across_models'].median()
    
    # Categorize subdomains
    stats['quadrant'] = 'Other'
    stats.loc[(stats['mean_accuracy'] >= median_acc) & (stats['std_across_models'] <= median_std), 'quadrant'] = 'Easy & Consistent'
    stats.loc[(stats['mean_accuracy'] >= median_acc) & (stats['std_across_models'] > median_std), 'quadrant'] = 'Easy & Inconsistent'
    stats.loc[(stats['mean_accuracy'] < median_acc) & (stats['std_across_models'] <= median_std), 'quadrant'] = 'Hard & Consistent'
    stats.loc[(stats['mean_accuracy'] < median_acc) & (stats['std_across_models'] > median_std), 'quadrant'] = 'Hard & Inconsistent'
    
    # Color map for quadrants
    quadrant_colors = {
        'Easy & Consistent': 'green',
        'Easy & Inconsistent': 'orange', 
        'Hard & Consistent': 'blue',
        'Hard & Inconsistent': 'red',
        'Other': 'gray'
    }
    
    for quadrant, color in quadrant_colors.items():
        subset = stats[stats['quadrant'] == quadrant]
        if not subset.empty:
            ax3.scatter(subset['mean_accuracy'], subset['std_across_models'], 
                       c=color, s=subset['n_questions']*10, alpha=0.7, 
                       label=f'{quadrant} ({len(subset)})', edgecolors='white', linewidth=1)
    
    # Add quadrant lines
    ax3.axvline(x=median_acc, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=median_std, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax3.text(0.02, 0.98, 'Hard & Inconsistent', transform=ax3.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3), verticalalignment='top')
    ax3.text(0.98, 0.98, 'Easy & Inconsistent', transform=ax3.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3), verticalalignment='top', horizontalalignment='right')
    ax3.text(0.02, 0.02, 'Hard & Consistent', transform=ax3.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3), verticalalignment='bottom')
    ax3.text(0.98, 0.02, 'Easy & Consistent', transform=ax3.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3), verticalalignment='bottom', horizontalalignment='right')
    
    ax3.set_xlabel('Mean Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=14, fontweight='bold')
    ax3.set_title('Quadrant Analysis', fontsize=16, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top/Bottom Performers Table
    ax4.axis('off')
    
    # Create summary table
    top_easy = stats.nlargest(5, 'mean_accuracy')[['subdomain_name', 'mean_accuracy', 'std_across_models']]
    top_consistent = stats.nsmallest(5, 'std_across_models')[['subdomain_name', 'mean_accuracy', 'std_across_models']]
    top_hard = stats.nsmallest(5, 'mean_accuracy')[['subdomain_name', 'mean_accuracy', 'std_across_models']]
    
    table_data = []
    table_data.append(['EASIEST SUBDOMAINS', '', ''])
    for _, row in top_easy.iterrows():
        table_data.append([row['subdomain_name'], f"{row['mean_accuracy']:.3f}", f"{row['std_across_models']:.3f}"])
    
    table_data.append(['', '', ''])
    table_data.append(['MOST CONSISTENT', '', ''])
    for _, row in top_consistent.iterrows():
        table_data.append([row['subdomain_name'], f"{row['mean_accuracy']:.3f}", f"{row['std_across_models']:.3f}"])
    
    table_data.append(['', '', ''])
    table_data.append(['HARDEST SUBDOMAINS', '', ''])
    for _, row in top_hard.iterrows():
        table_data.append([row['subdomain_name'], f"{row['mean_accuracy']:.3f}", f"{row['std_across_models']:.3f}"])
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Subdomain', 'Accuracy', 'Std Dev'],
                     cellLoc='left', loc='center',
                     colWidths=[0.5, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data)):
        if i == 0 or i == 6 or i == 12:  # Header rows
            for j in range(3):
                table[(i+1, j)].set_facecolor('#4CAF50')
                table[(i+1, j)].set_text_props(weight='bold', color='white')
        elif i in [1,2,3,4,5]:  # Easy subdomains
            for j in range(3):
                table[(i+1, j)].set_facecolor('#E8F5E8')
        elif i in [7,8,9,10,11]:  # Consistent subdomains
            for j in range(3):
                table[(i+1, j)].set_facecolor('#E3F2FD')
        else:  # Hard subdomains
            for j in range(3):
                table[(i+1, j)].set_facecolor('#FFEBEE')
    
    ax4.set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("subdomain_analysis_clean.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Clean visualization saved to {output_file}")
    
    # Also save as PDF
    pdf_file = Path("subdomain_analysis_clean.pdf")
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Clean visualization PDF saved to {pdf_file}")
    
    plt.show()

def main():
    """Main function."""
    print("Loading subdomain statistics...")
    stats = load_llm_results()
    
    if stats.empty:
        print("No data found. Make sure you have:")
        print("1. LLM test result files in current directory")
        print("2. 'merged_all_questions_with_subdomains_renamed.csv' file")
        return
    
    print(f"Found {len(stats)} subdomains with sufficient data.")
    print("\nCreating clean visualization...")
    create_clean_visualization(stats)

if __name__ == "__main__":
    main()
