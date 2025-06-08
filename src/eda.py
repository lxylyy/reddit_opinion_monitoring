import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Rectangle

def read_and_concat_csv_files():
    """
    Reads all CSV files ending with '_sentiment_analysis.csv' in the current directory,
    concatenates them into a single DataFrame, and returns it.
    """
    data = pd.DataFrame()
    for filename in os.listdir('.'):
        if filename.endswith('_sentiment_analysis.csv'):
            print(f"Processing file: {filename}")
            # Read the CSV file
            data_current = pd.read_csv(filename)
            data = pd.concat([data, data_current], ignore_index=True)
    return data


# 创建一个函数来分类情感
def classify_sentiment(row):
    if row['sentiment_selftext_score'] > 0 and row['comment_score'] > 0:
        return 'post_positive_comment_positive'
    elif row['sentiment_selftext_score'] < 0 and row['comment_score'] < 0:
        return 'post_negative_comment_negative'
    elif row['sentiment_selftext_score'] > 0 and row['comment_score'] < 0:
        return 'post_positive_comment_negative'
    else:
        return 'post_negative_comment_positive'


def visualize_sentiment_distribution(data):
    """
    Visualizes the sentiment distribution across different subreddits.
    正面情感显示在0轴上方，负面情感显示在0轴下方
    """
    # 应用分类函数
    data['sentiment_category'] = data.apply(classify_sentiment, axis=1)
    data.to_csv('all_sentiment_analysis.csv', index=False)

    # 统计每个子版块中四类情感的数量
    sentiment_counts = data.groupby(['subreddit', 'sentiment_category']).size().unstack(fill_value=0)
    
    # 确保所有四个类别都存在（如果某些类别缺失，用0填充）
    expected_categories = ['post_negative_comment_negative', 'post_negative_comment_positive', 
                          'post_positive_comment_negative', 'post_positive_comment_positive']
    for category in expected_categories:
        if category not in sentiment_counts.columns:
            sentiment_counts[category] = 0
    
    # 重新排序列以确保一致性
    sentiment_counts = sentiment_counts[expected_categories]
    
    # 分离正面和负面情感
    # 负面情感（显示在下方，使用负值）
    negative_post_negative_comment = -sentiment_counts['post_negative_comment_negative']
    negative_post_positive_comment = -sentiment_counts['post_negative_comment_positive']
    
    # 正面情感（显示在上方，使用正值）
    positive_post_negative_comment = sentiment_counts['post_positive_comment_negative']
    positive_post_positive_comment = sentiment_counts['post_positive_comment_positive']
    
    # 设置优雅的渐变色彩方案
    colors = {
        'post_negative_comment_negative': 'C0',
        'post_negative_comment_positive': 'C1',
        'post_positive_comment_negative': 'C2',
        'post_positive_comment_positive': 'C3'
    }
    plt.rcParams['font.family'] = 'Times New Roman'

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取x轴位置
    x_pos = np.arange(len(sentiment_counts.index))
    width = 0.8
    
    # 绘制负面情感（下方）
    bottom_neg = np.zeros(len(sentiment_counts.index))
    ax.bar(x_pos, negative_post_negative_comment, width, 
           label='Negative Post + Negative Comment', color=colors['post_negative_comment_negative'],
           bottom=bottom_neg, alpha=0.8)
    bottom_neg += negative_post_negative_comment
    
    ax.bar(x_pos, negative_post_positive_comment, width,
           label='Negative Post + Positive Comment', color=colors['post_negative_comment_positive'],
           bottom=bottom_neg, alpha=0.8)

    # 绘制正面情感（上方）
    bottom_pos = np.zeros(len(sentiment_counts.index))
    ax.bar(x_pos, positive_post_negative_comment, width,
           label='Positive Post + Negative Comment', color=colors['post_positive_comment_negative'],
           bottom=bottom_pos, alpha=0.8)
    bottom_pos += positive_post_negative_comment
    
    ax.bar(x_pos, positive_post_positive_comment, width,
           label='Positive Post + Positive Comment', color=colors['post_positive_comment_positive'],
           bottom=bottom_pos, alpha=0.8)
    
    # 添加0轴基准线
    ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.8)
    
    # 设置图表样式
    ax.set_title('Sentiment Distribution Across Subreddits', fontsize=25, pad=20)
    ax.set_xlabel('Subreddit', fontsize=20)
    ax.set_ylabel('Post Count', fontsize=20)

    # 设置x轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sentiment_counts.index, rotation=45, ha='right', fontsize=16)
    
    # 设置y轴
    ax.tick_params(axis='y', labelsize=16)

    # 添加网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(title='Sentiment Category', fontsize=16, title_fontsize=18, 
              loc='lower right', framealpha=0.9, fancybox=True)
    
    # 添加正负区域标签
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    
    ax.text(0.4, 0.95, 'Positive Sentiment Area', transform=ax.transAxes, 
            fontsize=16, alpha=0.7, ha='center')
    ax.text(0.4, 0.05, 'Negative Sentiment Area', transform=ax.transAxes, 
            fontsize=16, alpha=0.7, ha='center')

    # 优化布局
    plt.tight_layout()
    
    # 设置背景颜色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    # 可选：保存图表
    plt.savefig('sentiment_distribution.png')
    # 显示图表
    plt.show()

def sentiment_similarity(row):
    """
    Computes the sentiment similarity between selftext and comment scores.
    Returns a value between 0 and 1, where 1 means identical sentiment.
    """
    return 1 - abs(row['sentiment_selftext_score'] - row['comment_score'])


def get_sentiment_similarity_avg(data):
    """
    Computes the similarity between selftext and comment sentiments.
    """
    data['sentiment_similarity'] = data.apply(sentiment_similarity, axis=1)
    data.to_csv('all_sentiment_analysis.csv', index=False)
    
    # 计算相似度的平均值 by subreddit
    similarity_avg = data.groupby('subreddit')['sentiment_similarity'].mean()

    print(f"Average Sentiment Similarity:", similarity_avg)
    
    return similarity_avg



def create_subreddit_sentiment_visualization(data, 
                                           title="Sentiment Similarity and Keyword Distribution\nAcross Mental Health Subreddits",
                                           figsize=(12, 10), save_path=None, show_plot=True):
    """
    Create a four-quadrant visualization of subreddit sentiment similarity and keywords.
    
    Parameters:
    -----------
    sentiment_data : dict, optional
        Dictionary with subreddit names as keys and similarity scores as values
        Default uses the provided data
    keywords_data : dict, optional
        Dictionary with subreddit names as keys and list of (word, frequency) tuples as values
        Default uses the provided keywords
    title : str, optional
        Main title for the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        Base path for saving files (without extension). If None, uses default names
    show_plot : bool, optional
        Whether to display the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    similarity_avg = get_sentiment_similarity_avg(data)

    # Default sentiment data
    sentiment_data = {
        'r/SuicideWatch': similarity_avg['SuicideWatch'],
        'r/Depression': similarity_avg['depression'],
        'r/MentalHealth': similarity_avg['mentalhealth'],
        'r/OffMyChest': similarity_avg['offmychest']
    }
    
    # Default keywords data
    keywords_data = {
        'r/SuicideWatch': [
            ('suicide', 20), ('kill', 16), ('want', 14), ('crisis', 12), 
            ('help', 10), ('pain', 10), ('alone', 8)
        ],
        'r/Depression': [
            ('tired', 18), ('family', 16), ('addicted', 14), ('struggle', 12),
            ('chronic', 10), ('fatigue', 10), ('empty', 8)
        ],
        'r/MentalHealth': [
            ('clean', 18), ('today', 16), ('health', 14), ('recovery', 12),
            ('progress', 10), ('healing', 10), ('support', 8)
        ],
        'r/OffMyChest': [
            ('cheating', 18), ('husband', 16), ('friend', 14), ('betrayal', 12),
            ('relationship', 10), ('angry', 10), ('trust', 8)
        ]
    }
    
    # Define quadrant positions and colors
    quadrant_config = {
        'r/SuicideWatch': {'pos': (0, 0.5, 0.5, 0.5), 'color': '#d32f2f'},
        'r/Depression': {'pos': (0.5, 0.5, 0.5, 0.5), 'color': '#1976d2'},
        'r/MentalHealth': {'pos': (0, 0, 0.5, 0.5), 'color': '#388e3c'},
        'r/OffMyChest': {'pos': (0.5, 0, 0.5, 0.5), 'color': '#f57c00'}
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Clear axis and set properties
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw quadrant boundaries
    ax.axhline(y=0.5, color='black', linewidth=1.5)
    ax.axvline(x=0.5, color='black', linewidth=1.5)
    
    # Add subtle background rectangles
    for subreddit, config in quadrant_config.items():
        if subreddit in sentiment_data:
            x, y, width, height = config['pos']
            rect = Rectangle((x, y), width, height, 
                           facecolor=config['color'], alpha=0.05, 
                           edgecolor='none')
            ax.add_patch(rect)
    
    # Add content to each quadrant
    for subreddit, config in quadrant_config.items():
        if subreddit not in sentiment_data:
            continue
            
        x, y, width, height = config['pos']
        center_x = x + width/2
        center_y = y + height/2
        
        # Add subreddit label
        ax.text(center_x, center_y + 0.18, subreddit, 
                fontsize=14, fontweight='bold', 
                ha='center', va='center',
                color=config['color'])
        
        # Add similarity score
        similarity_score = sentiment_data[subreddit]
        ax.text(center_x, center_y + 0.14, 
                f'Sentiment Similarity: {similarity_score}',
                fontsize=11, ha='center', va='center',
                color='black')
        
        # Add word cloud
        if subreddit in keywords_data:
            words = keywords_data[subreddit]
            word_positions = [
                (center_x-0.15, center_y+0.06), (center_x, center_y+0.06), (center_x+0.15, center_y+0.06),
                (center_x-0.12, center_y+0.02), (center_x+0.12, center_y+0.02),
                (center_x-0.15, center_y-0.02), (center_x+0.15, center_y-0.02)
            ]
            
            for i, (word, size) in enumerate(words):
                if i < len(word_positions):
                    pos_x, pos_y = word_positions[i]
                    font_size = 8 + (size - 8) * 0.3  # Scale font size based on word importance
                    ax.text(pos_x, pos_y, word,
                           fontsize=font_size, ha='center', va='center',
                           color=config['color'], alpha=0.8,
                           bbox=dict(boxstyle="round,pad=0.2", 
                                    facecolor=config['color'], alpha=0.1,
                                    edgecolor='none'))
    
    # Add title
    ax.text(0.5, 0.95, title,
            fontsize=16, fontweight='bold', ha='center', va='center',
            color='black')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save the figure
    if save_path is None:
        save_path = 'subreddit_sentiment_analysis'
    
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    # Show plot
    if show_plot:
        plt.show()





def main():
    data = read_and_concat_csv_files()
    # filter out Anxiety subreddit
    data = data[data['subreddit'] != 'Anxiety']
    visualize_sentiment_distribution(data)

    # Use with default parameters
    # create_subreddit_sentiment_visualization(data)


if __name__ == "__main__":
    main()
