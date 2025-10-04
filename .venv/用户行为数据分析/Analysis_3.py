#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

"""兴趣与行为的匹配度分析"""


def analyze_interest_behavior(df):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据预处理和兴趣匹配度计算
    df = prepare_match(df)

    # 匹配度对用户行为的影响分析
    analyze_impact(df)

    # 推荐和建议
    recommendation(df)

    return df


# 数据预处理和兴趣匹配度计算
def prepare_match(df):
    # 定义兴趣-偏好品类映射关系
    interest_category = {
        'Sports': ['Apparel', 'Health & Beauty'],
        'Technology': ['Electronics', 'Books', 'Home & Kitchen'],
        'Fashion': ['Apparel', 'Health & Beauty'],
        'Travel': ['Books', 'Electronics', 'Apparel'],
        'Food': ['Home & Kitchen', 'Health & Beauty', 'Books']
    }

    # 计算兴趣与偏好的匹配度
    def calculate_match_score(interest, preference):
        if pd.isna(interest) or pd.isna(preference):
            return 0

        interest_str = str(interest).strip()
        preference_str = str(preference).strip()

        # 如果兴趣在映射表中，检查偏好是否匹配
        if interest_str in interest_category:
            related_categories = interest_category[interest_str]
            if preference_str in related_categories:
                return 1.0  # 完全匹配
        return 0.0  # 不匹配

    # 应用匹配度计算
    df['Match_Score'] = df.apply(
        lambda row: calculate_match_score(row['Interests'], row['Product_Category_Preference']),
        axis=1
    )

    # 定义匹配等级
    def define_match_level(score):
        if score >= 0.7:
            return '高匹配'
        elif score >= 0.4:
            return '中匹配'
        else:
            return '低匹配'

    df['Match_Level'] = df['Match_Score'].apply(define_match_level)

    print("数据预处理完成")
    print(f"平均匹配度: {df['Match_Score'].mean():.3f}")

    # 显示匹配等级分布
    match_stats = df['Match_Level'].value_counts()
    for level, count in match_stats.items():
        percentage = (count / len(df)) * 100
        print(f"{level}: {count}人 ({percentage:.1f}%)")

    return df


# 匹配度对用户行为的影响分析
def analyze_impact(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('匹配度对用户行为的影响分析', fontsize=16, fontweight='bold')

    # 1.不同匹配等级的关键指标对比
    match_behavior = df.groupby('Match_Level').agg({
        'Purchase_Frequency': 'mean',
        'Total_Spending': 'mean',
        'Time_Spent_on_Site_Minutes': 'mean',
        'Pages_Viewed': 'mean'
    }).round(2)

    level_order = ['高匹配', '中匹配', '低匹配']
    match_behavior = match_behavior.reindex(level_order)

    # 绘制多指标柱状图
    x = np.arange(len(match_behavior))
    width = 0.2
    metrics = ['Purchase_Frequency', 'Total_Spending', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed']
    metric_labels = ['购买频率', '消费金额', '停留时间(分钟)', '浏览页面数']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i in range(len(metrics)):
        metric = metrics[i]
        label = metric_labels[i]
        color = colors[i]

        # 获取当前指标的数据
        data = match_behavior[metric].values

        axes[0].bar(
            x + i * width,
            data,
            width,
            label=label,
            color=color,
            alpha=0.8
        )

    axes[0].set_xlabel('匹配等级')
    axes[0].set_ylabel('平均值')
    axes[0].set_title('不同匹配等级用户行为对比')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(match_behavior.index)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # 2.各兴趣群体的匹配度与消费关系
    interests = ['Sports', 'Technology', 'Fashion', 'Travel', 'Food']
    colors_interest = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC']

    for i, interest in enumerate(interests):
        interest_data = df[df['Interests'] == interest]
        if len(interest_data) > 0:
            avg_match = interest_data['Match_Score'].mean()
            avg_spending = interest_data['Total_Spending'].mean()

            axes[1].scatter(avg_match, avg_spending, s=100, color=colors_interest[i], label=interest)
            axes[1].text(avg_match, avg_spending, interest, fontsize=9, ha='center', va='bottom')

    axes[1].set_xlabel('平均匹配度')
    axes[1].set_ylabel('平均消费金额')
    axes[1].set_title('各兴趣群体匹配度与消费关系')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # 简化的分析报告
    print("\n分析结果:")

    # 高匹配 vs 低匹配对比
    high_match = df[df['Match_Level'] == '高匹配']
    low_match = df[df['Match_Level'] == '低匹配']

    if len(high_match) > 0 and len(low_match) > 0:
        spending_diff = high_match['Total_Spending'].mean() - low_match['Total_Spending'].mean()
        frequency_diff = high_match['Purchase_Frequency'].mean() - low_match['Purchase_Frequency'].mean()
        print(f"高匹配用户比低匹配用户平均多消费: ¥{spending_diff:.0f}")
        print(f"高匹配用户购买频率比低匹配用户高: {frequency_diff:.2f}")

    # 各兴趣群体表现
    print(f"\n各兴趣群体表现:")
    for interest in interests:
        interest_data = df[df['Interests'] == interest]
        if len(interest_data) > 0:
            match_rate = interest_data['Match_Score'].mean()
            avg_spending = interest_data['Total_Spending'].mean()
            user_count = len(interest_data)
            print(f"{interest}: {match_rate:.0%}匹配, ¥{avg_spending:.0f}消费, {user_count}人")


# 推荐效果评估和优化建议
def recommendation(df):
    print("\n推荐效果评估和优化建议:")
    print("-" * 40)

    # 基础评估指标
    overall_match_rate = df['Match_Score'].mean()
    high_match_ratio = len(df[df['Match_Level'] == '高匹配']) / len(df)

    print(f"整体匹配度: {overall_match_rate:.3f}")
    print(f"高匹配用户占比: {high_match_ratio:.1%}")

    # 各兴趣匹配情况分析
    interests = ['Sports', 'Technology', 'Fashion', 'Travel', 'Food']
    print(f"\n各兴趣匹配详情:")

    for interest in interests:
        interest_users = df[df['Interests'] == interest]
        if len(interest_users) > 0:
            match_rate = interest_users['Match_Score'].mean()
            user_count = len(interest_users)
            matched_count = len(interest_users[interest_users['Match_Score'] > 0])

            print(f"{interest}: {match_rate:.1%}匹配率 ({matched_count}/{user_count}人)")

    # 优化建议
    print(f"\n优化建议:")
    if overall_match_rate < 0.5:
        print("1.   当前匹配度较低，建议:")
        print("   • 重新评估兴趣-品类映射关系")
        print("   • 增加品类覆盖范围")
    else:
        print("1.   匹配度良好，建议:")
        print("   • 持续优化推荐算法")
        print("   • 关注低匹配兴趣群体")

    # 识别需要优化的兴趣群体
    low_match_interests = []
    for interest in interests:
        interest_users = df[df['Interests'] == interest]
        if len(interest_users) > 0:
            match_rate = interest_users['Match_Score'].mean()
            if match_rate < 0.5:
                low_match_interests.append((interest, match_rate))

    if low_match_interests:
        print("2. 需要重点优化的兴趣:")
        for interest, match_rate in low_match_interests:
            print(f"   • {interest} (当前匹配率: {match_rate:.1%})")


def main():
    # 读取数据
    df = pd.read_csv("E:\\数分数据集\\user_personalized_features.csv")
    # 执行兴趣与行为匹配度分析
    df = analyze_interest_behavior(df)

    return df


if __name__ == "__main__":
    df_result = main()