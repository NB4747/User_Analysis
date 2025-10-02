#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

"""将RFM进行打分"""
def score_column(df, column_name):
    column_data = df[column_name]
    max_value = column_data.max()
    min_value = column_data.min()
    print(f"\n{column_name} - 最大值: {max_value}")
    print(f"{column_name} - 最小值: {min_value}")
    print(f"{column_name} - 数据范围: {min_value} 到 {max_value}")
    # 计算区间宽度（整数）
    range_total = max_value - min_value
    interval_width = (range_total + 5) // 6
    interval_width = max(1, interval_width)
    # 生成6个区间的整数边界
    bins = []
    current = min_value
    for i in range(7):
        bins.append(current)
        current += interval_width
    if bins[-1] < max_value:
        bins[-1] = max_value + 1

    print(f"{column_name} - 区间划分:")
    for i in range(6):
        start = bins[i]
        end = bins[i + 1] if i < 5 else max_value
        print(f"区间 {i + 1}: [{start}, {end}]")
    # 定义打分规则：第一个区间6分，依次递减，最后一个区间1分
    scores = [6, 5, 4, 3, 2, 1]
    score_column_name = f"{column_name}_score"
    df[score_column_name] = pd.cut(
        column_data,
        bins=bins,
        labels=scores,
        include_lowest=True
    ).astype(int)  # 转换为整数类型
    print(f"\n{column_name} 添加分数列后的前几行数据:")
    print(df[[column_name, score_column_name]].head())

"""RFM用户分群"""
def create_rfm_segments(df):
    print("\n" + "=" * 50)
    print("开始RFM用户分群分析")
    print("=" * 50)
    # 计算RFM总分和平均分
    df['RFM_Total_Score'] = df['Last_Login_Days_Ago_score'] + df['Purchase_Frequency_score'] + df[
        'Total_Spending_score']
    df['RFM_Avg_Score'] = df['RFM_Total_Score'] / 3
    # 基于分数创建用户分群
    conditions = [
        # 核心用户 - 活跃度，消费频率，消费额都高
        (df['Last_Login_Days_Ago_score'] >= 5) & (df['Purchase_Frequency_score'] >= 5) & (
                    df['Total_Spending_score'] >= 5),
        # 高价值新用户 - 最近活跃、消费高但频率不高
        (df['Last_Login_Days_Ago_score'] >= 5) & (df['Total_Spending_score'] >= 5) & (
                    df['Purchase_Frequency_score'] <= 3),
        # 忠诚用户 - 经常购买但客单价不高
        (df['Last_Login_Days_Ago_score'] >= 5) & (df['Purchase_Frequency_score'] >= 5) & (
                    df['Total_Spending_score'] <= 3),
        # 需唤醒用户 - 曾经价值高但最近不活跃
        (df['Last_Login_Days_Ago_score'] <= 2) & (df['Total_Spending_score'] >= 5) & (
                    df['Purchase_Frequency_score'] >= 4),
        # 潜在用户 - 最近活跃但消费不多
        (df['Last_Login_Days_Ago_score'] >= 5) & (df['RFM_Total_Score'] >= 10),
        # 流失用户 - 各方面都不活跃
        (df['Last_Login_Days_Ago_score'] <= 2) & (df['Purchase_Frequency_score'] <= 2) & (
                    df['Total_Spending_score'] <= 2)
    ]
    choices = [
        '核心用户',
        '高价值新用户',
        '忠诚用户',
        '需唤醒用户',
        '潜在用户',
        '流失用户'
    ]

    df['RFM_Segment'] = np.select(conditions, choices, default='一般用户')

    return df

"""分析各用户群的特征"""
def analyze_segments(df):
    print("\n" + "=" * 50)
    print("各用户分群统计分析")
    print("=" * 50)

    # 各分群用户数量分布
    segment_counts = df['RFM_Segment'].value_counts()
    print("各分群用户数量分布:")
    for segment, count in segment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{segment}: {count}人 ({percentage:.2f}%)")
    # 各分群的关键指标平均值
    segment_analysis = df.groupby('RFM_Segment').agg({
        'Last_Login_Days_Ago': 'mean',
        'Purchase_Frequency': 'mean',
        'Total_Spending': 'mean',
        'Average_Order_Value': 'mean',
        'Time_Spent_on_Site_Minutes': 'mean',
        'User_ID': 'count'
    }).round(2)
    segment_analysis = segment_analysis.rename(columns={'User_ID': '用户数量'})
    print(f"\n各分群关键指标平均值:")
    print(segment_analysis)
    return segment_analysis

"""为每种用户群体生成相关策略"""
def generate_segment_strategies(df):
    print("\n" + "=" * 50)
    print("各用户分群运营策略建议")
    print("=" * 50)
    strategies = {
        '核心用户': {
            '特征': '高活跃、高频率、高消费',
            '策略': [
                '提供VIP专属特权和服务',
                '新品预览和独家购买权',
                '高价值忠诚度奖励',
                '个性化产品推荐'
            ]
        },
        '高价值新用户': {
            '特征': '最近活跃、高消费但频率不高',
            '策略': [
                '推送个性化产品推荐提高复购率',
                '邀请订阅营销资讯',
                '提供多买多优惠活动'
            ]
        },
        '忠诚用户': {
            '特征': '高活跃、高频率但客单价不高',
            '策略': [
                '通过捆绑销售提高客单价',
                '推送高单价互补产品',
                '提供满减优惠券'
            ]
        },
        '需唤醒用户': {
            '特征': '曾经价值高但最近不活跃',
            '策略': [
                '发送"我们想你了"召回邮件',
                '提供高价值回归优惠券',
                '告知平台重要更新和新功能'
            ]
        },
        '潜在用户': {
            '特征': '有一定活跃度但消费潜力未完全释放',
            '策略': [
                '精准营销推送',
                '提供尝鲜优惠券',
                '加强用户教育和引导'
            ]
        },
        '流失用户': {
            '特征': '各方面都不活跃',
            '策略': [
                '低成本广撒网式召回',
                '调查流失原因',
                '考虑暂时放弃以节省成本'
            ]
        },
        '一般用户': {
            '特征': '中等表现用户',
            '策略': [
                '常规营销活动',
                '培育用户习惯',
                '观察向其他分群的转化'
            ]
        }
    }

    for segment, strategy_info in strategies.items():
        print(f"\n【{segment}】")
        print(f"特征: {strategy_info['特征']}")
        print("推荐策略:")
        for i, strategy in enumerate(strategy_info['策略'], 1):
            print(f"  {i}. {strategy}")

"""相关图表"""
def visualize_rfm_analysis(df):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 用户分群分布饼图
    segment_counts = df['RFM_Segment'].value_counts()
    axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('用户分群分布', fontsize=14, fontweight='bold')

    # 各分群平均消费金额柱状图
    segment_spending = df.groupby('RFM_Segment')['Total_Spending'].mean().sort_values(ascending=False)
    bars = axes[0, 1].bar(segment_spending.index, segment_spending.values, color='skyblue')
    axes[0, 1].set_title('各分群平均总消费金额', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, segment_spending.values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(segment_spending.values) * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=9)

    # 各分群最近登录天数柱状图
    segment_login = df.groupby('RFM_Segment')['Last_Login_Days_Ago'].mean().sort_values(ascending=True)
    bars_h = axes[1, 0].barh(segment_login.index, segment_login.values, color='lightgreen')
    axes[1, 0].set_title('各分群平均最近登录天数', fontsize=14, fontweight='bold')
    for bar, value in zip(bars_h, segment_login.values):
        axes[1, 0].text(bar.get_width() + max(segment_login.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{value:.0f}天', ha='left', va='center', fontsize=9)

    # RFM分数分布箱线图
    rfm_scores = df[['Last_Login_Days_Ago_score', 'Purchase_Frequency_score', 'Total_Spending_score']]
    box_plot = axes[1, 1].boxplot([rfm_scores['Last_Login_Days_Ago_score'],
                                   rfm_scores['Purchase_Frequency_score'],
                                   rfm_scores['Total_Spending_score']],
                                  labels=['最近登录', '购买频率', '总消费'],
                                  patch_artist=True)  # 允许填充颜色
    colors = ['lightblue', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    axes[1, 1].set_title('RFM各维度分数分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('分数')

    plt.tight_layout()

    plt.show()


def main():
    # 读取数据
    df = pd.read_csv("E:\\数分数据集\\user_personalized_features.csv")

    print("数据基本信息:")
    print(f"数据集形状: {df.shape}")
    print(f"数据列名: {df.columns.tolist()}")

    # 对RFM相关列进行打分
    score_columns = ["Last_Login_Days_Ago", "Purchase_Frequency", "Average_Order_Value", "Total_Spending"]
    for col in score_columns:
        score_column(df, col)

    # 创建RFM用户分群
    df = create_rfm_segments(df)

    # 分析各分群特征
    segment_analysis = analyze_segments(df)

    # 生成策略建议
    generate_segment_strategies(df)

    # 可视化分析结果
    visualize_rfm_analysis(df)

    print("\n" + "=" * 50)
    print("RFM用户价值分析完成！")
    print("=" * 50)
    print(f"总用户数: {len(df)}")
    print("\n分析结果包括:")
    print("1. 用户分群统计信息")
    print("2. 各分群关键指标分析")
    print("3. 针对性的运营策略建议")
    print("4. 可视化图表 (rfm_analysis_visualization.png)")


if __name__ == "__main__":
    main()