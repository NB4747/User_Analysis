#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def user_analysis(df):

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 基础画像分析
    basic_profile_analysis(df)

    # 兴趣偏好分析
    interest_preference_analysis(df)

    # 用户价值分层分析
    user_value_segmentation(df)

    # 交叉分析
    cross_analysis(df)

    return df


def basic_profile_analysis(df):
    """基础画像分析"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 年龄分布
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100],
                             labels=['25岁以下', '25-35岁', '35-45岁', '45岁以上'])
    age_dist = df['Age_Group'].value_counts().sort_index()
    axes[0, 0].pie(age_dist.values, labels=age_dist.index, autopct='%1.1f%%')
    axes[0, 0].set_title('年龄分布')

    # 性别分布
    gender_dist = df['Gender'].value_counts()
    axes[0, 1].bar(gender_dist.index, gender_dist.values, color=['#FF9999', '#66B2FF'])
    axes[0, 1].set_title('性别分布')

    # 地域分布
    location_dist = df['Location'].value_counts().head(5)
    axes[1, 0].bar(location_dist.index, location_dist.values, color='lightgreen')
    axes[1, 0].set_title('地域分布(TOP5)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 收入分布
    income_dist = df['Income'].value_counts().sort_index()
    axes[1, 1].bar(income_dist.index, income_dist.values, color='orange')
    axes[1, 1].set_title('收入分布')

    plt.tight_layout()
    plt.show()

    # 基础统计
    print(f"平均年龄: {df['Age'].mean():.1f}岁")
    print(
        f"性别比例: 男{gender_dist.get('Male', 0) / len(df) * 100:.1f}% vs 女{gender_dist.get('Female', 0) / len(df) * 100:.1f}%")
    print(f"主要地域: {location_dist.index[0]}({location_dist.iloc[0]}人)")


def interest_preference_analysis(df):
    """兴趣偏好分析"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 兴趣分布
    interest_dist = df['Interests'].value_counts().head(8)
    axes[0].bar(interest_dist.index, interest_dist.values, color=plt.cm.Set3(range(len(interest_dist))))
    axes[0].set_title('用户兴趣分布')
    axes[0].tick_params(axis='x', rotation=45)

    # 品类偏好
    category_dist = df['Product_Category_Preference'].value_counts().head(8)
    axes[1].bar(category_dist.index, category_dist.values, color=plt.cm.Pastel1(range(len(category_dist))))
    axes[1].set_title('品类偏好分布')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print(f"最受欢迎兴趣: {interest_dist.index[0]}({interest_dist.iloc[0]}人)")
    print(f"最受欢迎品类: {category_dist.index[0]}({category_dist.iloc[0]}人)")


def user_value_segmentation(df):
    """用户价值分层分析"""

    df['Value_Score'] = (df['Total_Spending'] / df['Total_Spending'].max() * 0.6 +
                         df['Purchase_Frequency'] / df['Purchase_Frequency'].max() * 0.4)

    df['Value_Level'] = pd.cut(df['Value_Score'], bins=[0, 0.3, 0.7, 1],
                               labels=['低价值', '中价值', '高价值'])

    value_dist = df['Value_Level'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.pie(value_dist.values, labels=value_dist.index, autopct='%1.1f%%',
            colors=['#FF9999', '#FFD700', '#90EE90'])
    plt.title('用户价值分层分布')
    plt.show()

    # 各价值层级特征
    value_analysis = df.groupby('Value_Level').agg({
        'Total_Spending': 'mean',
        'Purchase_Frequency': 'mean',
        'Time_Spent_on_Site_Minutes': 'mean'
    }).round(2)

    print("各价值层级特征:")
    print(value_analysis)


def cross_analysis(df):
    """交叉分析"""
    print("\n4. 交叉分析")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 年龄vs消费
    age_spending = df.groupby('Age_Group')['Total_Spending'].mean()
    axes[0, 0].bar(age_spending.index, age_spending.values, color='skyblue')
    axes[0, 0].set_title('各年龄段平均消费')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 性别vs消费
    gender_spending = df.groupby('Gender')['Total_Spending'].mean()
    axes[0, 1].bar(gender_spending.index, gender_spending.values, color=['#FF9999', '#66B2FF'])
    axes[0, 1].set_title('性别平均消费对比')

    # 收入vs消费
    income_spending = df.groupby('Income')['Total_Spending'].mean()
    axes[1, 0].bar(income_spending.index, income_spending.values, color='orange')
    axes[1, 0].set_title('各收入层级平均消费')

    # 兴趣vs活跃度
    interest_engagement = df.groupby('Interests')['Time_Spent_on_Site_Minutes'].mean().nlargest(6)
    axes[1, 1].bar(interest_engagement.index, interest_engagement.values, color='lightgreen')
    axes[1, 1].set_title('各兴趣群体平均停留时间(TOP6)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 关键洞察
    print("关键洞察:")
    print(f"• 消费力最强年龄段: {age_spending.idxmax()}(¥{age_spending.max():.0f})")
    print(f"• 消费力最强性别: {gender_spending.idxmax()}(¥{gender_spending.max():.0f})")
    print(f"• 最活跃兴趣群体: {interest_engagement.idxmax()}({interest_engagement.max():.1f}分钟)")


def main():
    # 读取数据
    df = pd.read_csv("E:\\数分数据集\\user_personalized_features.csv")

    df = user_analysis(df)

    return df


if __name__ == "__main__":
    df_result = main()