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


def analysis(df):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 相关性分析
    corr_analysis(df)
    # 四象限分析
    df = four_analysis(df)
    return df


"""相关性分析"""


def corr_analysis(df):
    # 确定分析变量
    """自变量"""
    e_vars = ['Time_Spent_on_Site_Minutes', 'Pages_Viewed', 'Last_Login_Days_Ago']
    """因变量"""
    c_vars = ['Purchase_Frequency', 'Total_Spending', 'Average_Order_Value']
    all_vars = e_vars + c_vars

    # 计算相关系数
    c = df[all_vars].corr()

    # 可视化相关性矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 相关性热力图
    mask = np.triu(np.ones_like(c, dtype=bool))
    sns.heatmap(c, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', ax=axes[0], mask=mask,
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('参与度与转化率相关性热力图', fontsize=14, fontweight='bold')

    # 关键关系散点图
    scatter = axes[1].scatter(df['Time_Spent_on_Site_Minutes'],
                              df['Total_Spending'],
                              c=df['Purchase_Frequency'],
                              cmap='viridis', alpha=0.6, s=50)
    axes[1].set_xlabel('网站停留时间')
    axes[1].set_ylabel('总消费金额')
    axes[1].set_title('时间花费 vs 消费金额 (颜色表示购买频率)', fontsize=12)
    plt.colorbar(scatter, ax=axes[1], label='购买频率')

    plt.tight_layout()
    plt.show()

    # 计算关键相关系数并推测原因
    time_spending_corr = df['Time_Spent_on_Site_Minutes'].corr(df['Total_Spending'])
    pages_freq_corr = df['Pages_Viewed'].corr(df['Purchase_Frequency'])
    if time_spending_corr < 0.3:
        print("时间投入与消费金额相关性较弱，可能存在转化障碍")
    if pages_freq_corr < 0.3:
        print("页面浏览与购买频率相关性较弱，导航或推荐可能需优化")


"""基于数据相关性决定权重分配"""


def date_weight(df):
    # 计算各参与度指标与商业价值的相关性
    corr = {
        '时间花费': df['Time_Spent_on_Site_Minutes'].corr(df['Total_Spending']),
        '页面浏览': df['Pages_Viewed'].corr(df['Total_Spending']),
        # 最近上次登录时间是负面指标，所以需要 1 / (df['Last_Login_Days_Ago']
        '登录活跃度': (1 / (df['Last_Login_Days_Ago'] + 1)).corr(df['Total_Spending'])
    }

    # 基于相关性分配权重（取绝对值，相关性越强权重越高）
    abs_corr = {k: abs(v) for k, v in corr.items()}
    total_corr = sum(abs_corr.values())
    if total_corr > 0:
        optimized_weights = [
            abs_corr['时间花费'] / total_corr,
            abs_corr['页面浏览'] / total_corr,
            abs_corr['登录活跃度'] / total_corr
        ]
    else:
        # 如果相关性都很弱，使用默认权重
        optimized_weights = [0.4, 0.4, 0.2]

    return optimized_weights


"""四象限分析"""


def four_analysis(df):
    # 将数据标准化并且分配权重
    """创建一个最大最小标准化器"""
    scaler = MinMaxScaler()
    # 参与度特征
    engagement_features = pd.DataFrame({
        'time_spent': df['Time_Spent_on_Site_Minutes'],
        'pages_viewed': df['Pages_Viewed'],
        'login_recency': 1 / (df['Last_Login_Days_Ago'] + 1)  # 转换为正向指标
    })
    # 标准化
    engagement_scaled = scaler.fit_transform(engagement_features)
    data_weights = date_weight(df)
    df['Engagement_Score'] = (
            engagement_scaled[:, 0] * data_weights[0] +  # 时间花费
            engagement_scaled[:, 1] * data_weights[1] +  # 页面浏览
            engagement_scaled[:, 2] * data_weights[2]  # 登录活跃度
    )

    # 3.2 商业价值评分
    value_features = pd.DataFrame({
        'total_spending': df['Total_Spending'],
        'purchase_freq': df['Purchase_Frequency'],
        'avg_order_value': df['Average_Order_Value']
    })

    value_scaled = scaler.fit_transform(value_features)

    # 商业价值权重(这里使用的是默认的权重)
    value_weights = [0.5, 0.3, 0.2]  # 总消费50%，频率30%，客单价20%
    df['Value_Score'] = (
            value_scaled[:, 0] * value_weights[0] +
            value_scaled[:, 1] * value_weights[1] +
            value_scaled[:, 2] * value_weights[2]
    )

    # 划分四象限
    engagement_median = df['Engagement_Score'].median()
    value_median = df['Value_Score'].median()

    def quadrant(row):
        if row['Engagement_Score'] >= engagement_median and row['Value_Score'] >= value_median:
            return '核心用户'
        elif row['Engagement_Score'] >= engagement_median and row['Value_Score'] < value_median:
            return '高参与低价值用户'
        elif row['Engagement_Score'] < engagement_median and row['Value_Score'] >= value_median:
            return '低参与高价值用户'
        else:
            return '普通用户'

    df['User_Quadrant'] = df.apply(quadrant, axis=1)

    # 统计各象限的人数
    quadrant_stats = df['User_Quadrant'].value_counts()
    print("各象限用户分布:")
    total_users = len(df)
    for quadrant, count in quadrant_stats.items():
        percentage = (count / total_users) * 100
        print(f"  {quadrant}: {count}人 ({percentage:.1f}%)")

    # 可视化四象限
    visualize_quadrant_analysis(df, engagement_median, value_median)
    # 深度特征分析
    analyze_quadrant_features(df)
    # 生成建议
    generate_strategic_recommendations(df, quadrant_stats)

    return df


# 可视化四象限
def visualize_quadrant_analysis(df, engagement_median, value_median):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列，axes现在是一维数组

    # 用户分布饼图 - 使用 axes[0]
    colors = {
        '核心用户': '#FF6B6B',  # 红色
        '高参与低价值用户': '#4ECDC4',  # 青色
        '低参与高价值用户': '#45B7D1',  # 蓝色
        '普通用户': '#96CEB4'  # 绿色
    }
    quadrant_counts = df['User_Quadrant'].value_counts()
    axes[0].pie(quadrant_counts.values, labels=quadrant_counts.index,
                autopct='%1.1f%%', startangle=90,
                colors=[colors.get(q, 'gray') for q in quadrant_counts.index])
    axes[0].set_title('用户四象限分布', fontsize=14, fontweight='bold')

    # 各象限平均消费金额对比
    quadrant_spending = df.groupby('User_Quadrant')['Total_Spending'].mean()
    # 按预设顺序排序
    ordered_quadrants = ['核心用户', '高参与低价值用户', '低参与高价值用户', '普通用户']
    quadrant_spending = quadrant_spending.reindex(ordered_quadrants)

    bars = axes[1].bar(quadrant_spending.index, quadrant_spending.values,
                       color=[colors.get(q, 'gray') for q in quadrant_spending.index])
    axes[1].set_xlabel('用户象限')
    axes[1].set_ylabel('平均消费金额')
    axes[1].set_title('各象限平均消费金额对比', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    # 在柱状图上添加数值标签
    for bar, value in zip(bars, quadrant_spending.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(quadrant_spending.values) * 0.01,
                     f'¥{value:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


# 深度特征分析
def analyze_quadrant_features(df):

    # 关键指标
    Key_Indicators = ['Time_Spent_on_Site_Minutes', 'Pages_Viewed', 'Last_Login_Days_Ago',
                      'Purchase_Frequency', 'Total_Spending', 'Average_Order_Value']

    # 深度洞察
    key_users = df[df['User_Quadrant'] == '核心用户']
    high_engage_low_value = df[df['User_Quadrant'] == '高参与低价值用户']
    low_engage_high_value = df[df['User_Quadrant'] == '低参与高价值用户']

    if len(key_users) > 0:
        print(f"核心用户特征:")
        print(f"  • 平均停留时间: {key_users['Time_Spent_on_Site_Minutes'].mean():.1f}分钟")
        print(f"  • 平均消费金额: ¥{key_users['Total_Spending'].mean():.0f}")
        print(f"  • 平均购买频率: {key_users['Purchase_Frequency'].mean():.1f}")

    if len(high_engage_low_value) > 0:
        print(f"高参与低价值用户特征:")
        print(f"  • 参与度高但消费低，可能存在转化障碍")
        print(
            f"  • 平均浏览{high_engage_low_value['Pages_Viewed'].mean():.1f}页，但仅消费¥{high_engage_low_value['Total_Spending'].mean():.0f}")

    if len(low_engage_high_value) > 0:
        print(f"低参与高价值用户特征:")
        print(f"  • 消费能力强但参与度低，是重要机会点")
        print(
            f"  • 平均{low_engage_high_value['Last_Login_Days_Ago'].mean():.1f}天未登录，但消费¥{low_engage_high_value['Total_Spending'].mean():.0f}")


# 生成策略
def generate_strategic_recommendations(df, quadrant_stats):

    total_users = len(df)
    recommendations = {
        '核心用户': {
            '目标': '维护和提升终身价值',
            '策略': [
                'VIP专属服务和优先支持',
                '新品预览和独家购买权',
                '高价值忠诚度奖励计划',
                '个性化产品推荐'
            ],
            'KPI': '复购率、客单价提升、推荐率'
        },
        '高参与低价值用户': {
            '目标': '提高转化率和客单价',
            '策略': [
                '分析并优化转化路径障碍',
                '推送个性化优惠和促销',
                '推荐高价值关联商品',
                '提供会员升级激励'
            ],
            'KPI': '转化率、客单价、会员升级率'
        },
        '低参与高价值用户': {
            '目标': '提高参与度和活跃频率',
            '策略': [
                '个性化召回活动和内容推送',
                '回归专属优惠和特权',
                '邀请参与专属活动和社群',
                '定期推送个性化内容'
            ],
            'KPI': '登录频率、停留时间、页面浏览'
        },
        '普通用户': {
            '目标': '培育基础和引导转化',
            '策略': [
                '基础用户教育和引导',
                '推送入门级产品和优惠',
                '培育使用习惯和品牌认知',
                '低成本触达和培育'
            ],
            'KPI': '激活率、首次购买率'
        }
    }

    for user, strategy in recommendations.items():
        count = quadrant_stats.get(user, 0)
        ratio = (count / total_users) * 100

        print(f"\n【{user}】({count}人, {ratio:.1f}%)")
        print(f"目标: {strategy['目标']}")
        print("策略:")
        for i, tactic in enumerate(strategy['策略'], 1):
            print(f"  {i}. {tactic}")
        print(f"关键指标: {strategy['KPI']}")


def main():
    # 读取数据
    df = pd.read_csv("E:\\数分数据集\\user_personalized_features.csv")

    # 执行分析
    df_analyzed = analysis(df)

    return df_analyzed


if __name__ == "__main__":
    df_result = main()