#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成测试数据脚本
创建模拟的新闻、Reddit和股票价格数据，用于测试情感分类分析
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import SENTIMENT_CATEGORIES, PATHS

# 情感类别关键词
FISCAL_KEYWORDS = SENTIMENT_CATEGORIES['fiscal']['keywords']
DATA_DRIVEN_KEYWORDS = SENTIMENT_CATEGORIES['data_driven']['keywords']
OPINION_KEYWORDS = SENTIMENT_CATEGORIES['opinion']['keywords']

# 新闻标题模板
NEWS_TITLE_TEMPLATES = [
    "市场分析: {company} 股票{direction}",
    "{company} 发布{quarter}财报，{result}预期",
    "分析师{opinion} {company}的未来前景",
    "经济数据显示{indicator}{direction}，{company}股票受影响",
    "{company}宣布{announcement}，股价{direction}",
    "技术分析: {company}股票{pattern}",
    "市场情绪{sentiment}，{company}股票{direction}",
    "{company}的{product}销售{performance}，股价{direction}"
]

# Reddit标题模板
REDDIT_TITLE_TEMPLATES = [
    "我{opinion} {company}的股票",
    "{company}值得投资吗？{opinion}",
    "刚买了{company}的股票，{sentiment}",
    "对{company}的{quarter}财报有何看法？",
    "{company}的技术分析 - {pattern}形成",
    "为什么{company}会{direction}？{opinion}",
    "{company}的长期前景如何？{timeframe}投资者的视角",
    "市场崩盘时{company}会怎样？{opinion}"
]

# 内容生成模板
CONTENT_TEMPLATES = {
    'fiscal': [
        "{company}公布了{quarter}财报，每股收益为{eps}美元，{result}分析师预期的{expected_eps}美元。收入为{revenue}亿美元，{result}预期的{expected_revenue}亿美元。{direction}的主要原因是{reason}。管理层对下一季度的指引{guidance}，预计收入将在{next_revenue}亿美元左右。",
        "最新经济数据显示通胀率{inflation_direction}至{inflation_rate}%，失业率{unemployment_direction}至{unemployment_rate}%。这些数据{impact} {company}等公司的股价。分析师预计美联储将{fed_action}，这可能会{market_impact}市场。",
        "{company}宣布{dividend_action}股息至每股{dividend}美元，这{dividend_impact}投资者信心。同时，公司还宣布了{buyback}亿美元的股票回购计划，这表明管理层{management_confidence}公司的未来前景。"
    ],
    'data_driven': [
        "技术分析显示{company}股票形成了{pattern}形态，这通常是{pattern_indication}信号。相对强弱指标(RSI)为{rsi}，表明股票{rsi_indication}。移动平均线{ma_relation}，这是{ma_indication}趋势的迹象。交易量{volume_trend}，进一步{volume_confirmation}这一观点。",
        "根据我们的量化模型，{company}的股价与其基本面相比{valuation}。该模型考虑了{metrics}等指标，得出的公允价值为{fair_value}美元，比当前价格{price_relation}。历史数据显示，当出现这种差异时，股价通常会在{timeframe}内{price_action}。",
        "对{company}过去10年的数据进行回测表明，当{indicator}达到当前水平时，股票在接下来的{period}内平均{performance}。这一策略的夏普比率为{sharpe}，最大回撤为{drawdown}%。基于这些统计数据，我们的算法给出{rating}评级。"
    ],
    'opinion': [
        "我{belief} {company}是一个{quality}的投资机会。尽管市场{market_sentiment}，但我{confidence}公司能够{performance}。许多投资者可能{other_investors_action}，但我认为这是{opportunity_type}。",
        "市场情绪目前非常{market_mood}，这使得像{company}这样的股票{stock_reaction}。我个人{feeling}这种反应{reaction_quality}。从长远来看，我{expectation}公司会{long_term_performance}，因为{reason}。",
        "作为一个{investor_type}投资者，我{opinion} {company}的管理团队。他们的战略决策{decision_quality}，这让我对公司的未来{confidence_level}。虽然短期内可能会有{short_term_challenges}，但我{belief}长期前景{long_term_outlook}。"
    ]
}

# 公司名称
COMPANIES = [
    "苹果", "微软", "谷歌", "亚马逊", "特斯拉", "英伟达", "Meta", "Netflix",
    "阿里巴巴", "腾讯", "百度", "京东", "拼多多", "美团", "小米", "华为"
]

def generate_date_range(start_date='2023-01-01', end_date=None, freq='D'):
    """
    生成日期范围
    
    Args:
        start_date: 开始日期
        end_date: 结束日期，默认为当前日期
        freq: 频率，默认为每天
        
    Returns:
        日期范围列表
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return date_range

def generate_stock_data(date_range, volatility=0.02, trend=0.0001):
    """
    生成模拟股票价格数据
    
    Args:
        date_range: 日期范围
        volatility: 波动率
        trend: 趋势系数
        
    Returns:
        股票价格数据DataFrame
    """
    # 初始价格
    initial_price = 100.0
    
    # 生成随机价格
    n = len(date_range)
    prices = [initial_price]
    
    for i in range(1, n):
        # 添加随机波动和趋势
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建DataFrame
    stock_data = pd.DataFrame({
        'Date': date_range,
        'Open': prices,
        'Close': prices,
        'High': [p * (1 + random.uniform(0, 0.01)) for p in prices],
        'Low': [p * (1 - random.uniform(0, 0.01)) for p in prices],
        'Volume': [int(random.uniform(1000000, 10000000)) for _ in range(n)]
    })
    
    # 调整开盘价和收盘价
    for i in range(1, n):
        # 随机决定开盘价是高于还是低于前一天收盘价
        if random.random() > 0.5:
            stock_data.loc[i, 'Open'] = stock_data.loc[i-1, 'Close'] * (1 + random.uniform(0, 0.005))
        else:
            stock_data.loc[i, 'Open'] = stock_data.loc[i-1, 'Close'] * (1 - random.uniform(0, 0.005))
        
        # 随机决定收盘价是高于还是低于开盘价
        if random.random() > 0.5:
            stock_data.loc[i, 'Close'] = stock_data.loc[i, 'Open'] * (1 + random.uniform(0, 0.01))
        else:
            stock_data.loc[i, 'Close'] = stock_data.loc[i, 'Open'] * (1 - random.uniform(0, 0.01))
        
        # 确保高低价合理
        stock_data.loc[i, 'High'] = max(stock_data.loc[i, 'Open'], stock_data.loc[i, 'Close']) * (1 + random.uniform(0, 0.005))
        stock_data.loc[i, 'Low'] = min(stock_data.loc[i, 'Open'], stock_data.loc[i, 'Close']) * (1 - random.uniform(0, 0.005))
    
    return stock_data

def generate_text_with_sentiment(sentiment_type):
    """
    根据情感类型生成文本
    
    Args:
        sentiment_type: 情感类型 ('fiscal', 'data_driven', 'opinion')
        
    Returns:
        (标题, 内容) 元组
    """
    company = random.choice(COMPANIES)
    
    # 根据情感类型选择模板
    if sentiment_type == 'fiscal':
        title_template = random.choice([t for t in NEWS_TITLE_TEMPLATES if '{quarter}' in t or '{indicator}' in t or '{announcement}' in t])
        content_template = random.choice(CONTENT_TEMPLATES['fiscal'])
        keywords = FISCAL_KEYWORDS
    elif sentiment_type == 'data_driven':
        title_template = random.choice([t for t in NEWS_TITLE_TEMPLATES if '{pattern}' in t or '技术分析' in t])
        content_template = random.choice(CONTENT_TEMPLATES['data_driven'])
        keywords = DATA_DRIVEN_KEYWORDS
    else:  # opinion
        title_template = random.choice([t for t in NEWS_TITLE_TEMPLATES if '{opinion}' in t or '{sentiment}' in t])
        content_template = random.choice(CONTENT_TEMPLATES['opinion'])
        keywords = OPINION_KEYWORDS
    
    # 填充标题模板
    title_params = {
        'company': company,
        'direction': random.choice(['上涨', '下跌', '波动', '稳定']),
        'quarter': random.choice(['第一季度', '第二季度', '第三季度', '第四季度']),
        'result': random.choice(['超出', '符合', '低于']),
        'opinion': random.choice(['看好', '担忧', '中立于']),
        'indicator': random.choice(['GDP', '通胀', '就业率', '消费者信心']),
        'announcement': random.choice(['新产品', '重组计划', '收购', '裁员']),
        'pattern': random.choice(['形成头肩顶', '突破阻力位', '跌破支撑位', '形成双底']),
        'sentiment': random.choice(['乐观', '悲观', '谨慎', '混合']),
        'product': random.choice(['新产品', '核心业务', '服务', '国际业务']),
        'performance': random.choice(['强劲', '疲软', '符合预期', '出人意料'])
    }
    
    title = title_template.format(**{k: v for k, v in title_params.items() if '{'+k+'}' in title_template})
    
    # 填充内容模板
    content_params = {
        'company': company,
        'quarter': random.choice(['第一季度', '第二季度', '第三季度', '第四季度']),
        'eps': round(random.uniform(0.5, 3.0), 2),
        'expected_eps': round(random.uniform(0.5, 3.0), 2),
        'revenue': round(random.uniform(5, 50), 1),
        'expected_revenue': round(random.uniform(5, 50), 1),
        'result': random.choice(['超出', '符合', '低于']),
        'direction': random.choice(['上涨', '下跌']),
        'reason': random.choice(['强劲的产品需求', '成本上升', '市场竞争加剧', '新市场扩张']),
        'guidance': random.choice(['乐观', '谨慎', '保守']),
        'next_revenue': round(random.uniform(5, 50), 1),
        'inflation_direction': random.choice(['上升', '下降', '保持不变']),
        'inflation_rate': round(random.uniform(1, 8), 1),
        'unemployment_direction': random.choice(['上升', '下降', '保持不变']),
        'unemployment_rate': round(random.uniform(3, 10), 1),
        'impact': random.choice(['积极影响', '负面影响', '几乎不影响']),
        'fed_action': random.choice(['加息', '降息', '维持利率不变']),
        'market_impact': random.choice(['提振', '打压', '稳定']),
        'dividend_action': random.choice(['提高', '维持', '降低']),
        'dividend': round(random.uniform(0.1, 2.0), 2),
        'dividend_impact': random.choice(['增强了', '维持了', '削弱了']),
        'buyback': round(random.uniform(1, 20), 1),
        'management_confidence': random.choice(['对', '不确定']),
        'pattern': random.choice(['头肩顶', '双底', '三角形', '旗形']),
        'pattern_indication': random.choice(['看涨', '看跌', '中性']),
        'rsi': round(random.uniform(20, 80), 1),
        'rsi_indication': random.choice(['超买', '超卖', '中性']),
        'ma_relation': random.choice(['金叉', '死叉', '平行运行']),
        'ma_indication': random.choice(['上升', '下降', '盘整']),
        'volume_trend': random.choice(['增加', '减少', '保持稳定']),
        'volume_confirmation': random.choice(['支持', '反驳', '中和']),
        'valuation': random.choice(['被低估', '被高估', '公平定价']),
        'metrics': random.choice(['市盈率、市净率、自由现金流', '收入增长、利润率、资本回报率', 'EBITDA、债务比率、股息收益率']),
        'fair_value': round(random.uniform(50, 500), 2),
        'price_relation': random.choice(['高出', '低于', '接近']),
        'timeframe': random.choice(['3个月', '6个月', '1年']),
        'price_action': random.choice(['回归均值', '继续偏离', '保持稳定']),
        'indicator': random.choice(['相对强弱指标', '移动平均线', '布林带', 'MACD']),
        'period': random.choice(['3个月', '6个月', '1年']),
        'performance': random.choice(['上涨15%', '下跌10%', '波动不超过5%']),
        'sharpe': round(random.uniform(0.5, 2.5), 2),
        'drawdown': round(random.uniform(5, 30), 1),
        'rating': random.choice(['买入', '持有', '卖出']),
        'belief': random.choice(['相信', '怀疑', '确信']),
        'quality': random.choice(['优质', '一般', '有风险']),
        'market_sentiment': random.choice(['恐慌', '贪婪', '谨慎', '乐观']),
        'confidence': random.choice(['有信心', '不确定', '担心']),
        'performance': random.choice(['表现优异', '保持稳定', '面临挑战']),
        'other_investors_action': random.choice(['恐慌抛售', '盲目追高', '持币观望']),
        'opportunity_type': random.choice(['买入机会', '观望时机', '减持时刻']),
        'market_mood': random.choice(['恐慌', '贪婪', '谨慎', '乐观']),
        'stock_reaction': random.choice(['被低估', '被高估', '波动加剧']),
        'feeling': random.choice(['认为', '感觉', '相信']),
        'reaction_quality': random.choice(['过度', '合理', '不足']),
        'expectation': random.choice(['预期', '希望', '担心']),
        'long_term_performance': random.choice(['表现优异', '保持稳定', '面临挑战']),
        'investor_type': random.choice(['价值', '成长', '收入', '动量']),
        'opinion': random.choice(['信任', '质疑', '欣赏']),
        'decision_quality': random.choice(['明智', '有风险', '有前瞻性']),
        'confidence_level': random.choice(['充满信心', '谨慎乐观', '有所担忧']),
        'short_term_challenges': random.choice(['波动', '竞争压力', '成本上升']),
        'long_term_outlook': random.choice(['光明', '不确定', '充满挑战'])
    }
    
    content = content_template.format(**{k: v for k, v in content_params.items() if '{'+k+'}' in content_template})
    
    # 增加关键词密度
    extra_keywords = ' '.join(random.sample(keywords, min(5, len(keywords))))
    content += f" {extra_keywords}"
    
    return title, content

def generate_news_data(date_range, num_per_day=5):
    """
    生成模拟新闻数据
    
    Args:
        date_range: 日期范围
        num_per_day: 每天的新闻数量
        
    Returns:
        新闻数据DataFrame
    """
    news_data = []
    
    for date in date_range:
        # 每天生成多条新闻
        for _ in range(num_per_day):
            # 随机选择情感类型
            sentiment_type = random.choice(['fiscal', 'data_driven', 'opinion'])
            
            # 生成标题和内容
            title, content = generate_text_with_sentiment(sentiment_type)
            
            # 添加时间
            hour = random.randint(8, 20)
            minute = random.randint(0, 59)
            datetime_str = f"{date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:00"
            
            news_data.append({
                'datetime': datetime_str,
                'title': title,
                'content': content,
                'sentiment_type': sentiment_type  # 添加真实情感类型标签，用于验证
            })
    
    return pd.DataFrame(news_data)

def generate_reddit_data(date_range, num_per_day=8):
    """
    生成模拟Reddit数据
    
    Args:
        date_range: 日期范围
        num_per_day: 每天的Reddit帖子数量
        
    Returns:
        Reddit数据DataFrame
    """
    reddit_data = []
    
    for date in date_range:
        # 每天生成多条Reddit帖子
        for _ in range(num_per_day):
            # 随机选择情感类型，Reddit上观点类型更多
            weights = [0.2, 0.2, 0.6]  # fiscal, data_driven, opinion的权重
            sentiment_type = random.choices(['fiscal', 'data_driven', 'opinion'], weights=weights)[0]
            
            # 生成标题和内容
            if sentiment_type == 'opinion':
                # 使用Reddit特定的标题模板
                title_template = random.choice(REDDIT_TITLE_TEMPLATES)
                company = random.choice(COMPANIES)
                
                title_params = {
                    'company': company,
                    'opinion': random.choice(['看好', '担忧', '中立于', '好奇']),
                    'sentiment': random.choice(['感觉不错', '有点担心', '很兴奋', '很困惑']),
                    'quarter': random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
                    'pattern': random.choice(['头肩顶', '双底', '三角形', '突破']),
                    'direction': random.choice(['上涨', '下跌', '横盘']),
                    'timeframe': random.choice(['短期', '中期', '长期']),
                }
                
                title = title_template.format(**{k: v for k, v in title_params.items() if '{'+k+'}' in title_template})
                _, content = generate_text_with_sentiment(sentiment_type)
            else:
                title, content = generate_text_with_sentiment(sentiment_type)
            
            # 添加时间
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            datetime_str = f"{date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:00"
            
            reddit_data.append({
                'datetime': datetime_str,
                'title': title,
                'content': content,
                'sentiment_type': sentiment_type  # 添加真实情感类型标签，用于验证
            })
    
    return pd.DataFrame(reddit_data)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成测试数据')
    
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)，默认为当前日期')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径，默认使用配置文件中的路径')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else PATHS['data']['raw']
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成日期范围
    date_range = generate_date_range(args.start_date, args.end_date)
    print(f"生成 {len(date_range)} 天的测试数据，从 {date_range[0].strftime('%Y-%m-%d')} 到 {date_range[-1].strftime('%Y-%m-%d')}")
    
    # 生成股票数据
    stock_data = generate_stock_data(date_range)
    stock_path = os.path.join(output_dir, 'stock_data.csv')
    stock_data.to_csv(stock_path, index=False)
    print(f"股票数据已保存至: {stock_path}")
    
    # 生成新闻数据
    news_data = generate_news_data(date_range)
    news_path = os.path.join(output_dir, 'news_data.csv')
    news_data.to_csv(news_path, index=False)
    print(f"新闻数据已保存至: {news_path}")
    
    # 生成Reddit数据
    reddit_data = generate_reddit_data(date_range)
    reddit_path = os.path.join(output_dir, 'reddit_data.csv')
    reddit_data.to_csv(reddit_path, index=False)
    print(f"Reddit数据已保存至: {reddit_path}")
    
    print("\n数据生成完成！可以使用以下命令运行分析:")
    print(f"python src/main.py --news_path {news_path} --reddit_path {reddit_path} --stock_path {stock_path}")

if __name__ == "__main__":
    main() 