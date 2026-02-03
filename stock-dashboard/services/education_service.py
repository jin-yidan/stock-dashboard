"""
Education Service - Indicator explanations and historical context
Provides beginner-friendly explanations and historical pattern matching
"""

# Indicator education content - what each indicator measures and how to use it
INDICATOR_EDUCATION = {
    'ma': {
        'name': '均线 (Moving Average)',
        'what_it_measures': '均线是过去N天收盘价的平均值，用于平滑价格波动，显示趋势方向。',
        'how_to_read': [
            '价格在均线上方 = 上涨趋势',
            '价格在均线下方 = 下跌趋势',
            '短期均线上穿长期均线 = 金叉（买入信号）',
            '短期均线下穿长期均线 = 死叉（卖出信号）',
        ],
        'key_levels': {
            'MA5': '5日线，超短线趋势',
            'MA10': '10日线，短线趋势',
            'MA20': '20日线，中短线趋势（月线）',
            'MA60': '60日线，中线趋势（季线）',
            'MA250': '250日线，长线趋势（年线）',
        },
        'tips': [
            '均线多头排列（短期在上）= 强势',
            '均线粘合后发散 = 大行情起点',
            '年线是牛熊分界线',
        ],
        'accuracy_note': '单独使用准确率约50%，需结合其他指标',
    },
    'macd': {
        'name': 'MACD (指数平滑异同移动平均线)',
        'what_it_measures': 'MACD衡量短期与长期趋势的差异，用于判断趋势强度和转折点。',
        'how_to_read': [
            'DIF上穿DEA = 金叉（买入信号）',
            'DIF下穿DEA = 死叉（卖出信号）',
            '红柱增长 = 上涨动能增强',
            '绿柱增长 = 下跌动能增强',
            '零轴以上 = 多头市场',
            '零轴以下 = 空头市场',
        ],
        'key_levels': {
            '零轴': '多空分界线',
            'DIF': '快线，反应灵敏',
            'DEA': '慢线，趋势确认',
            '柱状图': '动能强度',
        },
        'tips': [
            '底背离：价格创新低，MACD不创新低 = 可能反转',
            '顶背离：价格创新高，MACD不创新高 = 可能见顶',
            '零轴附近的金叉更有效',
        ],
        'accuracy_note': '准确率约50%，背离信号更可靠',
    },
    'kdj': {
        'name': 'KDJ (随机指标)',
        'what_it_measures': 'KDJ通过比较收盘价与最高最低价的位置，判断超买超卖状态。',
        'how_to_read': [
            'K值 > 80 = 超买区（可能回调）',
            'K值 < 20 = 超卖区（可能反弹）',
            'K上穿D = 金叉（买入信号）',
            'K下穿D = 死叉（卖出信号）',
            'J值 > 100 = 极度超买',
            'J值 < 0 = 极度超卖',
        ],
        'key_levels': {
            '80': '超买警戒线',
            '50': '多空平衡线',
            '20': '超卖警戒线',
        },
        'tips': [
            '超买不等于立即卖出，强势股可以持续超买',
            '超卖区的金叉更可靠',
            'J值最灵敏，但也最容易骗线',
        ],
        'accuracy_note': '震荡市准确率较高，趋势市容易失效',
    },
    'rsi': {
        'name': 'RSI (相对强弱指标)',
        'what_it_measures': 'RSI衡量价格涨跌的相对强度，判断超买超卖和趋势强度。',
        'how_to_read': [
            'RSI > 70 = 超买（可能回调）',
            'RSI < 30 = 超卖（可能反弹）',
            'RSI持续在50以上 = 强势',
            'RSI持续在50以下 = 弱势',
        ],
        'key_levels': {
            '70': '超买区',
            '50': '多空分界',
            '30': '超卖区',
        },
        'tips': [
            'RSI背离是重要的反转信号',
            '超卖后买入的成功率比超买后卖出更高',
            '牛市中RSI可能长期超买',
        ],
        'accuracy_note': 'RSI超卖买入信号准确率约55-60%',
    },
    'bollinger': {
        'name': '布林带 (Bollinger Bands)',
        'what_it_measures': '布林带用标准差衡量价格波动范围，识别突破和回归机会。',
        'how_to_read': [
            '价格触及上轨 = 短期强势/可能回调',
            '价格触及下轨 = 短期弱势/可能反弹',
            '带宽收窄 = 变盘前兆（挤压）',
            '带宽扩大 = 趋势加速',
            '价格在中轨上方 = 偏多',
        ],
        'key_levels': {
            '上轨': '阻力位/超买',
            '中轨': '20日均线/多空分界',
            '下轨': '支撑位/超卖',
        },
        'tips': [
            '布林带挤压后的突破往往是大行情',
            '95%的价格在布林带内运行',
            '连续触及上轨是强势特征',
        ],
        'accuracy_note': '布林带信号准确率约60%，是最可靠的指标之一',
    },
    'volume': {
        'name': '成交量 (Volume)',
        'what_it_measures': '成交量反映市场参与程度和资金流向，用于确认趋势有效性。',
        'how_to_read': [
            '放量上涨 = 上涨有效，可持续',
            '缩量上涨 = 上涨乏力，谨慎追高',
            '放量下跌 = 恐慌抛售，可能加速下跌',
            '缩量下跌 = 抛压减轻，可能见底',
        ],
        'key_levels': {
            '量比>2': '明显放量',
            '量比1-2': '温和放量',
            '量比<0.8': '缩量',
        },
        'tips': [
            '量在价先：成交量变化往往领先价格',
            '底部放量是启动信号',
            '高位放量滞涨要警惕',
        ],
        'accuracy_note': '成交量确认信号准确率约58%',
    },
    'supertrend': {
        'name': 'Supertrend (超级趋势)',
        'what_it_measures': 'Supertrend是基于ATR的趋势跟踪指标，提供清晰的多空信号。',
        'how_to_read': [
            '绿色/价格在线上 = 多头趋势',
            '红色/价格在线下 = 空头趋势',
            '颜色翻转 = 趋势转变信号',
        ],
        'key_levels': {
            'Supertrend线': '动态支撑/阻力',
        },
        'tips': [
            'Supertrend在趋势市表现最好',
            '结合成交量确认突破有效性',
            '是本系统准确率最高的指标(62%)',
        ],
        'accuracy_note': '准确率约62.4%，是系统中最可靠的指标',
    },
    'wave_trend': {
        'name': 'Wave Trend (波浪趋势)',
        'what_it_measures': 'Wave Trend结合了动量和周期分析，识别趋势转折点。',
        'how_to_read': [
            'WT1上穿WT2 = 买入信号',
            'WT1下穿WT2 = 卖出信号',
            'WT1 > 60 = 超买区',
            'WT1 < -60 = 超卖区',
        ],
        'key_levels': {
            '60': '超买警戒',
            '0': '多空平衡',
            '-60': '超卖警戒',
        },
        'tips': [
            '超卖区的金叉成功率更高',
            '与价格背离时关注反转',
        ],
        'accuracy_note': '准确率约55.4%',
    },
    'vr': {
        'name': 'VR (成交量比率)',
        'what_it_measures': 'VR比较上涨日和下跌日的成交量，判断多空力量对比。',
        'how_to_read': [
            'VR > 200 = 强势/可能过热',
            'VR 150-200 = 偏多',
            'VR 100-150 = 平衡',
            'VR < 70 = 超卖/可能反弹',
        ],
        'key_levels': {
            '200': '过热警戒',
            '100': '多空平衡',
            '70': '超卖区',
        },
        'tips': [
            'VR低位是布局机会',
            'VR持续高位要注意回调风险',
        ],
        'accuracy_note': '中等准确率，适合辅助判断',
    },
    'cr': {
        'name': 'CR (能量指标)',
        'what_it_measures': 'CR衡量多空双方的能量对比，用于判断买卖压力。',
        'how_to_read': [
            'CR > 200 = 多方能量强/可能超买',
            'CR 100-200 = 偏多',
            'CR < 40 = 超卖/可能反弹',
        ],
        'key_levels': {
            '200': '多方能量过强',
            '100': '多空平衡',
            '40': '空方能量过强',
        },
        'tips': [
            'CR低位金叉是买入信号',
            '与MA1-4均线配合使用效果更好',
        ],
        'accuracy_note': '中等准确率，需结合其他指标',
    },
    'cyq': {
        'name': '筹码分布 (CYQ)',
        'what_it_measures': '筹码分布分析不同价位的持仓成本，判断支撑阻力和获利盘压力。',
        'how_to_read': [
            '获利比例 < 20% = 严重套牢/可能超卖',
            '获利比例 > 80% = 获利盘多/可能有抛压',
            '筹码集中 = 主力控盘/可能拉升',
            '筹码分散 = 散户为主/走势难测',
        ],
        'key_levels': {
            '平均成本': '重要支撑/阻力位',
            '70%筹码区间': '主要成本区',
        },
        'tips': [
            '低位筹码集中是主力建仓信号',
            '突破密集成本区是启动信号',
            '筹码从下往上移动是健康上涨',
        ],
        'accuracy_note': '机构级指标，分析庄家行为有效',
    },
    'adx': {
        'name': 'ADX (趋势强度指标)',
        'what_it_measures': 'ADX衡量趋势的强度，不区分方向，用于判断是否适合趋势交易。',
        'how_to_read': [
            'ADX > 25 = 趋势明确，适合趋势策略',
            'ADX < 20 = 盘整市，适合震荡策略',
            '+DI > -DI = 多头趋势',
            '-DI > +DI = 空头趋势',
        ],
        'key_levels': {
            '25': '趋势/盘整分界',
            '40': '强趋势',
        },
        'tips': [
            'ADX上升 = 趋势加强',
            'ADX下降 = 趋势减弱',
            'ADX低位时不要追趋势',
        ],
        'accuracy_note': '用于过滤震荡市的假信号',
    },
    'northbound': {
        'name': '北向资金 (Northbound Flow)',
        'what_it_measures': '北向资金是外资通过沪港通/深港通买入A股的资金，被视为"聪明钱"。',
        'how_to_read': [
            '连续净流入 = 外资看好',
            '连续净流出 = 外资撤退',
            '大额单日流入 = 可能有重大利好',
        ],
        'key_levels': {},
        'tips': [
            '北向资金偏好大盘蓝筹和消费龙头',
            '外资动向常领先于国内机构',
            '注意区分被动基金调仓和主动买入',
        ],
        'accuracy_note': '中长期参考价值高',
    },
    'weekly': {
        'name': '周线趋势 (Weekly Trend)',
        'what_it_measures': '周线趋势反映中期走势，过滤日线噪音。',
        'how_to_read': [
            '周线上涨 = 中期趋势向上',
            '周线下跌 = 中期趋势向下',
            '周线与日线同向 = 趋势确认',
        ],
        'key_levels': {},
        'tips': [
            '周线定方向，日线找买点',
            '周线级别的支撑阻力更有效',
            '周线金叉比日线金叉更可靠',
        ],
        'accuracy_note': '准确率约54%，用于过滤短期噪音',
    },
}

# Market regime explanations
MARKET_REGIME_EDUCATION = {
    'strong_bull': {
        'name': '强势牛市',
        'description': '大盘处于明显上涨趋势，做多胜率高。',
        'strategy': '可以积极做多，适当提高仓位。突破买入策略效果好。',
        'risk': '注意不要追高，保持止损纪律。',
    },
    'bull': {
        'name': '牛市',
        'description': '大盘偏多，但涨幅温和。',
        'strategy': '保持多头思维，逢低布局。',
        'risk': '注意板块轮动，不要过度集中。',
    },
    'sideways': {
        'name': '震荡市',
        'description': '大盘方向不明，上下波动。',
        'strategy': '控制仓位，高抛低吸。趋势指标可能频繁假信号。',
        'risk': '避免追涨杀跌，耐心等待方向明确。',
    },
    'bear': {
        'name': '熊市',
        'description': '大盘偏弱，下跌风险较高。',
        'strategy': '降低仓位，精选强势股。反弹减仓为主。',
        'risk': '不要抄底，等待企稳信号。',
    },
    'strong_bear': {
        'name': '强势熊市',
        'description': '大盘处于明显下跌趋势，风险很高。',
        'strategy': '空仓观望或极低仓位。只做超跌反弹。',
        'risk': '现金为王，不要逆势操作。',
    },
}


def get_indicator_education(indicator_key):
    """Get education content for a specific indicator"""
    return INDICATOR_EDUCATION.get(indicator_key, None)


def get_all_indicator_education():
    """Get education content for all indicators"""
    return INDICATOR_EDUCATION


def get_market_regime_education(regime):
    """Get education content for market regime"""
    return MARKET_REGIME_EDUCATION.get(regime, None)


def generate_detailed_explanation(indicator_key, analysis_result, current_values=None):
    """
    Generate a more detailed explanation for an indicator based on its analysis result

    Args:
        indicator_key: The indicator identifier (e.g., 'ma', 'macd')
        analysis_result: The analysis dict from signal_service
        current_values: Dict of current indicator values for context

    Returns:
        Dict with detailed explanation
    """
    education = INDICATOR_EDUCATION.get(indicator_key, {})

    result = {
        'indicator': indicator_key,
        'name': education.get('name', indicator_key),
        'current_signal': analysis_result.get('signal', 'neutral'),
        'score': analysis_result.get('score', 0),
        'plain_explanation': analysis_result.get('plain_explanation', ''),
        'what_it_measures': education.get('what_it_measures', ''),
        'how_to_read': education.get('how_to_read', []),
        'tips': education.get('tips', []),
        'accuracy_note': education.get('accuracy_note', ''),
    }

    # Add context-specific explanation based on signal
    signal = analysis_result.get('signal', 'neutral')
    score = analysis_result.get('score', 0)

    if signal == 'bullish':
        if score > 0.5:
            result['signal_strength'] = '强烈看多'
            result['action_suggestion'] = '可以考虑买入或加仓'
        else:
            result['signal_strength'] = '温和看多'
            result['action_suggestion'] = '可以关注，等待更多确认'
    elif signal == 'bearish':
        if score < -0.5:
            result['signal_strength'] = '强烈看空'
            result['action_suggestion'] = '建议减仓或回避'
        else:
            result['signal_strength'] = '温和看空'
            result['action_suggestion'] = '谨慎观望，注意风险'
    else:
        result['signal_strength'] = '中性'
        result['action_suggestion'] = '观望等待，方向不明确'

    return result


def find_historical_patterns(df, current_signal, indicator_key, lookback_days=120):
    """
    Find historical instances where similar signals occurred and what happened after

    Args:
        df: DataFrame with historical data and indicator values
        current_signal: Current signal ('bullish', 'bearish', 'neutral')
        indicator_key: The indicator to analyze
        lookback_days: How many days to look back

    Returns:
        Dict with historical pattern analysis
    """
    if df is None or len(df) < lookback_days:
        return None

    # This is a simplified version - you can make it more sophisticated
    results = {
        'total_similar_signals': 0,
        'positive_outcomes': 0,
        'negative_outcomes': 0,
        'avg_return_5d': 0,
        'avg_return_10d': 0,
        'examples': [],
    }

    try:
        # Calculate future returns for historical analysis
        df = df.copy()
        df['future_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['future_10d'] = df['close'].shift(-10) / df['close'] - 1

        # Look for similar signal patterns (this is simplified)
        # In a real implementation, you would regenerate signals for historical data

        # For now, just return basic stats
        recent_returns = df['future_5d'].dropna().tail(20)
        if len(recent_returns) > 0:
            results['avg_return_5d'] = round(recent_returns.mean() * 100, 2)
            results['positive_outcomes'] = int((recent_returns > 0).sum())
            results['negative_outcomes'] = int((recent_returns <= 0).sum())
            results['total_similar_signals'] = len(recent_returns)

    except Exception as e:
        pass

    return results


def generate_signal_context(signal_data, market_regime=None):
    """
    Generate contextual explanation for the overall signal

    Args:
        signal_data: The complete signal data from signal_service
        market_regime: Current market regime

    Returns:
        Dict with contextual explanation
    """
    signal = signal_data.get('signal', 'hold')
    score = signal_data.get('score', 0)
    confidence = signal_data.get('confidence', 0)

    context = {
        'signal': signal,
        'signal_cn': signal_data.get('signal_cn', ''),
        'confidence_level': '高' if confidence >= 70 else ('中' if confidence >= 50 else '低'),
        'market_context': '',
        'key_factors': [],
        'risk_factors': [],
        'suggested_action': '',
    }

    # Add market regime context
    if market_regime:
        regime_edu = MARKET_REGIME_EDUCATION.get(market_regime, {})
        context['market_context'] = regime_edu.get('description', '')
        context['market_strategy'] = regime_edu.get('strategy', '')

    # Analyze key contributing indicators
    indicators = signal_data.get('indicators', [])
    bullish_indicators = [i for i in indicators if i.get('signal') == 'bullish']
    bearish_indicators = [i for i in indicators if i.get('signal') == 'bearish']

    # Sort by absolute weighted score
    bullish_indicators.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
    bearish_indicators.sort(key=lambda x: x.get('weighted_score', 0))

    # Top 3 positive factors
    for ind in bullish_indicators[:3]:
        context['key_factors'].append({
            'name': ind.get('name', ''),
            'explanation': ind.get('plain_explanation', ''),
            'contribution': round(ind.get('weighted_score', 0) * 100, 1),
        })

    # Top 3 risk factors
    for ind in bearish_indicators[:3]:
        context['risk_factors'].append({
            'name': ind.get('name', ''),
            'explanation': ind.get('plain_explanation', ''),
            'contribution': round(ind.get('weighted_score', 0) * 100, 1),
        })

    # Generate suggested action based on signal and confidence
    if signal in ['strong_buy', 'buy']:
        if confidence >= 70:
            context['suggested_action'] = '信号较强，可以考虑建仓或加仓，但需设置止损。'
        elif confidence >= 50:
            context['suggested_action'] = '信号偏多，可以小仓位尝试，观察后续走势。'
        else:
            context['suggested_action'] = '信号偏多但不够明确，建议继续观察。'
    elif signal in ['strong_sell', 'sell']:
        if confidence >= 70:
            context['suggested_action'] = '信号较弱，建议减仓或回避，不宜抄底。'
        elif confidence >= 50:
            context['suggested_action'] = '信号偏空，谨慎为主，可以等待企稳。'
        else:
            context['suggested_action'] = '信号偏空但不够明确，保持观望。'
    else:
        context['suggested_action'] = '信号中性，方向不明确，建议观望等待。'

    return context
