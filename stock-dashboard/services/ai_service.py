"""
AI Analysis Service - GPT-powered stock analysis with web search.

Uses user's OpenAI API key to generate comprehensive analysis
combining our technical data with recent news and policy context.
"""

from openai import OpenAI


def build_analysis_prompt(stock_code, stock_name, signal_data, cyq_data, strategies_data, week52_data):
    """
    Build the prompt with all our technical data.

    The prompt asks GPT to:
    1. Analyze the technical data we provide
    2. Search for recent news about the stock
    3. Consider policy/sector trends
    4. Give comprehensive advice
    """

    # Format indicators table
    indicators = signal_data.get('indicators', [])
    indicators_sorted = sorted(indicators, key=lambda x: abs(x.get('weighted_score', 0)), reverse=True)

    indicators_text = ""
    for ind in indicators_sorted[:10]:  # Top 10 contributors
        score = ind.get('weighted_score', 0)
        sign = '+' if score > 0 else ''
        indicators_text += f"- {ind['name']}: {ind['signal']} ({sign}{score:.3f}) - {ind.get('plain_explanation', '')}\n"

    # Format CYQ data
    cyq_text = "数据不可用"
    if cyq_data and cyq_data.get('available'):
        sig = cyq_data.get('signal', {})
        details = sig.get('details', {})
        cyq_text = f"""- 获利盘比例: {details.get('benefit_part', 'N/A')}%
- 平均成本: {details.get('avg_cost', 'N/A')}
- 当前价格: {details.get('current_price', 'N/A')}"""

    # Format strategies
    strategies_text = "无策略触发"
    triggered = strategies_data.get('triggered', [])
    if triggered:
        strategies_text = ""
        for s in triggered:
            strategies_text += f"- {s.get('name_cn', s.get('strategy'))}\n"
            details = s.get('details', {})
            for k, v in list(details.items())[:3]:
                strategies_text += f"  {k}: {v}\n"

    # Format 52-week data
    week52_text = "数据不可用"
    if week52_data and not week52_data.get('error'):
        week52_text = f"""- 当前价: {week52_data.get('current')}
- 52周最高: {week52_data.get('high_52w')} ({week52_data.get('pct_from_high')}%)
- 52周最低: {week52_data.get('low_52w')} (+{week52_data.get('pct_from_low')}%)
- 位置: 52周区间的 {week52_data.get('position')}%"""

    # Risk management
    stop_loss = signal_data.get('stop_loss')
    take_profit = signal_data.get('take_profit')
    risk_text = f"- 建议止损: {stop_loss}\n- 建议止盈: {take_profit}" if stop_loss else "暂无建议"

    prompt = f"""你是一位专业的A股分析师。请对股票 **{stock_name} ({stock_code})** 进行全面分析。

## 第一部分：技术分析数据（已通过量化模型计算）

### 综合信号
- 信号: {signal_data.get('signal_cn', 'N/A')}
- 得分: {signal_data.get('score', 0):.3f} (范围 -1 到 +1)
- 置信度: {signal_data.get('confidence', 0)}%
- 市场环境: {signal_data.get('market_regime', 'unknown')}

### 指标明细（按贡献度排序）
{indicators_text}

### 筹码分布 (CYQ)
{cyq_text}

### 触发的交易策略
{strategies_text}

### 52周价格位置
{week52_text}

### 风险管理
{risk_text}

---

## 第二部分：请你搜索补充以下信息

1. **个股新闻**: {stock_name}最近7天的重要新闻（业绩、公告、重大事项）
2. **行业动态**: 该股票所属行业的近期政策或趋势变化
3. **资金动向**: 近期是否有北向资金、主力资金的明显流入流出
4. **机构观点**: 近期券商研报评级变化（如有）

---

## 第三部分：请给出综合分析报告

### 1. 一句话结论
（明确看多/看空/观望，20字以内）

### 2. 技术面分析
（基于上述量化数据，指出2-3个关键点）

### 3. 消息面分析
（基于你搜索到的新闻，指出重要影响因素）

### 4. 操作建议
- 仓位建议：轻仓/半仓/重仓
- 买入时机：具体价位或条件
- 止损位：{stop_loss if stop_loss else '建议设置'}
- 止盈位：{take_profit if take_profit else '建议设置'}

### 5. 风险提示
（列出2-3个需要警惕的风险点）

---

要求：
- 保持客观专业，不要过度乐观或悲观
- 总字数控制在400-500字
- 不要编造数据，如果搜索不到相关新闻请说明
- 给出的建议要具体可执行
"""

    return prompt


def generate_ai_analysis(api_key, stock_code, stock_name, signal_data, cyq_data, strategies_data, week52_data):
    """
    Generate AI analysis using OpenAI API.

    Args:
        api_key: User's OpenAI API key
        stock_code: Stock code
        stock_name: Stock name
        signal_data: Output from signal_service.generate_signal()
        cyq_data: Output from cyq_service.get_cyq_analysis()
        strategies_data: Output from strategy_service.run_all_strategies()
        week52_data: 52-week high/low data

    Returns:
        dict with 'success', 'analysis' or 'error'
    """
    try:
        client = OpenAI(api_key=api_key)

        prompt = build_analysis_prompt(
            stock_code, stock_name, signal_data,
            cyq_data, strategies_data, week52_data
        )

        # Use GPT-4o for best quality with web browsing capability
        # Fall back to gpt-4o-mini if needed (cheaper but still good)
        response = client.chat.completions.create(
            model="gpt-4o",  # Has web browsing capability
            messages=[
                {
                    "role": "system",
                    "content": "你是一位经验丰富的A股分析师，擅长结合技术分析和基本面分析给出投资建议。请使用搜索功能获取最新的股票新闻和市场信息。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        analysis = response.choices[0].message.content

        # Get token usage for cost estimation
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0

        return {
            'success': True,
            'analysis': analysis,
            'tokens_used': tokens_used,
            'model': 'gpt-4o'
        }

    except Exception as e:
        error_msg = str(e)

        # Provide helpful error messages
        if 'invalid_api_key' in error_msg.lower() or 'incorrect api key' in error_msg.lower():
            return {'success': False, 'error': 'API Key 无效，请检查后重试'}
        elif 'rate_limit' in error_msg.lower():
            return {'success': False, 'error': 'API 请求频率超限，请稍后重试'}
        elif 'insufficient_quota' in error_msg.lower():
            return {'success': False, 'error': 'API 余额不足，请充值后重试'}
        else:
            return {'success': False, 'error': f'分析失败: {error_msg}'}
