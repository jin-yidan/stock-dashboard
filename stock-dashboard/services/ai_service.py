"""
AI Analysis Service - Claude-powered stock analysis.
"""

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def build_analysis_prompt(stock_code, stock_name, signal_data, cyq_data, strategies_data, week52_data):
    """Build the prompt with all our technical data."""

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

    prompt = f"""你是一位私募基金经理，正在给客户做个股分析。客户是有一定经验的散户，需要专业但能听懂的建议。

股票：{stock_name} ({stock_code})

当前技术面数据：
- 综合信号：{signal_data.get('signal_cn', 'N/A')}，得分 {signal_data.get('score', 0):.2f}（满分1分），置信度 {signal_data.get('confidence', 0)}%
- 市场状态：{signal_data.get('market_regime', 'unknown')}
- 52周位置：{week52_text}

主要指标：
{indicators_text}

筹码分布：
{cyq_text}

触发的策略信号：
{strategies_text}

---

请给出分析报告。要求：

1. 结论先行
开头第一句话直接给结论：建议买入/卖出/持有/观望，以及核心理由（一句话）。

2. 关键判断依据
从上面的数据中，挑出2-3个最重要的指标，解释它们说明了什么问题。不要罗列所有指标，只说最关键的。比如："MACD刚形成金叉，但成交量没有放大配合，说明上涨动能不足"。

3. 具体操作建议
- 如果建议买入：在什么价位买？分几次建仓？
- 如果建议观望：等待什么信号出现再考虑？
- 止损设在哪里？为什么设在这个位置？

4. 主要风险
指出1-2个最需要警惕的风险，要具体。比如"下周五公布财报，如果业绩不及预期可能补跌"，而不是"注意业绩风险"。

格式要求：
- 用自然的段落，不要用太多列表
- 不要用"首先、其次、最后"这种八股文结构
- 像在微信上给朋友讲解一样，专业但不生硬
- 控制在250-350字
"""

    return prompt


def generate_ai_analysis(api_key, stock_code, stock_name, signal_data, cyq_data, strategies_data, week52_data):
    """
    Generate AI analysis using Claude API.

    Args:
        api_key: User's Anthropic API key (sk-ant-...)
        stock_code: Stock code
        stock_name: Stock name
        signal_data: Output from signal_service.generate_signal()
        cyq_data: Output from cyq_service.get_cyq_analysis()
        strategies_data: Output from strategy_service.run_all_strategies()
        week52_data: 52-week high/low data

    Returns:
        dict with 'success', 'analysis' or 'error'
    """
    if not HAS_ANTHROPIC:
        return {'success': False, 'error': '服务器未安装 anthropic 库'}

    if not api_key:
        return {'success': False, 'error': '请输入 API Key'}

    prompt = build_analysis_prompt(
        stock_code, stock_name, signal_data,
        cyq_data, strategies_data, week52_data
    )

    system_prompt = "你是一位经验丰富的A股分析师，擅长结合技术分析和基本面分析给出投资建议。"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        analysis = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        return {
            'success': True,
            'analysis': analysis,
            'tokens_used': tokens_used,
            'model': 'claude-sonnet-4'
        }

    except anthropic.AuthenticationError:
        return {'success': False, 'error': 'API Key 无效，请检查后重试'}
    except anthropic.RateLimitError:
        return {'success': False, 'error': 'API 请求频率超限，请稍后重试'}
    except anthropic.APIStatusError as e:
        # Show actual error for debugging
        return {'success': False, 'error': f'API 错误: {e.message}'}
    except Exception as e:
        return {'success': False, 'error': f'分析失败: {str(e)}'}
