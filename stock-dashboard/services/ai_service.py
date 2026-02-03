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

    prompt = f"""你是一位专业的A股分析师。请对股票 **{stock_name} ({stock_code})** 进行全面分析。

## 技术分析数据（已通过量化模型计算）

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

## 请给出综合分析报告

### 1. 一句话结论
（明确看多/看空/观望，20字以内）

### 2. 技术面分析
（基于上述量化数据，指出2-3个关键点）

### 3. 操作建议
- 仓位建议：轻仓/半仓/重仓
- 买入时机：具体价位或条件
- 止损位：{stop_loss if stop_loss else '建议设置'}
- 止盈位：{take_profit if take_profit else '建议设置'}

### 4. 风险提示
（列出2-3个需要警惕的风险点）

---

要求：
- 保持客观专业，不要过度乐观或悲观
- 总字数控制在300-400字
- 给出的建议要具体可执行
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

    if not api_key or not api_key.startswith('sk-ant-'):
        return {'success': False, 'error': 'API Key 格式错误，应以 sk-ant- 开头'}

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

    except Exception as e:
        error_msg = str(e)

        if 'invalid_api_key' in error_msg.lower() or 'authentication' in error_msg.lower():
            return {'success': False, 'error': 'API Key 无效，请检查后重试'}
        elif 'rate_limit' in error_msg.lower():
            return {'success': False, 'error': 'API 请求频率超限，请稍后重试'}
        elif 'credit' in error_msg.lower() or 'billing' in error_msg.lower():
            return {'success': False, 'error': 'API 余额不足，请充值后重试'}
        else:
            return {'success': False, 'error': f'分析失败: {error_msg}'}
