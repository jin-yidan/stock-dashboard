/**
 * New Features: Market Status, Data Freshness, Candlestick Patterns, Historical Accuracy
 */

// Load Market Status
function loadMarketStatus() {
    fetch('/api/market/status')
        .then(r => r.json())
        .then(data => {
            const el = document.getElementById('market-status');
            if (!el) return;

            const statusClass = data.is_open ? 'open' : 'closed';
            el.className = 'market-badge ' + statusClass;
            el.textContent = data.status_cn || data.status;

            // Update Beijing time box
            const timeEl = document.getElementById('beijing-time');
            if (timeEl && data.beijing_time) {
                timeEl.textContent = '北京时间 ' + data.beijing_time;
            }
        })
        .catch(err => console.log('Market status error:', err));
}

// Load Data Freshness
function loadDataFreshness() {
    const stockCode = window.stockCode;
    if (!stockCode) return;

    fetch('/api/stock/' + stockCode + '/data-freshness')
        .then(r => r.json())
        .then(data => {
            const el = document.getElementById('data-freshness');
            if (!el) return;

            if (data.error) {
                el.style.display = 'none';
                return;
            }

            const statusClass = data.fresh ? 'fresh' : 'stale';
            el.className = 'freshness-badge ' + statusClass;
            const label = data.status_cn || data.status || '数据';
            const age = data.age ? ` · ${data.age}` : '';
            el.textContent = label + age;
            const lastData = data.last_data_date ? `数据日期: ${data.last_data_date}` : '';
            const lastTrade = data.last_trading_date ? `最近交易日: ${data.last_trading_date}` : '';
            el.title = [lastData, lastTrade].filter(Boolean).join('，');
        })
        .catch(err => console.log('Data freshness error:', err));
}

// Load Candlestick Patterns
function loadCandlestickPatterns() {
    const stockCode = window.stockCode;
    if (!stockCode) return;

    fetch('/api/stock/' + stockCode + '/patterns')
        .then(r => r.json())
        .then(data => {
            const section = document.getElementById('patterns-section');
            const content = document.getElementById('patterns-content');
            const signal = document.getElementById('patterns-signal');

            if (!section || !content) return;

            if (data.error || !data.patterns || data.patterns.length === 0) {
                content.innerHTML = '<span class="text-secondary small">暂无明显K线形态</span>';
                section.style.display = 'block';
                return;
            }

            // Show signal
            if (signal) {
                const signalClass = data.signal === 'bullish' ? 'bullish' :
                                   (data.signal === 'bearish' ? 'bearish' : 'neutral');
                signal.className = 'pattern-signal ' + signalClass;
                signal.textContent = data.signal === 'bullish' ? '看涨' :
                                    (data.signal === 'bearish' ? '看跌' : '中性');
            }

            // Show patterns
            let html = '<div class="pattern-list">';
            data.patterns.forEach(p => {
                const typeClass = p.type === 'bullish' ? 'bullish' :
                                 (p.type === 'bearish' ? 'bearish' : 'reversal');
                html += `<span class="pattern-tag ${typeClass}">${p.name}</span>`;
            });
            html += '</div>';

            content.innerHTML = html;
            section.style.display = 'block';
        })
        .catch(err => console.log('Patterns error:', err));
}

// Load Historical Accuracy
function loadHistoricalAccuracy() {
    const stockCode = window.stockCode;
    if (!stockCode) return;

    fetch('/api/stock/' + stockCode + '/historical-accuracy')
        .then(r => r.json())
        .then(data => {
            const section = document.getElementById('accuracy-section');
            const valueEl = document.getElementById('accuracy-value');
            const signalsEl = document.getElementById('accuracy-signals');
            const periodEl = document.getElementById('accuracy-period');
            const buyEl = document.getElementById('accuracy-buy');
            const sellEl = document.getElementById('accuracy-sell');
            const expectedEl = document.getElementById('accuracy-expected');

            if (!section || !valueEl) return;

            if (data.error || data.total_signals === 0) {
                // No signals generated, hide section or show message
                if (data.total_signals === 0) {
                    valueEl.textContent = '--';
                    valueEl.className = 'accuracy-value';
                    if (signalsEl) signalsEl.textContent = '信号不足';
                    section.style.display = 'block';
                } else {
                    section.style.display = 'none';
                }
                return;
            }

            const accuracy = data.accuracy || 0;
            const reliabilityClass = accuracy > 55 ? 'high' : (accuracy > 45 ? 'moderate' : 'low');

            valueEl.textContent = accuracy.toFixed(1) + '%';
            valueEl.className = 'accuracy-value ' + reliabilityClass;

            if (signalsEl) {
                const total = data.total_signals || 0;
                const buyCount = data.buy_signals || 0;
                const sellCount = data.sell_signals || 0;
                if (buyCount > 0 && sellCount > 0) {
                    signalsEl.textContent = `${total} 个 (买入${buyCount} / 卖出${sellCount})`;
                } else if (buyCount > 0) {
                    signalsEl.textContent = `${total} 个买入信号`;
                } else if (sellCount > 0) {
                    signalsEl.textContent = `${total} 个卖出信号`;
                } else {
                    signalsEl.textContent = `${total} 个`;
                }
            }
            if (periodEl && data.period) {
                // Make period more descriptive
                const periodText = data.period.replace(/(\d+)天/, '$1 天');
                periodEl.textContent = periodText;
            }
            if (buyEl) {
                const buyAcc = data.buy_accuracy || 0;
                buyEl.textContent = buyAcc.toFixed(1) + '%';
            }
            if (sellEl) {
                const sellAcc = data.sell_accuracy || 0;
                sellEl.textContent = sellAcc.toFixed(1) + '%';
            }
            if (expectedEl) {
                const exp = data.expected_return || 0;
                const prefix = exp >= 0 ? '+' : '';
                expectedEl.textContent = `${prefix}${exp.toFixed(2)}%`;
                expectedEl.className = 'accuracy-detail-value ' + (exp >= 0 ? 'up' : 'down');
            }

            section.style.display = 'block';
        })
        .catch(err => console.log('Accuracy error:', err));
}

// Auto-initialize when DOM is ready
// Prioritize: fast/important loads first, heavy loads deferred
function initNewFeatures() {
    // Priority 1: Fast, lightweight (load immediately)
    loadMarketStatus();
    loadDataFreshness();

    // Priority 2: Medium weight (defer 500ms to let critical content load first)
    setTimeout(function() {
        loadCandlestickPatterns();
    }, 500);

    // Priority 3: Heavy computation (defer 2s, runs 60-day backtest)
    setTimeout(function() {
        loadHistoricalAccuracy();
    }, 2000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNewFeatures);
} else {
    setTimeout(initNewFeatures, 100);
}
