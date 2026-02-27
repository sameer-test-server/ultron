(function () {
    'use strict';

    var toFiniteNumber = function (value, fallback) {
        var numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : fallback;
    };

    var escapeHtml = function (value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    };

    var fmt = function (value, digits) {
        return toFiniteNumber(value, 0).toLocaleString(undefined, {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    };

    var fmtPct = function (value) {
        var numeric = toFiniteNumber(value, 0);
        return (numeric >= 0 ? '+' : '') + fmt(numeric, 2) + '%';
    };

    var fmtInr = function (value) {
        return '₹' + fmt(value, 2);
    };

    var buildUrl = function (base, params) {
        var search = new URLSearchParams(params);
        return base + '?' + search.toString();
    };

    var fetchJson = function (url) {
        return fetch(url, { headers: { Accept: 'application/json' } }).then(function (response) {
            var contentType = response.headers.get('content-type') || '';
            if (contentType.indexOf('application/json') === -1) {
                return response.text().then(function (text) {
                    throw new Error(text ? text.slice(0, 240) : 'Unexpected non-JSON response.');
                });
            }
            return response.json().then(function (payload) {
                if (!response.ok) {
                    throw new Error(payload.error || 'Request failed.');
                }
                return payload;
            });
        });
    };

    var plotWrap = document.querySelector('.plotly-wrapper');
    var chartRecentBtn = document.getElementById('chartRecentBtn');
    var chartFullBtn = document.getElementById('chartFullBtn');
    if (plotWrap && window.Plotly) {
        var plotEl = plotWrap.querySelector('.plotly-graph-div');
        var recentStart = plotWrap.dataset.recentStart;
        var recentEnd = plotWrap.dataset.recentEnd;
        var fullStart = plotWrap.dataset.fullStart;
        var fullEnd = plotWrap.dataset.fullEnd;

        var setActiveChartButton = function (button) {
            if (chartRecentBtn) {
                chartRecentBtn.classList.toggle('active', button === chartRecentBtn);
            }
            if (chartFullBtn) {
                chartFullBtn.classList.toggle('active', button === chartFullBtn);
            }
        };

        var setChartRange = function (startDate, endDate) {
            if (!plotEl || !startDate || !endDate) {
                return;
            }
            window.Plotly.relayout(plotEl, {
                'xaxis.range': [startDate, endDate],
                'xaxis2.range': [startDate, endDate]
            });
        };

        if (chartRecentBtn) {
            chartRecentBtn.addEventListener('click', function () {
                setChartRange(recentStart, recentEnd);
                setActiveChartButton(chartRecentBtn);
            });
        }
        if (chartFullBtn) {
            chartFullBtn.addEventListener('click', function () {
                setChartRange(fullStart, fullEnd);
                setActiveChartButton(chartFullBtn);
            });
        }
    }

    var adaptiveLab = document.querySelector('.adaptive-sim-lab');
    if (!adaptiveLab) {
        return;
    }

    var simUrl = adaptiveLab.dataset.simUrl || '';
    var projectionLab = document.querySelector('.projection-lab');
    var projectionUrl = projectionLab ? (projectionLab.dataset.projectionUrl || '') : '';

    var simCapital = document.getElementById('simCapital');
    var simPositionSize = document.getElementById('simPositionSize');
    var simMaxHoldDays = document.getElementById('simMaxHoldDays');
    var simStopLoss = document.getElementById('simStopLoss');
    var simTakeProfit = document.getElementById('simTakeProfit');
    var simBrokerage = document.getElementById('simBrokerage');
    var simSlippage = document.getElementById('simSlippage');
    var simRunBtn = document.getElementById('simRunBtn');
    var simRunStatus = document.getElementById('simRunStatus');
    var simLiveMetrics = document.getElementById('simLiveMetrics');
    var simTradesBody = document.getElementById('simTradesBody');
    var simTradesMeta = document.getElementById('simTradesMeta');
    var simTradesToggleBtn = document.getElementById('simTradesToggleBtn');

    var projectionAmount = document.getElementById('projectionAmount');
    var projectionRunBtn = document.getElementById('projectionRunBtn');
    var projectionStatus = document.getElementById('projectionStatus');
    var projectionCards = document.getElementById('projectionCards');
    var projectionMethod = document.getElementById('projectionMethod');
    var projectionCustomHorizon = document.getElementById('projectionCustomHorizon');

    var hasSimulationControls = Boolean(
        simUrl &&
        simCapital &&
        simPositionSize &&
        simMaxHoldDays &&
        simStopLoss &&
        simTakeProfit &&
        simBrokerage &&
        simSlippage &&
        simRunBtn &&
        simRunStatus &&
        simLiveMetrics &&
        simTradesBody
    );
    var hasProjectionControls = Boolean(
        projectionUrl &&
        projectionAmount &&
        projectionRunBtn &&
        projectionStatus &&
        projectionCards &&
        projectionCustomHorizon
    );

    var TRADE_PREVIEW_LIMIT = 12;
    var simAllTrades = [];
    var simShowAllTrades = false;

    var renderSimMetrics = function (result) {
        if (!simLiveMetrics) {
            return;
        }
        var cards = [
            ['Initial Capital', fmtInr(result.initial_capital), 'mono'],
            ['Final Capital', fmtInr(result.final_capital), 'mono'],
            ['Total Return', fmtPct(result.total_return_pct), toFiniteNumber(result.total_return_pct, 0) >= 0 ? 'long' : 'short'],
            ['Win Rate', fmt(result.win_rate, 1) + '%', 'mono'],
            ['Trades', String(toFiniteNumber(result.trades_count, 0)), 'mono'],
            ['Max Drawdown', fmt(result.max_drawdown_pct, 2) + '%', 'short'],
            ['Profit Factor', fmt(result.profit_factor, 2), 'mono'],
            ['Avg Trade Return', fmt(result.avg_trade_return_pct, 2) + '%', 'mono'],
            ['Total Fees', fmtInr(result.total_fees), 'mono']
        ];
        simLiveMetrics.innerHTML = cards.map(function (item) {
            return '<article class="mini-stat"><p class="kpi-label">' + item[0] + '</p><p class="' + item[2] + '">' + item[1] + '</p></article>';
        }).join('');
    };

    var renderSimTrades = function (trades) {
        if (!simTradesBody) {
            return;
        }
        simAllTrades = Array.isArray(trades) ? trades.slice() : [];
        var totalTrades = simAllTrades.length;
        var visibleTrades = simShowAllTrades ? simAllTrades : simAllTrades.slice(-TRADE_PREVIEW_LIMIT);

        if (!totalTrades) {
            simTradesBody.innerHTML = '<tr><td colspan="6">No completed trades for this parameter set.</td></tr>';
            if (simTradesMeta) {
                simTradesMeta.textContent = 'No trades found for current setup.';
            }
            if (simTradesToggleBtn) {
                simTradesToggleBtn.hidden = true;
            }
            return;
        }

        if (simTradesMeta) {
            if (simShowAllTrades || totalTrades <= TRADE_PREVIEW_LIMIT) {
                simTradesMeta.textContent = 'Showing all ' + totalTrades + ' trades.';
            } else {
                simTradesMeta.textContent = 'Showing latest ' + visibleTrades.length + ' of ' + totalTrades + ' trades.';
            }
        }

        if (simTradesToggleBtn) {
            simTradesToggleBtn.hidden = totalTrades <= TRADE_PREVIEW_LIMIT;
            simTradesToggleBtn.textContent = simShowAllTrades ? 'Show Recent Only' : 'Show All Trades';
        }

        simTradesBody.innerHTML = visibleTrades.map(function (trade) {
            var returnPct = toFiniteNumber(trade.return_pct, 0);
            var pnl = toFiniteNumber(trade.profit_loss, 0);
            var outcomeClass = trade.outcome === 'WIN' ? 'long' : 'short';
            var returnClass = returnPct >= 0 ? 'long' : 'short';
            var pnlClass = pnl >= 0 ? 'long' : 'short';
            return [
                '<tr>',
                '<td><div class="mono small">' + escapeHtml(trade.entry_date) + '</div><div class="text-body-secondary small">₹' + fmt(trade.entry_price, 2) + '</div></td>',
                '<td><div class="mono small">' + escapeHtml(trade.exit_date) + '</div><div class="text-body-secondary small">₹' + fmt(trade.exit_price, 2) + '</div></td>',
                '<td class="text-end"><span class="' + returnClass + '">' + fmtPct(returnPct) + '</span></td>',
                '<td class="text-end"><span class="' + pnlClass + '">' + fmtInr(pnl) + '</span></td>',
                '<td class="text-end mono">' + escapeHtml(trade.holding_days) + '</td>',
                '<td><div>' + escapeHtml(trade.exit_reason) + '</div><div class="small ' + outcomeClass + '">' + escapeHtml(trade.outcome) + '</div></td>',
                '</tr>'
            ].join('');
        }).join('');
    };

    var runAdaptiveSimulation = function () {
        if (!hasSimulationControls) {
            return Promise.resolve();
        }
        simRunStatus.textContent = 'Running adaptive simulation...';
        simRunBtn.disabled = true;

        var requestUrl = buildUrl(simUrl, {
            capital: simCapital.value,
            position_size_pct: simPositionSize.value,
            max_holding_days: simMaxHoldDays.value,
            stop_loss_pct: simStopLoss.value,
            take_profit_pct: simTakeProfit.value,
            brokerage_pct: simBrokerage.value,
            slippage_pct: simSlippage.value
        });

        return fetchJson(requestUrl)
            .then(function (payload) {
                simShowAllTrades = false;
                renderSimMetrics(payload.result || {});
                renderSimTrades((payload.result && payload.result.trades) || []);
                simRunStatus.textContent = 'Adaptive simulation updated.';
            })
            .catch(function (error) {
                simRunStatus.textContent = 'Simulation failed: ' + error.message;
                simTradesBody.innerHTML = '<tr><td colspan="6">Simulation failed. Check parameters and retry.</td></tr>';
                if (simTradesMeta) {
                    simTradesMeta.textContent = 'Simulation failed.';
                }
                if (simTradesToggleBtn) {
                    simTradesToggleBtn.hidden = true;
                }
                if (simLiveMetrics) {
                    simLiveMetrics.innerHTML = '';
                }
            })
            .finally(function () {
                simRunBtn.disabled = false;
            });
    };

    var runProjection = function () {
        if (!hasProjectionControls) {
            return Promise.resolve();
        }
        projectionStatus.textContent = 'Calculating forward estimator...';
        projectionRunBtn.disabled = true;

        var selected = Array.from(document.querySelectorAll('.projection-horizon:checked')).map(function (input) {
            return input.value;
        });
        if (!selected.length) {
            selected = ['1', '5', '20'];
        }

        var requestUrl = buildUrl(projectionUrl, {
            amount: projectionAmount.value,
            horizons: selected.join(','),
            custom_horizon_days: projectionCustomHorizon.value
        });

        return fetchJson(requestUrl)
            .then(function (payload) {
                if (projectionMethod) {
                    var learning = payload.learning_profile || {};
                    var learningText = learning.calibration_ready
                        ? ' Calibration quality: ' + escapeHtml(learning.quality_label || 'N/A') + ' (' + fmt(learning.quality_score_pct, 1) + '%).'
                        : ' Calibration is warming up with limited samples.';
                    projectionMethod.textContent = (payload.method_note || '') + learningText;
                }

                var cards = Array.isArray(payload.projections) ? payload.projections : [];
                if (!cards.length) {
                    projectionCards.innerHTML = '<p class="text-body-secondary small">No projections available.</p>';
                    projectionStatus.textContent = 'Projection returned no data.';
                    return;
                }

                projectionCards.innerHTML = cards.map(function (item) {
                    var projectedGain = toFiniteNumber(item.projected_gain, 0);
                    var gainClass = projectedGain >= 0 ? 'long' : 'short';
                    var completedValue = item.latest_completed_return_pct;
                    var completedText = completedValue === null || completedValue === undefined
                        ? 'N/A'
                        : fmtPct(completedValue);
                    var rawReturnText = item.raw_expected_return_pct === null || item.raw_expected_return_pct === undefined
                        ? 'N/A'
                        : fmtPct(item.raw_expected_return_pct);

                    return [
                        '<div class="col-12 col-lg-6">',
                        '<article class="projection-card">',
                        '<h3>' + escapeHtml(item.horizon_days) + ' Trading Day Horizon</h3>',
                        '<p><strong>Calibrated Return:</strong> <span class="' + gainClass + '">' + fmtPct(item.expected_return_pct) + '</span></p>',
                        '<p><strong>Raw Return:</strong> ' + rawReturnText + '</p>',
                        '<p><strong>Projected Value:</strong> ' + fmtInr(item.projected_value) + '</p>',
                        '<p><strong>Projected Gain:</strong> <span class="' + gainClass + '">' + fmtInr(projectedGain) + '</span></p>',
                        '<p><strong>Range (1-sigma):</strong> ' + fmtInr(item.range_low_value) + ' to ' + fmtInr(item.range_high_value) + '</p>',
                        '<p><strong>Reality Check Hit Rate:</strong> ' + fmt(item.direction_hit_rate, 2) + '%</p>',
                        '<p><strong>Avg Error:</strong> ' + fmt(item.avg_abs_error_pct, 2) + '% over ' + escapeHtml(item.validation_samples) + ' samples</p>',
                        '<p><strong>Last Completed Horizon:</strong> ' + completedText + '</p>',
                        '<p class="muted mono">Reference Close ' + fmt(item.reference_close, 2) + ' on ' + escapeHtml(item.reference_date) + '</p>',
                        '</article>',
                        '</div>'
                    ].join('');
                }).join('');

                projectionStatus.textContent = 'Projection updated.';
            })
            .catch(function (error) {
                projectionStatus.textContent = 'Projection failed: ' + error.message;
                projectionCards.innerHTML = '<p class="text-body-secondary small">Projection unavailable right now.</p>';
            })
            .finally(function () {
                projectionRunBtn.disabled = false;
            });
    };

    if (hasSimulationControls && simRunBtn) {
        simRunBtn.addEventListener('click', runAdaptiveSimulation);
    }
    if (simTradesToggleBtn) {
        simTradesToggleBtn.addEventListener('click', function () {
            simShowAllTrades = !simShowAllTrades;
            renderSimTrades(simAllTrades);
        });
    }
    if (hasProjectionControls && projectionRunBtn) {
        projectionRunBtn.addEventListener('click', runProjection);
    }

    if (hasSimulationControls) {
        runAdaptiveSimulation();
    } else if (simRunStatus) {
        simRunStatus.textContent = 'Simulation controls are unavailable in this view.';
    }

    if (hasProjectionControls) {
        runProjection();
    } else if (projectionStatus) {
        projectionStatus.textContent = 'Projection API not configured for this view.';
    }
})();
