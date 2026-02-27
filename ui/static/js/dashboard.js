document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('tickerSearch');
    const tableBody = document.querySelector('#overviewTable tbody');
    const noSearchRows = document.getElementById('noSearchRows');
    const liveTrackerSection = document.querySelector('.live-tracker');
    const liveQuotesBody = document.getElementById('liveQuotesBody');
    const liveTrackerStatus = document.getElementById('liveTrackerStatus');
    const liveTrackerUpdated = document.getElementById('liveTrackerUpdated');
    const marketTrendBadge = document.getElementById('marketTrendBadge');
    const marketPulseText = document.getElementById('marketPulseText');
    const marketUpCount = document.getElementById('marketUpCount');
    const marketDownCount = document.getElementById('marketDownCount');
    const marketFlatCount = document.getElementById('marketFlatCount');
    const processingCard = document.getElementById('analysisProcessingCard');
    const processingMessage = document.getElementById('analysisProcessingMessage');

    let previousQuotePriceMap = new Map();
    let liveRequestId = 0;

    function formatNumber(value, fractionDigits) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return '-';
        }
        return Number(value).toLocaleString(undefined, {
            minimumFractionDigits: fractionDigits,
            maximumFractionDigits: fractionDigits
        });
    }

    function formatTime(date) {
        return date.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function applyClientFilters() {
        if (!tableBody) {
            return;
        }
        const rows = Array.from(tableBody.querySelectorAll('tr.click-row'));
        if (!rows.length) {
            return;
        }
        const query = searchInput ? searchInput.value.trim().toUpperCase() : '';
        let visibleCount = 0;

        rows.forEach(function (row) {
            const ticker = (row.dataset.ticker || '').toUpperCase();
            const show = ticker.includes(query);
            row.style.display = show ? '' : 'none';
            if (show) {
                visibleCount += 1;
            }
        });

        if (noSearchRows) {
            noSearchRows.style.display = visibleCount === 0 ? '' : 'none';
        }
    }

    function renderMarketPulse(quotes) {
        const up = quotes.filter(function (q) { return Number(q.change_pct) > 0; }).length;
        const down = quotes.filter(function (q) { return Number(q.change_pct) < 0; }).length;
        const flat = Math.max(0, quotes.length - up - down);

        if (marketUpCount) {
            marketUpCount.textContent = String(up);
        }
        if (marketDownCount) {
            marketDownCount.textContent = String(down);
        }
        if (marketFlatCount) {
            marketFlatCount.textContent = String(flat);
        }

        if (!marketTrendBadge || !marketPulseText) {
            return;
        }

        marketTrendBadge.className = 'badge rounded-pill';
        if (!quotes.length) {
            marketTrendBadge.classList.add('bg-secondary-soft');
            marketTrendBadge.textContent = 'No Data';
            marketPulseText.textContent = 'Unable to determine market breadth right now.';
            return;
        }

        if (up > down) {
            marketTrendBadge.classList.add('bg-success-soft');
            marketTrendBadge.textContent = 'Market Up';
            marketPulseText.textContent = up + ' of ' + quotes.length + ' tracked symbols are advancing.';
            return;
        }
        if (down > up) {
            marketTrendBadge.classList.add('bg-danger-soft');
            marketTrendBadge.textContent = 'Market Down';
            marketPulseText.textContent = down + ' of ' + quotes.length + ' tracked symbols are declining.';
            return;
        }

        marketTrendBadge.classList.add('bg-secondary-soft');
        marketTrendBadge.textContent = 'Mixed';
        marketPulseText.textContent = 'Advancers and decliners are balanced in the tracked basket.';
    }

    function renderLiveQuotes(data) {
        if (!liveQuotesBody) {
            return;
        }

        const quotes = Array.isArray(data.quotes) ? data.quotes : [];
        renderMarketPulse(quotes);

        if (!quotes.length) {
            liveQuotesBody.innerHTML = '<tr><td colspan="3" class="text-center small text-body-secondary py-4">No live quotes available.</td></tr>';
            return;
        }

        liveQuotesBody.innerHTML = quotes.map(function (item) {
            const change = item.change_pct;
            let changeClass = 'text-body-secondary';
            if (change !== null && change !== undefined) {
                changeClass = change >= 0 ? 'text-success' : 'text-danger';
            }
            const changeText = (change === null || change === undefined)
                ? '-'
                : (change >= 0 ? '+' : '') + formatNumber(change, 2) + '%';

            const lastPrice = Number(item.last_price);
            const previousPrice = previousQuotePriceMap.get(item.ticker);
            let rowFlashClass = '';
            if (Number.isFinite(lastPrice) && Number.isFinite(previousPrice)) {
                if (lastPrice > previousPrice) {
                    rowFlashClass = 'flash-up';
                } else if (lastPrice < previousPrice) {
                    rowFlashClass = 'flash-down';
                }
            }
            if (Number.isFinite(lastPrice)) {
                previousQuotePriceMap.set(item.ticker, lastPrice);
            }

            return (
                '<tr class="' + rowFlashClass + '">' +
                '<td><span class="font-monospace small">' + escapeHtml(item.ticker) + '</span></td>' +
                '<td class="text-end font-monospace small">' + formatNumber(item.last_price, 2) + '</td>' +
                '<td class="text-end font-monospace small ' + changeClass + '">' + changeText + '</td>' +
                '</tr>'
            );
        }).join('');
    }

    async function refreshLiveTracker() {
        if (!liveTrackerSection || !liveQuotesBody) {
            return;
        }

        const tickers = (liveTrackerSection.dataset.tickers || '').trim();
        if (!tickers) {
            liveQuotesBody.innerHTML = '<tr><td colspan="3" class="text-center small text-body-secondary py-3">No tickers to track.</td></tr>';
            return;
        }

        if (liveTrackerStatus) {
            liveTrackerStatus.className = 'badge rounded-pill bg-secondary-soft';
            liveTrackerStatus.textContent = 'Refreshing';
        }
        const requestId = ++liveRequestId;

        try {
            const response = await fetch('/api/live-quotes?tickers=' + encodeURIComponent(tickers), { cache: 'no-store' });
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }

            const data = await response.json();
            if (requestId !== liveRequestId) {
                return;
            }
            renderLiveQuotes(data);

            if (liveTrackerStatus) {
                const quotes = Array.isArray(data.quotes) ? data.quotes : [];
                const liveCount = quotes.filter(function (q) { return q.mode === 'LIVE'; }).length;
                liveTrackerStatus.className = 'badge rounded-pill ' + (liveCount > 0 ? 'bg-success-soft' : 'bg-secondary-soft');
                liveTrackerStatus.textContent = liveCount + ' / ' + quotes.length + ' Live';
            }
            if (liveTrackerUpdated) {
                if (data.updated_at) {
                    liveTrackerUpdated.textContent = data.updated_at;
                } else {
                    liveTrackerUpdated.textContent = formatTime(new Date());
                }
            }
        } catch (error) {
            if (requestId !== liveRequestId) {
                return;
            }
            if (liveTrackerStatus) {
                liveTrackerStatus.className = 'badge rounded-pill bg-danger-soft';
                liveTrackerStatus.textContent = 'Error';
            }
            if (liveQuotesBody) {
                liveQuotesBody.innerHTML = '<tr><td colspan="3" class="text-center small text-body-secondary py-4">Live quotes unavailable. Retrying...</td></tr>';
            }
        }
    }

    async function pollAnalysisStatus() {
        if (!processingCard) {
            return;
        }
        try {
            const response = await fetch('/api/analysis-status', { cache: 'no-store' });
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            if (processingMessage && data.message) {
                processingMessage.textContent = data.message + '...';
            }
            if (data.status === 'ready' || data.status === 'error') {
                window.location.reload();
            }
        } catch (error) {
            // Silent retry on next interval.
        }
    }

    document.body.addEventListener('click', function (event) {
        const row = event.target.closest('.click-row');
        if (row && row.dataset.href) {
            window.location = row.dataset.href;
        }
    });

    if (searchInput) {
        searchInput.addEventListener('input', applyClientFilters);
        applyClientFilters();
    }

    if (liveTrackerSection) {
        refreshLiveTracker();
        const refreshSeconds = Number(liveTrackerSection.dataset.refreshSeconds || 20);
        window.setInterval(refreshLiveTracker, Math.max(10, refreshSeconds) * 1000);
    }

    if (processingCard) {
        const pollSeconds = Number(processingCard.dataset.pollSeconds || 4);
        window.setInterval(pollAnalysisStatus, Math.max(2, pollSeconds) * 1000);
    }
});
