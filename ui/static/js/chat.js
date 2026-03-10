(function () {
    'use strict';

    var panel = document.querySelector('.chat-panel');
    if (!panel) {
        return;
    }

    var historyEl = document.getElementById('chatHistory');
    var form = document.getElementById('chatForm');
    var input = document.getElementById('chatInput');
    var defaultTicker = panel.dataset.defaultTicker || '';

    var appendMessage = function (text, role) {
        if (!historyEl) {
            return;
        }
        var wrapper = document.createElement('div');
        wrapper.className = 'chat-message ' + role;
        var bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = text;
        wrapper.appendChild(bubble);
        historyEl.appendChild(wrapper);
        historyEl.scrollTop = historyEl.scrollHeight;
    };

    var history = [];
    var HISTORY_LIMIT = 20;

    var sendMessage = function (message) {
        appendMessage(message, 'user');
        appendMessage('Thinking...', 'bot');
        history.push('User: ' + message);
        if (history.length > HISTORY_LIMIT) {
            history = history.slice(-HISTORY_LIMIT);
        }

        return fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                default_ticker: defaultTicker,
                history: history
            })
        })
            .then(function (response) { return response.json(); })
            .then(function (payload) {
                var lastBot = historyEl.querySelector('.chat-message.bot:last-child .chat-bubble');
                if (lastBot) {
                    lastBot.textContent = payload.reply || 'No response available.';
                }
                history.push('Ultron: ' + (payload.reply || ''));
                if (history.length > HISTORY_LIMIT) {
                    history = history.slice(-HISTORY_LIMIT);
                }
                if (payload.needs && Array.isArray(payload.needs) && payload.needs.length) {
                    appendMessage('Ultron needs: ' + payload.needs.join(' | '), 'bot');
                }
            })
            .catch(function () {
                var lastBot = historyEl.querySelector('.chat-message.bot:last-child .chat-bubble');
                if (lastBot) {
                    lastBot.textContent = 'Chat failed. Please try again.';
                }
            });
    };

    if (form) {
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            if (!input) {
                return;
            }
            var message = String(input.value || '').trim();
            if (!message) {
                return;
            }
            input.value = '';
            sendMessage(message);
        });
    }
})();
