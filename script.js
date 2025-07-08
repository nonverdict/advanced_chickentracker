document.addEventListener('DOMContentLoaded', () => {
    const htmlElement = document.documentElement;
    const overlayBackdrop = document.getElementById('overlay-backdrop');
    const navButtons = document.querySelectorAll('.nav-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const normalFeedImg = document.getElementById('normal-feed-img');
    const diseaseList = document.getElementById('disease-list');
    const diseaseDetailsPane = document.getElementById('disease-details');
    const diseaseDetailTitle = document.getElementById('disease-detail-title');
    const diseaseSymptoms = document.getElementById('disease-symptoms');
    const diseaseCure = document.getElementById('disease-cure');
    const diseasePrevention = document.getElementById('disease-prevention');
    const reportsBell = document.getElementById('reports-bell');
    const reportsListPane = document.getElementById('reports-list');
    const reportsContent = document.getElementById('reports-content');
    const toggleFeedBtn = document.getElementById('toggle-feed-btn');
    const themeToggleButton = document.getElementById('theme-toggle-btn');

    // Dev Panel Elements
    const devModeOverlay = document.getElementById('dev-mode-overlay');
    const sidebarGradientStart = document.getElementById('sidebar-gradient-start');
    const sidebarGradientEnd = document.getElementById('sidebar-gradient-end');
    const januyaGradientStart = document.getElementById('januya-gradient-start');
    const januyaGradientEnd = document.getElementById('januya-gradient-end');
    const elementBgColor = document.getElementById('element-bg-color');
    const headerBgColor = document.getElementById('header-bg-color');
    const accentPrimaryColor = document.getElementById('accent-primary-color');
    const accentSecondaryColor = document.getElementById('accent-secondary-color');
    const fontPrimarySelect = document.getElementById('font-primary-select');
    const fontDisplaySelect = document.getElementById('font-display-select');
    const fontAccentSelect = document.getElementById('font-accent-select');
    const saveThemeBtn = document.getElementById('save-theme-btn');
    const resetThemeBtn = document.getElementById('reset-theme-btn');

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    let socket = null;
    let currentVideoUrl = null;
    let activeTabId = document.querySelector('.tab-pane.active')?.id || 'home-pane';
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelayBase = 3000;
    let reconnectTimer = null;
    let isLiveFeedActive = true;
    let isTransitioningTabs = false;
    const animationDuration = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--duration-medium') || '0.35s') * 1000;


    const diseaseData = {
        coccidiosis: { name: "Кокцидиоз", symptoms: "Кровавый понос, вялость, взъерошенность оперения, снижение аппетита и веса.", cure: "Применение кокцидиостатиков. Поддерживающая терапия.", prevention: "Соблюдение гигиены, регулярная чистка, качественные корма." },
        mareks: { name: "Болезнь Марека", symptoms: "Параличи, опухоли, помутнение глаз.", cure: "Эффективного лечения нет. Больных птиц выбраковывают.", prevention: "Вакцинация цыплят. Строгие ветеринарно-санитарные меры." },
        newcastle: { name: "Болезнь Ньюкасла", symptoms: "Респираторные признаки, диарея, нервные явления, снижение яйценоскости.", cure: "Специфического лечения нет. Больных птиц уничтожают.", prevention: "Вакцинация. Карантин. Санитарный контроль." },
        pullorosis: { name: "Пуллороз", symptoms: "У цыплят: слабость, белый понос, высокая смертность. У взрослых: снижение яйценоскости.", cure: "Антибиотики и сульфаниламиды. Малоэффективно для цыплят.", prevention: "Приобретение здоровых цыплят. Анализ крови у взрослого поголовья." },
        pasteurellosis: { name: "Пастереллёз", symptoms: "Внезапная гибель или вялость, синюшность, диарея, артриты.", cure: "Антибиотики на ранних стадиях.", prevention: "Вакцинация. Борьба с грызунами. Санитарные нормы." }
    };

    function generateSimulatedReports() {
        const reports = [];
        const statuses = [ { text: 'ЗДОРОВ', class: 'status-healthy' }, { text: 'ВНИМАНИЕ', class: 'status-warning' }, { text: 'ОПАСНОСТЬ', class: 'status-danger' }];
        const issues = [ 'Отклонений нет', 'Небольшая вариация температуры', 'Подозрительная активность', 'Обнаружена высокая температура', 'Замечено аномальное поведение', 'Снижение подвижности', 'Скученность птиц'];
        const reportCount = Math.floor(Math.random() * 4) + 2; // 2-5 reports
         for (let i = 0; i < reportCount; i++) {
              const randomStatusIndex = Math.floor(Math.random() * statuses.length);
              const randomStatus = statuses[randomStatusIndex];
              let randomIssue = issues[Math.floor(Math.random() * issues.length)];
              if(randomStatus.class === 'status-healthy') randomIssue = issues[0];

              const date = new Date(Date.now() - i * Math.random() * 1000 * 60 * 60 * 20);
              reports.push({
                  timestamp: date.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
                  status: randomStatus.text, statusClass: randomStatus.class,
                  details: `${randomIssue}. Зона: ${Math.floor(Math.random()*3)+1}.`
              });
         }
         reports.sort((a, b) => new Date(b.timestamp.split(', ')[0].split('.').reverse().join('-') + 'T' + b.timestamp.split(', ')[1]) - new Date(a.timestamp.split(', ')[0].split('.').reverse().join('-') + 'T' + a.timestamp.split(', ')[1]));
         return reports;
    }

    // --- REVISED THEME LOGIC ---
    function applyTheme(theme) {
        // 'default' means our Ethereal Mint (very light green) theme
        // 'dark' means our Modern Dark theme
        if (theme === 'dark') {
            htmlElement.setAttribute('data-theme', 'dark');
        } else {
            // For the default Ethereal Mint theme, we REMOVE the data-theme attribute
            // so that the base CSS rules under html {} apply.
            htmlElement.removeAttribute('data-theme');
        }
        localStorage.setItem('theme', theme); // Store 'default' or 'dark'
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            // Check if 'data-theme="dark"' is currently set
            const isCurrentlyDark = htmlElement.getAttribute('data-theme') === 'dark';
            if (isCurrentlyDark) {
                applyTheme('default'); // Switch to Ethereal Mint (default)
            } else {
                applyTheme('dark');    // Switch to Modern Dark
            }
        });
    }

    // Initialize theme:
    // 1. Check localStorage.
    // 2. If nothing in localStorage, it will be 'default' (Ethereal Mint) because
    //    we assume data-theme is NOT set on HTML by default, and CSS under html {} will apply.
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        applyTheme(savedTheme); // This will correctly set 'dark' or remove attribute for 'default'
    } else {
        // No saved theme, so it's already in the default "Ethereal Mint" state
        // (assuming HTML has no data-theme attribute initially).
        // We can save 'default' to localStorage for future visits.
        applyTheme('default'); // Explicitly apply and save 'default' if nothing is stored
    }
    // --- END OF REVISED THEME LOGIC ---


    function connectWebSocket() {
        if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        if (normalFeedImg) setVideoFeedStatus("Подключение WebSocket...");
        if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); currentVideoUrl = null; }
        
        try {
            socket = new WebSocket(wsUrl);
        } catch (e) {
            console.error("WebSocket instantiation error:", e);
            if (normalFeedImg) setVideoFeedStatus("Ошибка WebSocket URL.");
            return;
        }

        socket.onopen = () => {
            console.log("[WS] Connection established.");
            reconnectAttempts = 0;
            if (activeTabId === 'camera-pane') {
                const streamToSubscribe = isLiveFeedActive ? "normal_video" : "static_video";
                if (normalFeedImg) setVideoFeedStatus(`Запрос ${isLiveFeedActive ? 'live' : 'video'} потока...`);
                sendWebSocketMessage({ action: "subscribe", stream: streamToSubscribe });
            }
        };

        socket.onmessage = (event) => {
            if (event.data instanceof Blob) {
                if (activeTabId === 'camera-pane' && normalFeedImg) {
                    const newUrl = URL.createObjectURL(event.data);
                    if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); }
                    normalFeedImg.src = newUrl;
                    currentVideoUrl = newUrl;
                    normalFeedImg.classList.remove('feed-placeholder');
                    normalFeedImg.alt = isLiveFeedActive ? "Прямая трансляция" : "Обработанное видео";
                    normalFeedImg.onerror = () => { console.error("[WS] Error loading image from Blob URL."); normalFeedImg.classList.add('feed-placeholder'); normalFeedImg.alt = "Ошибка загрузки видео"; };
                } else {
                     // If not on camera pane or image element doesn't exist, just revoke to free memory
                     const tempUrl = URL.createObjectURL(event.data);
                     URL.revokeObjectURL(tempUrl);
                }
            } else if (typeof event.data === 'string') {
                 try {
                    const data = JSON.parse(event.data);
                    if (data.type === "anomaly_alert") addAnomalyToReports(data);
                    // Potentially other JSON message types here
                } catch (e) {
                    // console.log("[WS] Received non-JSON text message:", event.data);
                }
            }
        };

        socket.onerror = (error) => { 
            console.error("[WS] WebSocket error:", error);
            if(activeTabId === 'camera-pane' && normalFeedImg) setVideoFeedStatus("Ошибка WebSocket.");
        };

        socket.onclose = (event) => {
            console.log(`[WS] Connection closed: Code=${event.code}, Reason='${event.reason || 'N/A'}'`);
            socket = null;
            if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); currentVideoUrl = null; }
            if(activeTabId === 'camera-pane' && normalFeedImg) {
                 setVideoFeedStatus(`WebSocket отключен (Код: ${event.code})`);
                 normalFeedImg.src = ""; normalFeedImg.alt = `WebSocket отключен`;
                 normalFeedImg.classList.add('feed-placeholder');
            }
            // Only attempt to reconnect if it wasn't a clean close (1000) or browser navigation (1005)
            // and we haven't exceeded max attempts.
            if (event.code !== 1000 && event.code !== 1005 && reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                const delay = reconnectDelayBase * Math.pow(1.5, reconnectAttempts - 1);
                if (normalFeedImg && activeTabId === 'camera-pane') setVideoFeedStatus(`Переподключение #${reconnectAttempts}...`);
                reconnectTimer = setTimeout(connectWebSocket, delay);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                if (normalFeedImg && activeTabId === 'camera-pane') setVideoFeedStatus("Не удалось подключиться.");
            }
        };
    }

    function sendWebSocketMessage(message) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            try {
                socket.send(JSON.stringify(message));
            } catch (e) {
                console.error("[WS] Error sending message:", e);
            }
        }
    }

    function setVideoFeedStatus(statusText) {
        // Only update if the image is currently a placeholder or has no src
        if (normalFeedImg && (!normalFeedImg.src || normalFeedImg.classList.contains('feed-placeholder'))) {
            normalFeedImg.alt = statusText; // Update alt text which is visible for placeholders
            // If it's not already a placeholder, make it one
            if (!normalFeedImg.classList.contains('feed-placeholder')) {
                normalFeedImg.classList.add('feed-placeholder');
            }
        }
    }
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (isTransitioningTabs) return;
            const targetPaneId = button.getAttribute('data-tab');
            if (button.classList.contains('active') || !targetPaneId) return;

            isTransitioningTabs = true;
            const currentActivePane = document.querySelector('.tab-pane.active');
            const targetPane = document.getElementById(targetPaneId);

            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            if (currentActivePane) {
                currentActivePane.classList.add('exiting');
                // currentActivePane.style.display = 'none'; // Alternative to opacity for performance
            }

            if (targetPane) {
                targetPane.classList.remove('exiting');
                // targetPane.style.display = 'flex'; // Make sure it's flex for layout
                targetPane.scrollTop = 0; // Reset scroll position
                targetPane.classList.add('active');
                animateTabContent(targetPaneId);
            }
            
            const oldActiveTabId = activeTabId;
            activeTabId = targetPaneId;
            
            // Manage WebSocket subscriptions based on tab change
            const streamName = isLiveFeedActive ? "normal_video" : "static_video";
            if (targetPaneId === 'camera-pane' && oldActiveTabId !== 'camera-pane') {
                if (normalFeedImg) setVideoFeedStatus(`Запрос ${isLiveFeedActive ? 'live' : 'video'} потока...`);
                sendWebSocketMessage({ action: "subscribe", stream: streamName });
                if (!socket || socket.readyState !== WebSocket.OPEN) { // Connect if not already open
                    connectWebSocket();
                }
            } else if (targetPaneId !== 'camera-pane' && oldActiveTabId === 'camera-pane') {
                 sendWebSocketMessage({ action: "unsubscribe", stream: streamName });
                 if (normalFeedImg) {
                     normalFeedImg.src = ""; // Clear the image
                     normalFeedImg.alt = "Трансляция остановлена.";
                     normalFeedImg.classList.add('feed-placeholder');
                     if (currentVideoUrl) { // Revoke previous object URL
                         URL.revokeObjectURL(currentVideoUrl);
                         currentVideoUrl = null;
                     }
                     setVideoFeedStatus(""); // Clear status text from placeholder
                 }
                 // Optionally close WebSocket if no other tab needs it
                 // if (socket && socket.readyState === WebSocket.OPEN) socket.close(1000);
            }
            hideAllOverlays(); // Close any open overlays when switching tabs

            setTimeout(() => {
                if(currentActivePane) {
                    currentActivePane.classList.remove('active', 'exiting');
                    // currentActivePane.style.display = ''; // Reset display style
                }
                isTransitioningTabs = false;
            }, animationDuration);
        });
    });

    function animateTabContent(paneId) {
        const pane = document.getElementById(paneId);
        if (!pane) return;

        const elementsToAnimate = pane.querySelectorAll('.animatable-on-tab-load');
        elementsToAnimate.forEach((el, index) => {
            el.style.animation = 'none'; // Reset animation
            void el.offsetWidth; // Trigger reflow
            // Apply new animation
            el.style.animation = `fadeInSlideUp 0.6s var(--transition-swift) ${0.1 + index * 0.1}s forwards`;
        });
        
        if (paneId === 'info-pane' && diseaseList) {
            const items = diseaseList.querySelectorAll('.disease-item');
            items.forEach((item, index) => {
                item.style.animation = 'none';
                item.style.opacity = '0'; // Start invisible
                void item.offsetWidth; // Trigger reflow
                item.style.animation = `itemPopIn var(--duration-medium) var(--transition-bounce) forwards`;
                item.style.animationDelay = `${0.2 + index * 0.07}s`;
            });
        }
    }

    function showOverlay(overlayElement) {
        if (!overlayElement) return;
        hideAllOverlays(); // Ensure only one overlay is active
        overlayElement.classList.add('active');
        // Show backdrop only on desktop
        if (overlayBackdrop && window.innerWidth >= 768) {
            overlayBackdrop.classList.add('active');
        }
    }

    function hideOverlay(overlayElement) {
        if (!overlayElement || !overlayElement.classList.contains('active')) return;
        overlayElement.classList.remove('active');
        if (overlayBackdrop) { // Always try to remove backdrop class
            overlayBackdrop.classList.remove('active');
        }
    }
    
    function hideAllOverlays() {
        document.querySelectorAll('.details-overlay.active').forEach(hideOverlay);
    }

    // Centralized handler for closing overlays
    function setupOverlayClosers() {
        document.body.addEventListener('click', (event) => {
            // Close if click is on a .close-overlay-btn or the backdrop itself
            if (event.target.closest('.close-overlay-btn') || 
                (event.target.id === 'overlay-backdrop' && overlayBackdrop.classList.contains('active'))) {
                hideAllOverlays();
            }
        });
        // Add Escape key listener to close overlays
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                hideAllOverlays();
            }
        });
    }
    
    if (diseaseList) {
        diseaseList.addEventListener('click', (event) => {
            const listItem = event.target.closest('.disease-item');
            if (!listItem) return;

            const diseaseId = listItem.getAttribute('data-disease-id');
            const data = diseaseData[diseaseId];

            if (data && diseaseDetailsPane) {
                diseaseDetailTitle.textContent = data.name;
                diseaseSymptoms.textContent = data.symptoms;
                diseaseCure.textContent = data.cure;
                diseasePrevention.textContent = data.prevention;
                showOverlay(diseaseDetailsPane);
            }
        });
    }

    if (reportsBell) {
        reportsBell.addEventListener('click', () => {
            if (!reportsContent || !reportsListPane) return;
            // Check if reports need to be (re)generated
            if (reportsContent.children.length === 0 || (reportsContent.children.length === 1 && reportsContent.firstChild.nodeName === 'P')) {
                reportsContent.innerHTML = ''; // Clear "no reports" message or old reports
                const simulatedReports = generateSimulatedReports();
                if (simulatedReports.length > 0) {
                    simulatedReports.forEach(report => addReportToDOM(report, false));
                } else {
                    reportsContent.innerHTML = '<p style="text-align:center; padding: 20px; color: var(--text-placeholder);">Нет доступных отчетов.</p>';
                }
            }
            reportsBell.classList.remove('has-new-report'); // Clear notification dot
            showOverlay(reportsListPane);
        });
    }
    
    function addReportToDOM(reportData, isAnomaly = false) {
        if (!reportsContent) return;

        const reportDiv = document.createElement('div');
        reportDiv.className = 'report-item';

        let statusClass = reportData.statusClass || 'status-info';
        let statusText = reportData.status || 'ИНФО';

        // Override if it's an anomaly from WebSocket
        if (isAnomaly) {
            statusText = reportData.anomalyType === 'warning' ? 'ВНИМАНИЕ' : 'ОПАСНОСТЬ';
            statusClass = reportData.anomalyType === 'warning' ? 'status-warning' : 'status-danger';
        }

        reportDiv.innerHTML = `
            <div class="report-meta">${reportData.timestamp}</div>
            <div><span class="report-status ${statusClass}">${statusText}</span></div>
            <div>${reportData.details}</div>`;

        // If "no reports" message is present, remove it
        if (reportsContent.firstChild && reportsContent.firstChild.nodeName === 'P') {
            reportsContent.innerHTML = '';
        }
        reportsContent.prepend(reportDiv); // Add new report at the top
    }

    function addAnomalyToReports(anomalyData) {
        const report = {
            timestamp: new Date(anomalyData.timestamp * 1000).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
            details: anomalyData.message + (anomalyData.details ? ` (${anomalyData.details})` : ''),
            anomalyType: anomalyData.anomaly_type || 'danger' // Default to 'danger' if type is missing
        };
        addReportToDOM(report, true);
        if (reportsBell) reportsBell.classList.add('has-new-report');
    }

    if (toggleFeedBtn && normalFeedImg) {
        toggleFeedBtn.addEventListener('click', () => {
            isLiveFeedActive = !isLiveFeedActive;

            // Clear previous feed
            if (normalFeedImg) {
                normalFeedImg.src = ''; // Important to stop the browser from trying to load old blob
                if (currentVideoUrl) {
                    URL.revokeObjectURL(currentVideoUrl);
                    currentVideoUrl = null;
                }
                normalFeedImg.classList.add('feed-placeholder');
            }

            const newStream = isLiveFeedActive ? "normal_video" : "static_video";
            const oldStream = isLiveFeedActive ? "static_video" : "normal_video";
            toggleFeedBtn.textContent = isLiveFeedActive ? 'Показать Видео' : 'Камера (Live)';
            if (normalFeedImg) setVideoFeedStatus(`Запрос ${isLiveFeedActive ? 'live' : 'video'} потока...`);

            if (activeTabId === 'camera-pane') { // Only send WS messages if on camera tab
                sendWebSocketMessage({ action: "unsubscribe", stream: oldStream });
                sendWebSocketMessage({ action: "subscribe", stream: newStream });
                if (!socket || socket.readyState !== WebSocket.OPEN) { // Ensure connection
                    connectWebSocket();
                }
            }
        });
    }

    // Initialisation
    setupOverlayClosers();
    if (toggleFeedBtn) { // Set initial button text
        toggleFeedBtn.textContent = isLiveFeedActive ? 'Показать Видео' : 'Камера (Live)';
    }
    if(normalFeedImg) { // Ensure placeholder state initially
        normalFeedImg.classList.add('feed-placeholder');
    }
    // Start WebSocket connection
    connectWebSocket();
    // Animate content of the initially active tab
    animateTabContent(activeTabId);

    // --- DEV PANEL LOGIC ---
    const infoNavButton = document.querySelector('.nav-button[data-tab="info-pane"]');
    let devClickCount = 0;
    let devClickTimer = null;

    if (infoNavButton) {
        infoNavButton.addEventListener('click', () => {
            if (devModeOverlay && devModeOverlay.classList.contains('active')) return;
            devClickCount++;
            if (devClickTimer) clearTimeout(devClickTimer);
            if (devClickCount >= 3) {
                showOverlay(devModeOverlay);
                devClickCount = 0;
            } else {
                devClickTimer = setTimeout(() => { devClickCount = 0; }, 1000);
            }
        });
    }

    function applyCustomStyles() {
        const root = document.documentElement;
        const bottomNav = document.querySelector('.bottom-nav');
        const appTitle = document.querySelector('.app-title');

        // Sidebar
        const newSidebarGradient = `linear-gradient(160deg, ${sidebarGradientStart.value} 0%, ${sidebarGradientEnd.value} 100%)`;
        if (bottomNav) {
            bottomNav.style.background = newSidebarGradient;
        }
        
        // Januya Title
        const newJanuyaGradient = `linear-gradient(45deg, ${januyaGradientStart.value}, ${januyaGradientEnd.value})`;
        if (appTitle) {
            appTitle.style.background = newJanuyaGradient;
            appTitle.style.webkitBackgroundClip = 'text';
            appTitle.style.webkitTextFillColor = 'transparent';
        }

        // General Colors
        root.style.setProperty('--bg-secondary', elementBgColor.value);
        
        const hexToRgba = (hex, alpha = 0.85) => {
            const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
            return `rgba(${r},${g},${b},${alpha})`;
        };
        root.style.setProperty('--bg-glass', hexToRgba(headerBgColor.value));

        root.style.setProperty('--accent-primary', accentPrimaryColor.value);
        root.style.setProperty('--accent-secondary', accentSecondaryColor.value);

        // Fonts
        root.style.setProperty('--font-primary', fontPrimarySelect.value);
        root.style.setProperty('--font-display', fontDisplaySelect.value);
        root.style.setProperty('--font-accent', fontAccentSelect.value);
    }

    if (devModeOverlay) {
        devModeOverlay.addEventListener('input', applyCustomStyles);
    }

    if (resetThemeBtn) {
        resetThemeBtn.addEventListener('click', () => {
            // Reset inputs to default values
            sidebarGradientStart.value = '#e8f5e9';
            sidebarGradientEnd.value = '#dcedc8';
            januyaGradientStart.value = '#F59E0B';
            januyaGradientEnd.value = '#689f38';
            elementBgColor.value = '#e8f5e9';
            headerBgColor.value = '#e8f5e9';
            accentPrimaryColor.value = '#F59E0B';
            accentSecondaryColor.value = '#689f38';
            fontPrimarySelect.value = "'Poppins', sans-serif";
            fontDisplaySelect.value = "'Righteous', cursive";
            fontAccentSelect.value = "'Georgia', serif";
            
            // Remove all inline styles to revert to stylesheet
            document.documentElement.style.cssText = '';
            document.querySelector('.bottom-nav').style.background = '';
            const appTitle = document.querySelector('.app-title');
            if (appTitle) {
                appTitle.style.background = '';
                appTitle.style.webkitBackgroundClip = '';
                appTitle.style.webkitTextFillColor = '';
            }
        });
    }

    if (saveThemeBtn) {
        saveThemeBtn.addEventListener('click', () => {
            const hexToRgba = (hex, alpha = 0.85) => {
                const [r, g, b] = hex.match(/\w\w/g).map(x => parseInt(x, 16));
                return `rgba(${r},${g},${b},${alpha})`;
            };

            const customCss = `
/* --- Custom Theme Generated by Dev Panel --- */
:root {
    /* COLORS */
    --etm-sidebar-background: linear-gradient(160deg, ${sidebarGradientStart.value} 0%, ${sidebarGradientEnd.value} 100%);
    --accent-primary: ${accentPrimaryColor.value};
    --accent-secondary: ${accentSecondaryColor.value};
    --bg-secondary: ${elementBgColor.value};
    --bg-glass: ${hexToRgba(headerBgColor.value)};
    
    /* FONTS */
    --font-primary: ${fontPrimarySelect.value};
    --font-display: ${fontDisplaySelect.value};
    --font-accent: ${fontAccentSelect.value};
}

/* Apply sidebar background for desktop */
@media (min-width: 768px) {
    .bottom-nav {
        background: var(--etm-sidebar-background);
    }
}

/* Apply Januya title gradient */
.app-title {
    background: linear-gradient(45deg, ${januyaGradientStart.value}, ${januyaGradientEnd.value});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
`;
            const blob = new Blob([customCss.trim()], { type: 'text/css' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'custom_theme.css';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            alert('Custom theme saved as custom_theme.css! Link it in your HTML to use it.');
        });
    }
});
