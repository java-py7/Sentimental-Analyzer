const b = document.body;

// Global chart and data instances (declared globally to prevent TDZ issues)
var compGroupedChart = null;
var compDonutAChart = null;
var compDonutBChart = null;
var compareData = null;
var dashAggregateChart = null;

function toggleMode() {
    b.classList.toggle('dark-mode');
    const themeIcon = document.querySelector(".darklight-ModeToggle");

    if (document.body.classList.contains("dark-mode")) {
        if (themeIcon) {
            themeIcon.classList.remove("fa-sun");
            themeIcon.classList.add("fa-moon");
        }
    } else {
        if (themeIcon) {
            themeIcon.classList.remove("fa-moon");
            themeIcon.classList.add("fa-sun");
        }
    }

    // Refresh active charts with updated theme colors safely
    try {
        if (typeof sentimentChart !== 'undefined' && sentimentChart && typeof renderSentimentChart === 'function') {
            renderSentimentChart();
        }
    } catch (e) {
        console.warn("Could not re-render sentiment chart:", e);
    }

    try {
        if (compareData && typeof renderCompareGroupedChart === 'function' && typeof renderCompareDonutCharts === 'function') {
            renderCompareGroupedChart(compareData.video_a, compareData.video_b);
            renderCompareDonutCharts(compareData.video_a, compareData.video_b);
        }
    } catch (e) {
        console.warn("Could not re-render compare charts:", e);
    }

    try {
        if (dashAggregateChart && window.dashData && typeof renderDashAggregateChart === 'function') {
            renderDashAggregateChart(window.dashData.total_positive || 0, window.dashData.total_negative || 0, window.dashData.total_neutral || 0);
        }
    } catch (e) {
        console.warn("Could not re-render dashboard chart:", e);
    }
}

const menu = document.querySelector(".menu");
const toggle = document.querySelector(".toggle");
const menuIcon = document.querySelector(".menu-icon");

if (toggle && menu) {
    toggle.addEventListener("click", () => {
        menu.classList.toggle("active");

        if (menuIcon) {
            if (menu.classList.contains("active")) {
                menuIcon.classList.remove("fa-bars");
                menuIcon.classList.add("fa-times");
            } else {
                menuIcon.classList.remove("fa-times");
                menuIcon.classList.add("fa-bars");
            }
        }
    });
}

// =============== MOBILE SIDEBAR CONTROLS ===============
function openMobileSidebar() {
    const sidebar = document.getElementById("mobileSidebar");
    const backdrop = document.getElementById("mobileSidebarBackdrop");
    const toggleBtn = document.getElementById("mobileSidebarToggle");
    if (sidebar && backdrop) {
        sidebar.classList.add("open");
        sidebar.setAttribute("aria-hidden", "false");
        backdrop.classList.add("open");
        backdrop.setAttribute("aria-hidden", "false");
        if (toggleBtn) toggleBtn.setAttribute("aria-expanded", "true");
        document.body.classList.add("mobile-sidebar-open");
    }
}

function closeMobileSidebar() {
    const sidebar = document.getElementById("mobileSidebar");
    const backdrop = document.getElementById("mobileSidebarBackdrop");
    const toggleBtn = document.getElementById("mobileSidebarToggle");
    if (sidebar && backdrop) {
        sidebar.classList.remove("open");
        sidebar.setAttribute("aria-hidden", "true");
        backdrop.classList.remove("open");
        backdrop.setAttribute("aria-hidden", "true");
        if (toggleBtn) toggleBtn.setAttribute("aria-expanded", "false");
        document.body.classList.remove("mobile-sidebar-open");
    }
}

const mobileSidebarToggle = document.getElementById("mobileSidebarToggle");
if (mobileSidebarToggle) {
    mobileSidebarToggle.addEventListener("click", openMobileSidebar);
}

const mobileSidebarClose = document.getElementById("mobileSidebarClose");
if (mobileSidebarClose) {
    mobileSidebarClose.addEventListener("click", closeMobileSidebar);
}

const mobileSidebarBackdrop = document.getElementById("mobileSidebarBackdrop");
if (mobileSidebarBackdrop) {
    mobileSidebarBackdrop.addEventListener("click", closeMobileSidebar);
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        closeMobileSidebar();
    }
});

const logoutBtns = document.querySelectorAll(".logoutBtn");
const logoutModal = document.getElementById("logoutModal");
const cancelLogout = document.getElementById("cancelLogout");

logoutBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        logoutModal.classList.add("active");
    });
});

cancelLogout.addEventListener("click", () => {
    logoutModal.classList.remove("active");
});

logoutModal.addEventListener("click", (e) => {
    if (e.target === logoutModal) {
        logoutModal.classList.remove("active");
    }
});

const params = new URLSearchParams(window.location.search);

if (params.get("login") === "success") {
    showToast("Logged in successfully.", "success");
    window.history.replaceState({}, "", window.location.pathname);
}

if (params.get("logout") === "success") {
    showToast("Logged out successfully.", "success");
    window.history.replaceState({}, "", window.location.pathname);
}

if (params.get("toast") === "login_cancelled") {
    showToast("Google sign-in cancelled.", "warning");
    window.history.replaceState({}, "", window.location.pathname);
}

function showToast(message, type = "info") {

    const icons = {
        success: '<i class="fas fa-circle-check"></i>',
        error: '<i class="fas fa-circle-xmark"></i>',
        warning: '<i class="fas fa-triangle-exclamation"></i>',
        info: '<i class="fas fa-circle-info"></i>'
    };

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    toast.innerHTML = `
        ${icons[type] || icons.info}
        <span>${message}</span>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("show");
    }, 100);

    setTimeout(() => {
        toast.classList.remove("show");

        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// =============== GSAP-STYLE MOTION & REVEAL UTILITIES ===============
function triggerGsapReveals(container) {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const scope = container || document;
        scope.querySelectorAll(".gsap-reveal").forEach(el => el.classList.add("in-view"));
        return;
    }
    const scope = container || document;
    const elements = scope.querySelectorAll(".gsap-reveal");
    elements.forEach(el => {
        el.classList.remove("in-view");
        void el.offsetWidth; // Force reflow
        el.classList.add("in-view");
    });
}

function animateNumber(element, target, duration = 800) {
    if (!element) return;
    const num = parseInt(target, 10);
    if (isNaN(num)) {
        element.textContent = target;
        return;
    }
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        element.textContent = num.toLocaleString();
        return;
    }
    const startTime = performance.now();
    function step(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // easeOutExpo
        const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        const current = Math.round(num * ease);
        element.textContent = current.toLocaleString();
        if (progress < 1) {
            requestAnimationFrame(step);
        } else {
            element.textContent = num.toLocaleString();
        }
    }
    requestAnimationFrame(step);
}

// =============== PRELOADER FUNCTIONS ===============
const PRELOADER_CONFIG = {
    minimumShowTime: 5000,
    messageChangeInterval: 1000
};

const PRELOADER_MESSAGES = [
    { text: "Initializing Analysis...", subtext: "Connecting to YouTube API" },
    { text: "Fetching Comments...", subtext: "Retrieving video comment data" },
    { text: "Processing Data...", subtext: "Cleaning and preprocessing comments" },
    { text: "Running AI Analysis...", subtext: "Applying advanced sentiment algorithms" },
    { text: "Generating Visualizations...", subtext: "Creating charts and word clouds" }
];

const COMPARISON_PRELOADER_MESSAGES = [
    { text: "Connecting to YouTube API...", subtext: "Validating both video sources" },
    { text: "Fetching Video A Comments...", subtext: "Retrieving audience comments for first video" },
    { text: "Fetching Video B Comments...", subtext: "Retrieving audience comments for second video" },
    { text: "Running AI Sentiment Analysis...", subtext: "Classifying sentiments with NLP engine" },
    { text: "Calculating Comparative Deltas...", subtext: "Generating side-by-side comparative visualizations" }
];

function showPreloader(customMessages = null) {
    const preloader = document.getElementById('preloader');
    const textElement = document.getElementById('preloader-text');
    const subtextElement = document.getElementById('preloader-subtext');

    preloader.classList.add('active');
    document.body.style.overflow = 'hidden';

    const messages = customMessages || PRELOADER_MESSAGES;
    let messageIndex = 0;

    // Start with first message
    if (messages[0]) {
        textElement.textContent = messages[0].text;
        subtextElement.textContent = messages[0].subtext;
        messageIndex = 1;
    }

    // Change messages during preloader
    const messageInterval = setInterval(() => {
        if (messageIndex < messages.length) {
            textElement.textContent = messages[messageIndex].text;
            subtextElement.textContent = messages[messageIndex].subtext;
            messageIndex++;
        }
    }, PRELOADER_CONFIG.messageChangeInterval);

    return messageInterval;
}

function hidePreloader() {
    const preloader = document.getElementById('preloader');
    preloader.classList.remove('active');
    document.body.style.overflow = '';
}

document.querySelectorAll(".export-btn").forEach(btn => {
    btn.disabled = true;
});

// =============== FORM SUBMISSION (SINGLE ANALYSIS) ===============
document.addEventListener("DOMContentLoaded", () => {
    // Initial reveals on home page
    triggerGsapReveals(document.getElementById("home"));

    // Single video analysis submission
    document.addEventListener("submit", function (e) {
        if (!e.target.matches('form[action="/analyze"]')) return;

        e.preventDefault();

        const form = e.target;
        const formData = new FormData(form);

        if (!formData.get("video_url")) {
            showToast("Please enter a YouTube URL", "warning");
            return;
        }

        const messageInterval = showPreloader();

        fetch("/analyze", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                clearInterval(messageInterval);
                hidePreloader();

                if (!data.success) {
                    showToast(data.message, "error");
                    return;
                }

                window.chartData = data.summary;
                window.chartData = data.summary;
                window.commentData = data.comments;
                exportData = data;

                const resultsContainer = document.getElementById("resultsContainer");
                resultsContainer.style.display = "block";
                triggerGsapReveals(resultsContainer);

                // Populate video profile banner
                const analysisVideo = document.getElementById("analysisVideo");
                if (analysisVideo) {
                    analysisVideo.textContent = data.video_title || data.video_url;
                    analysisVideo.href = data.video_url;
                    analysisVideo.target = "_blank";
                }

                const thumb = document.getElementById("resultsThumbnail");
                const thumbFallback = document.getElementById("thumbnailFallback");
                if (data.video_id && thumb) {
                    thumb.src = `https://img.youtube.com/vi/${data.video_id}/hqdefault.jpg`;
                    thumb.style.display = "block";
                    if (thumbFallback) thumbFallback.style.display = "none";
                }

                const extLink = document.getElementById("resultsExternalLink");
                if (extLink && data.video_url) {
                    extLink.href = data.video_url;
                    extLink.style.display = "inline-flex";
                }

                const idTag = document.getElementById("resultsVideoIdTag");
                if (idTag && data.video_id) {
                    idTag.textContent = `ID: ${data.video_id}`;
                    idTag.style.display = "inline-flex";
                }

                const totalBadge = document.getElementById("resultsTotalBadge");
                if (totalBadge) totalBadge.textContent = (data.comments ? data.comments.length : 0).toLocaleString();

                renderSummary(data.summary);
                renderComments(data.comments);
                renderSentimentChart();
                renderWordCloud(data.comments);

                // Export Page
                document.getElementById("exportVideo").textContent = data.video_title;
                document.getElementById("exportComments").textContent = data.comments.length;
                document.getElementById("exportStatus").textContent = "Ready";

                document.querySelectorAll(".export-btn").forEach(btn => {
                    btn.disabled = false;
                });

                openTab("home");
                showToast("Analysis completed successfully!", "success");

                // Smooth scroll to results
                setTimeout(() => {
                    resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
                }, 150);
            })
            .catch(err => {
                clearInterval(messageInterval);
                hidePreloader();
                console.error(err);
                showToast("Something went wrong with the analysis.", "error");
            });
    });

    // Paste from clipboard button
    const pasteBtn = document.getElementById("pasteUrlBtn");
    if (pasteBtn) {
        pasteBtn.addEventListener("click", async () => {
            try {
                const text = await navigator.clipboard.readText();
                if (text) {
                    const input = document.getElementById("mainVideoUrl");
                    if (input) {
                        input.value = text.trim();
                        showToast("Pasted from clipboard!", "success");
                    }
                }
            } catch (e) {
                showToast("Please paste URL manually.", "info");
            }
        });
    }

    // Initialize Comparator
    initComparison();
});

// =============== TAB FUNCTIONALITY ===============
function openTab(tabName) {
    document.querySelectorAll(".tab-content").forEach(tab => {
        tab.classList.remove("active");
    });

    const currentTab = document.getElementById(tabName);
    if (currentTab) {
        currentTab.classList.add("active");
    }

    // Bottom navigation active state
    document.querySelectorAll(".tab-button").forEach(btn => {
        btn.classList.remove("active");
    });

    const bottomBtn = document.querySelector(`.tab-button[data-tab="${tabName}"], .tab-button[onclick="openTab('${tabName}')"]`);
    if (bottomBtn) {
        bottomBtn.classList.add("active");
    }

    // Mobile sidebar active state
    document.querySelectorAll(".mobile-sidebar-link").forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("data-tab") === tabName) {
            link.classList.add("active");
        }
    });

    // Close mobile sidebar if open
    closeMobileSidebar();

    // Close radial menu safely
    if (menu) menu.classList.remove("active");
    if (menuIcon) {
        menuIcon.classList.remove("fa-times");
        menuIcon.classList.add("fa-bars");
    }

    // Scroll smoothly to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Staggered GSAP reveal for newly visible tab
    if (currentTab) {
        triggerGsapReveals(currentTab);
    }

    switch (tabName) {
        case "dashboard":
            loadDashboard();
            break;

        case "compare":
            // Focus first input if empty
            const urlA = document.getElementById("compareUrlA");
            if (urlA && !urlA.value) {
                setTimeout(() => urlA.focus(), 300);
            }
            break;

        case "history":
            if (!isLoggedIn) {
                showToast("Please sign in to access your analysis history.", "warning");
                openTab("about");
                return;
            }
            loadHistory();
            break;

        case "export":
            break;

        case "settings":
            break;

        case "youtube":
            break;

        case "about":
            break;
    }
}

// =============== CHART.JS ===============

let sentimentChart = null;

function renderSentimentChart() {

    if (!window.chartData) return;

    const container = document.getElementById("sentimentChart");

    if (!container) return;

    // Clear previous chart
    container.innerHTML = '<canvas id="sentimentCanvas"></canvas>';

    const ctx = document.getElementById("sentimentCanvas").getContext("2d");

    if (sentimentChart) {
        sentimentChart.destroy();
    }

    const isDark = document.body.classList.contains("dark-mode");
    const textColor = isDark ? "#e5e7eb" : "#374151";
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)";

    sentimentChart = new Chart(ctx, {

        type: "bar",

        data: {

            labels: [
                "Positive",
                "Negative",
                "Neutral"
            ],

            datasets: [{
                label: "Comments",

                data: [
                    window.chartData.positive,
                    window.chartData.negative,
                    window.chartData.neutral
                ],

                backgroundColor: [
                    "#10b981",
                    "#ef4444",
                    "#f59e0b"
                ],

                borderRadius: 10
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 400,
                onComplete: function () {
                    window.sentimentChartRendered = true;
                    saveAnalysisImages();
                }
            },
            plugins: {

                legend: {
                    display: false
                },
                tooltip: {
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' }
                }

            },

            scales: {

                x: {
                    ticks: { color: textColor, font: { family: 'Inter', weight: 500 } },
                    grid: { color: gridColor }
                },

                y: {
                    beginAtZero: true,
                    ticks: { color: textColor, font: { family: 'Inter' } },
                    grid: { color: gridColor }
                }

            }

        }

    });

}

document.addEventListener("DOMContentLoaded", () => {

    renderSentimentChart();

});


function escapeHtml(text) {
    if (!text) return "";
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function fillSampleVideo(url) {
    const input = document.getElementById("mainVideoUrl");
    if (input) {
        input.value = url;
        input.focus();
        showToast("Sample video loaded! Click Analyze Video.", "info");
    }
}

function selectSentimentFilter(sentiment) {
    document.querySelectorAll(".sentiment-pill-group .filter-pill").forEach(pill => {
        pill.classList.remove("active");
        if (pill.getAttribute("data-sentiment") === sentiment) {
            pill.classList.add("active");
        }
    });

    const select = document.getElementById("sentimentFilter");
    if (select) {
        select.value = sentiment;
        filterComments();
    }
}

function clearCommentSearch() {
    const search = document.getElementById("commentSearch");
    if (search) {
        search.value = "";
        const clearBtn = document.getElementById("searchClearBtn");
        if (clearBtn) clearBtn.style.display = "none";
        filterComments();
    }
}

function renderSummary(summary) {
    const total = summary.positive + summary.negative + summary.neutral;
    animateNumber(document.querySelector(".total-comments"), total);
    animateNumber(document.querySelector(".positive-card .stat-number"), summary.positive);
    animateNumber(document.querySelector(".negative-card .stat-number"), summary.negative);
    animateNumber(document.querySelector(".neutral-card .stat-number"), summary.neutral);

    const posPct = total > 0 ? ((summary.positive / total) * 100).toFixed(1) : '0.0';
    const negPct = total > 0 ? ((summary.negative / total) * 100).toFixed(1) : '0.0';
    const neuPct = total > 0 ? ((summary.neutral / total) * 100).toFixed(1) : '0.0';

    const posTag = document.getElementById("posTrendTag");
    if (posTag) posTag.textContent = `+${posPct}%`;
    const negTag = document.getElementById("negTrendTag");
    if (negTag) negTag.textContent = `${negPct}%`;
    const neuTag = document.getElementById("neuTrendTag");
    if (neuTag) neuTag.textContent = `${neuPct}%`;

    const posFill = document.getElementById("posProgressFill");
    if (posFill) posFill.style.width = `${posPct}%`;
    const negFill = document.getElementById("negProgressFill");
    if (negFill) negFill.style.width = `${negPct}%`;
    const neuFill = document.getElementById("neuProgressFill");
    if (neuFill) neuFill.style.width = `${neuPct}%`;
}


// =============== COMMENTS ===============

let allComments = [];

function renderComments(comments) {
    allComments = comments || [];

    const search = document.getElementById("commentSearch");
    const filter = document.getElementById("sentimentFilter");
    const clearBtn = document.getElementById("searchClearBtn");

    if (search) search.value = "";
    if (filter) filter.value = "all";
    if (clearBtn) clearBtn.style.display = "none";

    document.querySelectorAll(".sentiment-pill-group .filter-pill").forEach(pill => {
        pill.classList.remove("active");
        if (pill.getAttribute("data-sentiment") === "all") pill.classList.add("active");
    });

    displayComments(allComments);

    if (search) {
        search.oninput = () => {
            if (clearBtn) clearBtn.style.display = search.value.trim() ? "block" : "none";
            filterComments();
        };
    }

    if (filter) {
        filter.onchange = filterComments;
    }
}

function displayComments(comments) {
    const list = document.querySelector(".comments-list");
    if (!list) return;

    const counter = document.getElementById("commentsCountText");
    if (counter) counter.textContent = `${comments.length} Comments`;

    list.innerHTML = "";

    if (comments.length === 0) {
        list.innerHTML = `
            <div class="no-comments">
                <div class="empty-icon-ring"><i class="far fa-folder-open"></i></div>
                <h3>No Matching Comments</h3>
                <p>Try searching for a different keyword or resetting your sentiment filter.</p>
            </div>
        `;
        return;
    }

    comments.forEach((item, index) => {
        const sentimentClass = (item.sentiment || 'neutral').toLowerCase();
        let sentimentIcon = 'fa-meh';
        if (sentimentClass === 'positive') sentimentIcon = 'fa-smile';
        else if (sentimentClass === 'negative') sentimentIcon = 'fa-frown';

        list.innerHTML += `
            <div class="comment-item ${sentimentClass}">
                <div class="comment-avatar ${sentimentClass}">
                    <i class="fas ${sentimentIcon}"></i>
                </div>
                <div class="comment-body">
                    <div class="comment-meta-header">
                        <span class="sentiment-badge badge-${sentimentClass}">
                            <i class="fas ${sentimentIcon}"></i> ${item.sentiment}
                        </span>
                        <span class="comment-index">#${index + 1}</span>
                    </div>
                    <div class="comment-text">
                        ${escapeHtml(item.comment)}
                    </div>
                </div>
            </div>
        `;
    });
}

function filterComments() {
    const searchInput = document.getElementById("commentSearch");
    const filterSelect = document.getElementById("sentimentFilter");

    if (!searchInput || !filterSelect) return;

    const search = searchInput.value.trim().toLowerCase();
    const filter = filterSelect.value;

    const filtered = allComments.filter(item => {
        const matchText = (item.comment || '').toLowerCase().includes(search);
        const matchSentiment = filter === "all" || item.sentiment === filter;
        return matchText && matchSentiment;
    });

    displayComments(filtered);
}


const WORDCLOUD_STOPWORDS = new Set([
    "the", "a", "an", "and", "or", "to", "of", "is", "are", "was", "were",
    "i", "you", "he", "she", "it", "they", "we", "this", "that", "in", "on",
    "for", "with", "my", "your", "our", "their", "be", "have", "has", "had",
    "at", "as", "from", "by", "but", "if", "so", "not", "me", "do", "did",
    "does", "can", "will", "just", "very", "video", "videos", "youtube", "channel",
    "subscribe", "watch", "watching", "comment", "comments", "sub", "view", "views",
    "http", "https", "www", "com", "would", "could", "should", "one", "all",
    "there", "their", "what", "out", "about", "who", "get", "which", "when", "make",
    "time", "know", "people", "into", "some", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after",
    "how", "work", "first", "well", "way", "even", "new", "want", "because", "any",
    "these", "give", "day", "most", "us", "got", "bro", "really", "much", "dont",
    "cant", "wont", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent"
]);

const WORDCLOUD_PALETTE = [
    "#6366f1", "#8b5cf6", "#ec4899", "#3b82f6",
    "#10b981", "#14b8a6", "#f59e0b", "#f97316"
];

// Single source of truth for word cloud capacity
const WORDCLOUD_MAX_WORDS = 150;
const regeneratedHistoryWordClouds = new Set();
let historyViewSequence = 0;
let currentActiveHistoryId = null;

function buildWordCloudOnCanvas(comments, canvas, onComplete) {
    if (!canvas) return;

    if (!comments || !Array.isArray(comments) || comments.length === 0) {
        if (typeof onComplete === "function") onComplete();
        return;
    }

    const words = {};

    comments.forEach(item => {
        const text = (typeof item === "string" ? item : item.comment) || "";
        text.toLowerCase()
            .replace(/[^\w\s]/g, " ")
            .split(/\s+/)
            .forEach(word => {
                if (word.length < 3 || word.length > 22) return;
                if (WORDCLOUD_STOPWORDS.has(word)) return;
                if (/^\d+$/.test(word)) return; // filter pure numbers
                words[word] = (words[word] || 0) + 1;
            });
    });

    const entries = Object.entries(words);
    if (entries.length === 0) {
        if (typeof onComplete === "function") onComplete();
        return;
    }

    // Sort descending by frequency and pick the configured top words
    entries.sort((a, b) => b[1] - a[1]);
    const maxWordsCount = Math.min(entries.length, WORDCLOUD_MAX_WORDS);
    const topWords = entries.slice(0, maxWordsCount);

    const maxFreq = topWords[0][1];
    const minFreq = topWords[topWords.length - 1][1];

    // Responsive font sizes based on canvas width and height
    const minSize = Math.max(13, Math.round(canvas.width / 52));
    const maxSize = Math.min(46, Math.max(28, Math.round(canvas.width / 16)));

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let finished = false;
    function finish() {
        if (finished) return;
        finished = true;
        canvas.removeEventListener("wordcloudstop", finish);
        canvas.removeEventListener("wordcloudabort", finish);
        if (typeof onComplete === "function") {
            onComplete();
        }
    }

    canvas.addEventListener("wordcloudstop", finish);
    canvas.addEventListener("wordcloudabort", finish);
    setTimeout(finish, 1800); // Safety fallback

    try {
        WordCloud(canvas, {
            list: topWords,
            gridSize: Math.max(6, Math.round(8 * (canvas.width / 800))),
            weightFactor: function (size) {
                if (maxFreq <= minFreq) return Math.round((minSize + maxSize) / 2);
                // Math.sqrt scaling normalizes outliers so top words never dominate excessively
                const ratio = Math.sqrt(size - minFreq) / Math.sqrt(maxFreq - minFreq);
                return Math.round(minSize + ratio * (maxSize - minSize));
            },
            fontFamily: "Outfit, Inter, system-ui, sans-serif",
            fontWeight: "600",
            color: function (word) {
                let hash = 0;
                for (let i = 0; i < word.length; i++) {
                    hash = (hash << 5) - hash + word.charCodeAt(i);
                }
                return WORDCLOUD_PALETTE[Math.abs(hash) % WORDCLOUD_PALETTE.length];
            },
            rotateRatio: 0.25,
            rotationSteps: 2,
            backgroundColor: "transparent",
            drawOutOfBound: false,
            shrinkToFit: true,
            shuffle: true
        });
    } catch (e) {
        console.error("WordCloud generation error:", e);
        finish();
    }
}

function renderWordCloud(comments) {
    const canvas = document.getElementById("wordCloud");
    const emptyState = document.getElementById("wordCloudEmpty");
    if (!canvas) return;

    if (!comments || comments.length === 0) {
        canvas.style.display = "none";
        if (emptyState) emptyState.style.display = "flex";
        return;
    }

    const wrapper = document.getElementById("wordCloudWrapper");
    if (wrapper && wrapper.clientWidth > 0) {
        canvas.width = Math.max(600, wrapper.clientWidth);
        canvas.height = 340;
    } else {
        canvas.width = 800;
        canvas.height = 340;
    }

    if (emptyState) emptyState.style.display = "none";
    canvas.style.display = "block";

    window.wordCloudRendered = false;

    buildWordCloudOnCanvas(comments, canvas, () => {
        window.wordCloudRendered = true;
        saveAnalysisImages();
    });

    // Fallback save in case event is missed
    setTimeout(() => {
        saveAnalysisImages();
    }, 1500);
}

let isSavingImages = false;
let lastSavedDbId = null;

async function saveAnalysisImages() {
    if (!exportData || !exportData.database_id) {
        return;
    }

    if (lastSavedDbId === exportData.database_id) {
        return; // Already saved
    }

    const chartCanvas = document.getElementById("sentimentCanvas");
    const wordCanvas = document.getElementById("wordCloud");

    if (!chartCanvas || !wordCanvas) {
        return;
    }

    if (isSavingImages) return;
    isSavingImages = true;

    try {
        const chartDataUrl = chartCanvas.toDataURL("image/png");
        const wordDataUrl = wordCanvas.toDataURL("image/png");

        const response = await fetch("/save-images", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                analysis_id: exportData.database_id,
                chart_image: chartDataUrl,
                wordcloud_image: wordDataUrl
            })
        });

        const result = await response.json();
        if (result && result.success) {
            lastSavedDbId = exportData.database_id;
            console.log("Analysis images saved to database successfully.");
        }
    } catch (e) {
        console.warn("Could not save analysis images:", e);
    } finally {
        isSavingImages = false;
    }
}



/*============================================
                EXPORT DATA
============================================*/

let exportData = null;

/*============================================
            EXPORT FUNCTIONS
============================================*/

function downloadTXT() {

    if (!exportData) {

        alert("Please analyze a YouTube video first.");

        return;

    }

    let txt = "";

    txt += "YOUTUBE SENTIMENT ANALYSIS\n";
    txt += "========================================\n\n";

    txt += "Video Title : " + exportData.video_title + "\n";
    txt += "YouTube URL : " + exportData.video_url + "\n\n";

    txt += "Positive : " + exportData.summary.positive + "\n";
    txt += "Negative : " + exportData.summary.negative + "\n";
    txt += "Neutral  : " + exportData.summary.neutral + "\n";
    txt += "Total Comments : " + exportData.comments.length + "\n\n";

    txt += "========================================\n\n";

    exportData.comments.forEach((item, index) => {

        txt += `${index + 1}. [${item.sentiment}]\n`;
        txt += item.comment + "\n\n";

    });

    const blob = new Blob([txt], {
        type: "text/plain;charset=utf-8"
    });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = "YouTube_Sentiment_Report.txt";

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);

}


function downloadCSV() {

    if (!exportData) {

        alert("Please analyze a YouTube video first.");

        return;

    }

    let csv = "";

    csv += "Sentiment,Comment\n";

    exportData.comments.forEach(item => {

        const comment = item.comment
            .replace(/"/g, '""')
            .replace(/\n/g, " ");

        csv += `"${item.sentiment}","${comment}"\n`;

    });

    const blob = new Blob(
        [csv],
        {
            type: "text/csv;charset=utf-8;"
        }
    );

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = "YouTube_Comments.csv";

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);

}


function downloadJSON() {

    if (!exportData) {

        alert("Please analyze a YouTube video first.");

        return;

    }

    const json = JSON.stringify(exportData, null, 4);

    const blob = new Blob(
        [json],
        {
            type: "application/json"
        }
    );

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = "YouTube_Analysis.json";

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);

}


function downloadChart() {

    const link = document.createElement("a");

    link.download = "Sentiment_Chart.png";

    if (exportData && exportData.chart) {

        link.href = exportData.chart;

    } else {

        if (!window.chartData) {

            alert("Please analyze a YouTube video first.");

            return;

        }

        const canvas = document.getElementById("sentimentCanvas");

        if (!canvas) {

            alert("Chart not found.");

            return;

        }

        link.href = canvas.toDataURL("image/png");

    }

    link.click();

}


function downloadWordCloud() {

    const link = document.createElement("a");

    link.download = "Word_Cloud.png";

    if (exportData && exportData.wordcloud) {

        link.href = exportData.wordcloud;

    } else {

        const canvas = document.getElementById("wordCloud");

        if (!canvas) {

            alert("Word Cloud not found.");

            return;

        }

        link.href = canvas.toDataURL("image/png");

    }

    link.click();

}


/*============================================
                DOWNLOAD PDF
============================================*/
async function downloadPDF() {
    if (!exportData) {
        alert("Please analyze a YouTube video first.");
        return;
    }
    const doc = {
        pageSize: "A4",
        pageMargins: [40, 45, 40, 45],
        content: [
            {
                text: "YouTube Sentiment Analysis Report",
                style: "title",
                alignment: "center"
            },
            {
                canvas: [
                    {
                        type: "line",
                        x1: 0,
                        y1: 0,
                        x2: 515,
                        y2: 0,
                        lineWidth: 1.5,
                        lineColor: "#7c3aed"
                    }
                ],
                margin: [0, 12, 0, 18]
            },
            {
                table: {
                    widths: ["*"],
                    body: [
                        [
                            {
                                stack: [
                                    {
                                        columns: [
                                            { text: "Video Title:", width: 80, bold: true, color: "#7c3aed" },
                                            { text: exportData.video_title, width: "*" }
                                        ],
                                        margin: [0, 0, 0, 6]
                                    },
                                    {
                                        columns: [
                                            { text: "Video URL:", width: 80, bold: true, color: "#7c3aed" },
                                            { text: exportData.video_url, color: "blue", width: "*" }
                                        ],
                                        margin: [0, 0, 0, 6]
                                    },
                                    {
                                        columns: [
                                            { text: "Generated:", width: 80, bold: true, color: "#7c3aed" },
                                            { text: new Date().toLocaleString(), width: "*" }
                                        ]
                                    }
                                ],
                                fillColor: "#f8fafc",
                                margin: [12, 12, 12, 12]
                            }
                        ]
                    ]
                },
                layout: {
                    hLineWidth: function () { return 0.5; },
                    vLineWidth: function () { return 0.5; },
                    hLineColor: function () { return "#e2e8f0"; },
                    vLineColor: function () { return "#e2e8f0"; }
                },
                margin: [0, 0, 0, 15]
            },
            {
                text: "Analysis Summary",
                style: "section"
            },
            {
                columns: [
                    {
                        width: "23%",
                        table: {
                            widths: ["*"],
                            body: [
                                [
                                    {
                                        text: [
                                            { text: "POSITIVE\n", fontSize: 9, bold: true, color: "#ffffff" },
                                            { text: exportData.summary.positive.toString(), fontSize: 20, bold: true, color: "#ffffff" }
                                        ],
                                        fillColor: "#22c55e",
                                        alignment: "center",
                                        margin: [0, 10, 0, 10]
                                    }
                                ]
                            ]
                        },
                        layout: "noBorders"
                    },
                    { width: "2.6%", text: "" },
                    {
                        width: "23%",
                        table: {
                            widths: ["*"],
                            body: [
                                [
                                    {
                                        text: [
                                            { text: "NEUTRAL\n", fontSize: 9, bold: true, color: "#ffffff" },
                                            { text: exportData.summary.neutral.toString(), fontSize: 20, bold: true, color: "#ffffff" }
                                        ],
                                        fillColor: "#3b82f6",
                                        alignment: "center",
                                        margin: [0, 10, 0, 10]
                                    }
                                ]
                            ]
                        },
                        layout: "noBorders"
                    },
                    { width: "2.6%", text: "" },
                    {
                        width: "23%",
                        table: {
                            widths: ["*"],
                            body: [
                                [
                                    {
                                        text: [
                                            { text: "NEGATIVE\n", fontSize: 9, bold: true, color: "#ffffff" },
                                            { text: exportData.summary.negative.toString(), fontSize: 20, bold: true, color: "#ffffff" }
                                        ],
                                        fillColor: "#ef4444",
                                        alignment: "center",
                                        margin: [0, 10, 0, 10]
                                    }
                                ]
                            ]
                        },
                        layout: "noBorders"
                    },
                    { width: "2.6%", text: "" },
                    {
                        width: "23%",
                        table: {
                            widths: ["*"],
                            body: [
                                [
                                    {
                                        text: [
                                            { text: "TOTAL\n", fontSize: 9, bold: true, color: "#ffffff" },
                                            { text: exportData.comments.length.toString(), fontSize: 20, bold: true, color: "#ffffff" }
                                        ],
                                        fillColor: "#7c3aed",
                                        alignment: "center",
                                        margin: [0, 10, 0, 10]
                                    }
                                ]
                            ]
                        },
                        layout: "noBorders"
                    }
                ],
                margin: [0, 0, 0, 20]
            },
            {
                text: "Sentiment Chart",
                style: "section"
            },
            {
                image: document.getElementById("sentimentCanvas").toDataURL("image/png"),
                width: 480,
                alignment: "center",
                margin: [0, 5, 0, 15]
            },
            {
                text: "Word Cloud",
                style: "section"
            },
            {
                image: document.getElementById("wordCloud").toDataURL("image/png"),
                width: 480,
                alignment: "center",
                margin: [0, 5, 0, 15]
            },
            {
                text: "Comments Analysis",
                style: "section",
                pageBreak: "before"
            },
            {
                table: {
                    headerRows: 1,
                    widths: [35, 75, "*"],
                    body: [
                        [
                            { text: "No.", style: "tableHeader" },
                            { text: "Sentiment", style: "tableHeader" },
                            { text: "Comment", style: "tableHeader" }
                        ],
                        ...exportData.comments.map((item, index) => {
                            let sentimentColor = "#3b82f6";
                            if (item.sentiment === "Positive") {
                                sentimentColor = "#22c55e";
                            }
                            else if (item.sentiment === "Negative") {
                                sentimentColor = "#ef4444";
                            }
                            return [
                                {
                                    text: String(index + 1),
                                    alignment: "center",
                                    margin: [0, 5, 0, 5]
                                },
                                {
                                    text: item.sentiment,
                                    bold: true,
                                    color: "white",
                                    fillColor: sentimentColor,
                                    alignment: "center",
                                    margin: [0, 5, 0, 5]
                                },
                                {
                                    text: item.comment,
                                    margin: [5, 5, 5, 5]
                                }
                            ];
                        })
                    ]
                },
                layout: {
                    fillColor: function (row) {
                        if (row === 0) {
                            return "#111827";
                        }
                        return row % 2 === 0 ? "#f9fafb" : null;
                    },
                    hLineColor: function () { return "#e5e7eb"; },
                    vLineColor: function () { return "#e5e7eb"; },
                    hLineWidth: function () { return 0.5; },
                    vLineWidth: function () { return 0.5; }
                }
            }
        ],
        styles: {
            title: {
                fontSize: 22,
                bold: true,
                color: "#111827"
            },
            heading: {
                fontSize: 12,
                bold: true,
                color: "#7c3aed"
            },
            section: {
                fontSize: 16,
                bold: true,
                color: "#1f2937",
                margin: [0, 15, 0, 8]
            },
            tableHeader: {
                bold: true,
                color: "white",
                alignment: "center",
                fontSize: 11,
                margin: [0, 6, 0, 6]
            }
        }
    };
    doc.footer = function (currentPage, pageCount) {
        return {
            columns: [
                {
                    text: "YouTube Sentiment Analyzer",
                    margin: [40, 0]
                },
                {
                    text: new Date().toLocaleDateString(),
                    alignment: "center"
                },
                {
                    text: "Page " + currentPage + " of " + pageCount,
                    alignment: "right",
                    margin: [0, 0, 40, 0]
                }
            ],
            fontSize: 9,
            color: "#9ca3af"
        };
    };
    pdfMake.createPdf(doc).download("YouTube_Sentiment_Report.pdf");
}


function searchYouTube() {
    const input = document.getElementById("youtubeSearchInput");
    const query = input.value.trim();

    if (!query) {
        input.focus();
        return;
    }

    window.open(
        "https://www.youtube.com/results?search_query=" + encodeURIComponent(query),
        "_blank"
    );
}


// ==================== HISTORY PAGE ====================

let currentHistoryAnalysis = null;
let historyData = [];
let deleteAnalysisId = null;

// Load History
async function loadHistory() {

    const response = await fetch("/get-history");

    if (response.status !== 200) {

        document.getElementById("historyList").innerHTML = "";
        document.getElementById("historyEmpty").style.display = "flex";

        return;
    }

    historyData = await response.json();

    renderHistory(historyData);

}

// Render History Cards
function renderHistory(history) {

    const historyList = document.getElementById("historyList");
    const historyEmpty = document.getElementById("historyEmpty");

    historyList.innerHTML = "";

    if (history.length === 0) {

        historyEmpty.style.display = "flex";

        return;
    }

    historyEmpty.style.display = "none";

    history.forEach(item => {

        historyList.innerHTML += `

<div class="history-row">

    <div class="history-title">
        <i class="fas fa-file-alt"></i>
        <span>${item.title}</span>
    </div>

    <div class="history-date">
        ${item.created_at}
    </div>

    <button
        class="history-action"
        onclick="viewAnalysis(${item.id})">
        <i class="fas fa-eye"></i>
        View
    </button>

    <button
        class="history-action delete"
        onclick="showDeleteModal(${item.id})">
        <i class="fas fa-trash"></i>
        Delete
    </button>

</div>

`;

    });

}

// Search
const historySearchInput = document.getElementById("historySearch");
if (historySearchInput) {
    historySearchInput.addEventListener("input", function () {
        const value = this.value.toLowerCase();
        const filtered = (historyData || []).filter(item =>
            (item.title && item.title.toLowerCase().includes(value)) ||
            (item.video_id && item.video_id.toLowerCase().includes(value))
        );
        renderHistory(filtered);
    });
}

// Sort
const historySortSelect = document.getElementById("historySort");
if (historySortSelect) {
    historySortSelect.addEventListener("change", function () {
        let sorted = [...(historyData || [])];
        switch (this.value) {

            case "newest":
                sorted.sort((a, b) => b.id - a.id);
                break;

            case "oldest":
                sorted.sort((a, b) => a.id - b.id);
                break;

            case "positive":
                sorted.sort((a, b) => b.positive - a.positive);
                break;

            case "negative":
                sorted.sort((a, b) => b.negative - a.negative);
                break;

            case "comments":
                sorted.sort((a, b) => b.total_comments - a.total_comments);
                break;

        }

        renderHistory(sorted);

    });
}

// Dynamic fallback renderers for historical records that were saved without images:
function generateHistoricalChartDataUrl(summary) {
    return new Promise((resolve) => {
        try {
            const canvas = document.createElement("canvas");
            canvas.width = 600;
            canvas.height = 340;
            const ctx = canvas.getContext("2d");
            const isDark = document.body.classList.contains("dark-mode");
            const textColor = isDark ? "#e5e7eb" : "#374151";

            const chart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: ["Positive", "Negative", "Neutral"],
                    datasets: [{
                        data: [summary.positive || 0, summary.negative || 0, summary.neutral || 0],
                        backgroundColor: ["#10b981", "#ef4444", "#f59e0b"],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: false,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: textColor, font: { weight: "600", family: "Inter" } } },
                        y: { ticks: { color: textColor, font: { family: "Inter" } }, beginAtZero: true }
                    }
                }
            });
            const dataUrl = canvas.toDataURL("image/png");
            chart.destroy();
            resolve(dataUrl);
        } catch (e) {
            console.error("Historical chart generation error:", e);
            resolve(null);
        }
    });
}

function generateHistoricalWordCloudDataUrl(comments) {
    return new Promise((resolve) => {
        try {
            if (!comments || comments.length === 0) {
                resolve(null);
                return;
            }
            const canvas = document.createElement("canvas");
            canvas.width = 700;
            canvas.height = 360;
            buildWordCloudOnCanvas(comments, canvas, () => {
                try {
                    const dataUrl = canvas.toDataURL("image/png");
                    resolve(dataUrl);
                } catch (e) {
                    resolve(null);
                }
            });
        } catch (e) {
            console.error("Historical wordcloud generation error:", e);
            resolve(null);
        }
    });
}

// View Analysis
async function viewAnalysis(id) {
    const targetId = Number(id) || id;
    const thisRequestId = ++historyViewSequence;
    currentActiveHistoryId = targetId;

    const modal = document.getElementById("historyModal");
    const chartImg = document.getElementById("historyChart");
    const chartLoader = document.getElementById("historyChartLoader");
    const chartFallback = document.getElementById("historyChartFallback");
    const wordImg = document.getElementById("historyWordCloud");
    const wordLoader = document.getElementById("historyWordCloudLoader");
    const wordFallback = document.getElementById("historyWordCloudFallback");

    if (chartLoader) chartLoader.style.display = "flex";
    if (chartImg) chartImg.style.display = "none";
    if (chartFallback) chartFallback.style.display = "none";

    if (wordLoader) wordLoader.style.display = "flex";
    if (wordImg) wordImg.style.display = "none";
    if (wordFallback) wordFallback.style.display = "none";

    if (modal) modal.style.display = "flex";

    try {
        const response = await fetch(`/get-analysis/${targetId}`);
        const data = await response.json();

        // Guard against race conditions if another analysis was opened while fetching
        if (currentActiveHistoryId !== targetId || historyViewSequence !== thisRequestId) {
            return;
        }

        if (!data.success || !data.analysis) {
            showToast("Analysis record could not be loaded.", "error");
            if (chartLoader) chartLoader.style.display = "none";
            if (chartFallback) chartFallback.style.display = "flex";
            if (wordLoader) wordLoader.style.display = "none";
            if (wordFallback) wordFallback.style.display = "flex";
            return;
        }

        currentHistoryAnalysis = data.analysis;

        /*========================================
                    HEADER
        ========================================*/
        document.getElementById("historyModalTitle").textContent = data.analysis.video_title || "YouTube Analysis";

        const videoLink = document.getElementById("historyVideoLink");
        if (videoLink) {
            videoLink.href = data.analysis.video_url || "#";
            videoLink.textContent = data.analysis.video_url || "Link";
        }

        /*========================================
                    SUMMARY
        ========================================*/
        document.getElementById("historyPositive").textContent = data.analysis.summary.positive || 0;
        document.getElementById("historyNegative").textContent = data.analysis.summary.negative || 0;
        document.getElementById("historyNeutral").textContent = data.analysis.summary.neutral || 0;
        document.getElementById("historyComments").textContent = (data.analysis.comments || []).length;

        /*========================================
                    COMMENTS
        ========================================*/
        historyComments = data.analysis.comments || [];
        renderHistoryComments(historyComments);
        const searchInput = document.getElementById("historyCommentSearch");
        const filterSelect = document.getElementById("historyCommentFilter");
        if (searchInput) {
            searchInput.value = "";
            searchInput.oninput = filterHistoryComments;
        }
        if (filterSelect) {
            filterSelect.value = "all";
            filterSelect.onchange = filterHistoryComments;
        }

        /*========================================
                    IMAGES RECOVERY & RENDER
        ========================================*/
        let resolvedChart = data.analysis.chart;
        if (!resolvedChart && data.analysis.summary) {
            resolvedChart = await generateHistoricalChartDataUrl(data.analysis.summary);
            if (currentActiveHistoryId !== targetId || historyViewSequence !== thisRequestId) {
                return;
            }
        }

        if (resolvedChart && resolvedChart.startsWith("data:image")) {
            chartImg.src = resolvedChart;
            chartImg.style.display = "block";
            if (chartLoader) chartLoader.style.display = "none";
            if (chartFallback) chartFallback.style.display = "none";

            // If chart was missing in DB and newly generated, safely backfill chart only
            if (!data.analysis.chart) {
                fetch("/save-images", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        analysis_id: targetId,
                        chart_image: resolvedChart
                    })
                }).catch(e => console.warn("Chart backfill save failed:", e));
            }
        } else {
            if (chartLoader) chartLoader.style.display = "none";
            if (chartImg) chartImg.style.display = "none";
            if (chartFallback) chartFallback.style.display = "flex";
        }

        // WORD CLOUD: Display existing immediately if present, and regenerate with current configuration
        const existingWordCloud = data.analysis.wordcloud;
        if (existingWordCloud && existingWordCloud.startsWith("data:image")) {
            wordImg.src = existingWordCloud;
            wordImg.style.display = "block";
            if (wordLoader) wordLoader.style.display = "none";
            if (wordFallback) wordFallback.style.display = "none";
        }

        const cacheKey = `${targetId}_${WORDCLOUD_MAX_WORDS}`;
        const hasComments = Array.isArray(data.analysis.comments) && data.analysis.comments.length > 0;
        const alreadyRegenerated = regeneratedHistoryWordClouds.has(cacheKey);

        if (hasComments && !alreadyRegenerated) {
            // If no existing image is currently shown, ensure loader is visible
            if (!existingWordCloud || !existingWordCloud.startsWith("data:image")) {
                if (wordLoader) wordLoader.style.display = "flex";
                if (wordImg) wordImg.style.display = "none";
                if (wordFallback) wordFallback.style.display = "none";
            }

            try {
                const newWordCloud = await generateHistoricalWordCloudDataUrl(data.analysis.comments);

                // Race condition check: ensure modal has not moved to another record
                if (currentActiveHistoryId !== targetId || historyViewSequence !== thisRequestId) {
                    return;
                }

                if (newWordCloud && newWordCloud.startsWith("data:image")) {
                    // Update in-memory reference
                    regeneratedHistoryWordClouds.add(cacheKey);
                    if (currentHistoryAnalysis && currentHistoryAnalysis.id === targetId) {
                        currentHistoryAnalysis.wordcloud = newWordCloud;
                    }

                    // Display newly generated word cloud
                    wordImg.src = newWordCloud;
                    wordImg.style.display = "block";
                    if (wordLoader) wordLoader.style.display = "none";
                    if (wordFallback) wordFallback.style.display = "none";

                    // Safely persist via /save-images (partial update: only wordcloud_image, sentiment chart is untouched)
                    fetch("/save-images", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            analysis_id: targetId,
                            wordcloud_image: newWordCloud
                        })
                    }).catch(err => console.warn("Failed to persist regenerated word cloud:", err));
                } else {
                    // Generation returned null: fallback to existing image if available, else show fallback placeholder
                    if (!existingWordCloud || !existingWordCloud.startsWith("data:image")) {
                        if (wordLoader) wordLoader.style.display = "none";
                        if (wordImg) wordImg.style.display = "none";
                        if (wordFallback) wordFallback.style.display = "flex";
                    }
                }
            } catch (genErr) {
                console.error("Word cloud regeneration failed:", genErr);
                if (!existingWordCloud || !existingWordCloud.startsWith("data:image")) {
                    if (wordLoader) wordLoader.style.display = "none";
                    if (wordImg) wordImg.style.display = "none";
                    if (wordFallback) wordFallback.style.display = "flex";
                }
            }
        } else if (!existingWordCloud || !existingWordCloud.startsWith("data:image")) {
            // No comments to regenerate from and no existing image saved
            if (wordLoader) wordLoader.style.display = "none";
            if (wordImg) wordImg.style.display = "none";
            if (wordFallback) wordFallback.style.display = "flex";
        }

    } catch (e) {
        if (currentActiveHistoryId === targetId && historyViewSequence === thisRequestId) {
            console.error("Error opening history analysis:", e);
            showToast("Error loading saved analysis.", "error");
            if (chartLoader) chartLoader.style.display = "none";
            if (chartFallback) chartFallback.style.display = "flex";
            if (wordLoader) wordLoader.style.display = "none";
            if (wordFallback) wordFallback.style.display = "flex";
        }
    }
}


let historyComments = [];

function renderHistoryComments(comments) {

    const list =
        document.getElementById("historyCommentList");

    list.innerHTML = "";

    if (comments.length === 0) {

        list.innerHTML = `
            <div class="no-comments">
                No comments found.
            </div>
        `;

        return;
    }

    comments.forEach(item => {

        list.innerHTML += `

        <div class="history-comment">

            <div class="history-comment-text">

                ${item.comment}

            </div>

            <div class="history-comment-badge ${item.sentiment.toLowerCase()}">

                ${item.sentiment}

            </div>

        </div>

        `;

    });

}

function filterHistoryComments() {

    const search = document.getElementById("historyCommentSearch").value.toLowerCase();
    const filter = document.getElementById("historyCommentFilter").value;
    const filtered = historyComments.filter(item => {
        const matchText = item.comment.toLowerCase().includes(search);
        const matchSentiment = filter === "all" || item.sentiment === filter;
        return matchText && matchSentiment;

    });

    renderHistoryComments(filtered);

}


// Close Modal
function closeHistoryModal() {
    currentActiveHistoryId = null;
    document.getElementById("historyModal").style.display = "none";
}


function exportHistoryAnalysis() {

    if (!currentHistoryAnalysis) {
        return;
    }

    exportData = structuredClone(currentHistoryAnalysis);
    document.getElementById("exportVideo").textContent = exportData.video_title;
    document.getElementById("exportComments").textContent = exportData.comments.length;
    document.getElementById("exportStatus").textContent = "Loaded From History";
    document.querySelectorAll(".export-btn").forEach(btn => {
        btn.disabled = false;
    });

    closeHistoryModal();

    openTab("export");

}


function showDeleteModal(id) {
    deleteAnalysisId = id;
    document.getElementById("deleteHistoryModal").style.display = "flex";
}

function closeDeleteModal() {
    deleteAnalysisId = null;
    document.getElementById("deleteHistoryModal").style.display = "none";
}

async function confirmDeleteAnalysis() {

    const response = await fetch(

        `/delete-analysis/${deleteAnalysisId}`,

        {
            method: "DELETE"
        }

    );

    const result = await response.json();

    if (result.success) {
        closeDeleteModal();
        closeHistoryModal();
        loadHistory();
    }

}

// ================= CLEAR HISTORY =================

const clearHistoryModal = document.getElementById("clearHistoryModal");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
if (clearHistoryBtn) {
    clearHistoryBtn.onclick = function () {
        if (clearHistoryModal) clearHistoryModal.style.display = "flex";
    };
}

const cancelClearHistory = document.getElementById("cancelClearHistory");
if (cancelClearHistory) {
    cancelClearHistory.onclick = function () {
        if (clearHistoryModal) clearHistoryModal.style.display = "none";
    };
}

window.addEventListener("click", function (e) {
    if (clearHistoryModal && e.target === clearHistoryModal) {
        clearHistoryModal.style.display = "none";
    }
});

const confirmClearHistory = document.getElementById("confirmClearHistory");
if (confirmClearHistory) {
    confirmClearHistory.onclick = async function () {
        if (typeof historyData !== 'undefined' && historyData.length === 0) {
            if (clearHistoryModal) clearHistoryModal.style.display = "none";
            showToast("No history found.", "warning");
            return;
        }

        try {
            const response = await fetch("/clear-history", {
                method: "DELETE"
            });
            const result = await response.json();
            if (clearHistoryModal) clearHistoryModal.style.display = "none";

            if (result.success) {
                if (typeof historyData !== 'undefined') historyData = [];
                if (typeof loadHistory === 'function') loadHistory();
                showToast("History cleared successfully.", "success");
            } else {
                showToast("Failed to clear history.", "error");
            }
        } catch (err) {
            if (clearHistoryModal) clearHistoryModal.style.display = "none";
            showToast("Failed to clear history.", "error");
        }
    };
}


// =========================================================================
//                   YOUTUBE VIDEO COMPARATOR LOGIC
// =========================================================================

function initComparison() {
    const form = document.getElementById("compareForm");
    if (!form) return;

    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const urlA = document.getElementById("compareUrlA").value.trim();
        const urlB = document.getElementById("compareUrlB").value.trim();

        if (!urlA || !urlB) {
            showToast("Please provide YouTube URLs for both Video A and Video B.", "warning");
            return;
        }

        if (urlA.toLowerCase() === urlB.toLowerCase()) {
            showToast("Please enter two different YouTube video URLs to compare.", "warning");
            return;
        }

        const formData = new FormData(form);
        const messageInterval = showPreloader(COMPARISON_PRELOADER_MESSAGES);

        const submitBtn = document.getElementById("compareSubmitBtn");
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Comparing Videos...';
        }

        fetch("/compare", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                clearInterval(messageInterval);
                hidePreloader();

                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-balance-scale"></i> Compare Videos';
                }

                if (!data.success) {
                    showToast(data.message || "Failed to compare videos.", "error");
                    return;
                }

                compareData = data;
                renderComparisonResults(data);
                showToast("Video comparison completed!", "success");
            })
            .catch(err => {
                clearInterval(messageInterval);
                hidePreloader();
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-balance-scale"></i> Compare Videos';
                }
                console.error("Comparison error:", err);
                showToast("An error occurred while comparing videos. Please verify the URLs.", "error");
            });
    });
}

function renderComparisonResults(data) {
    const vA = data.video_a;
    const vB = data.video_b;
    const comp = data.comparison;

    // Toggle containers
    const emptyState = document.getElementById("compareEmptyState");
    const resultsContainer = document.getElementById("compareResultsContainer");
    if (emptyState) emptyState.style.display = "none";
    if (resultsContainer) {
        resultsContainer.style.display = "block";
        triggerGsapReveals(resultsContainer);
    }

    // Summary Banner
    const leadBadge = document.getElementById("compareLeadBadge");
    const diffPill = document.getElementById("compareDiffPill");
    const summaryText = document.getElementById("compareSummaryText");

    if (comp.lead_video === "Tied") {
        leadBadge.innerHTML = '<i class="fas fa-handshake"></i> Result: Tied Sentiment';
        diffPill.textContent = "0.0% Difference";
    } else {
        const shortTitle = comp.lead_title.length > 28 ? comp.lead_title.slice(0, 25) + "..." : comp.lead_title;
        leadBadge.innerHTML = `<i class="fas fa-trophy"></i> Lead: ${comp.lead_video} (${shortTitle})`;
        diffPill.textContent = `${Math.abs(comp.diff_positive)}% Positive Delta`;
    }
    summaryText.textContent = comp.summary;

    // Video A Profile Card
    document.getElementById("compTitleA").textContent = vA.title;
    const linkA = document.getElementById("compLinkA");
    if (linkA) { linkA.href = vA.url; }
    const thumbA = document.getElementById("compThumbA");
    if (thumbA) {
        thumbA.src = vA.thumbnail;
        thumbA.style.display = "block";
    }
    animateNumber(document.getElementById("compTotalA"), vA.total_comments);
    document.getElementById("compPosA").textContent = vA.positive;
    document.getElementById("compPosPctA").textContent = vA.pct_positive + "%";
    document.getElementById("compNegA").textContent = vA.negative;
    document.getElementById("compNegPctA").textContent = vA.pct_negative + "%";
    document.getElementById("compNeuA").textContent = vA.neutral;
    document.getElementById("compNeuPctA").textContent = vA.pct_neutral + "%";

    // Video B Profile Card
    document.getElementById("compTitleB").textContent = vB.title;
    const linkB = document.getElementById("compLinkB");
    if (linkB) { linkB.href = vB.url; }
    const thumbB = document.getElementById("compThumbB");
    if (thumbB) {
        thumbB.src = vB.thumbnail;
        thumbB.style.display = "block";
    }
    animateNumber(document.getElementById("compTotalB"), vB.total_comments);
    document.getElementById("compPosB").textContent = vB.positive;
    document.getElementById("compPosPctB").textContent = vB.pct_positive + "%";
    document.getElementById("compNegB").textContent = vB.negative;
    document.getElementById("compNegPctB").textContent = vB.pct_negative + "%";
    document.getElementById("compNeuB").textContent = vB.neutral;
    document.getElementById("compNeuPctB").textContent = vB.pct_neutral + "%";

    // Visualizations
    renderCompareGroupedChart(vA, vB);
    renderCompareDonutCharts(vA, vB);

    // Metrics Table
    renderCompareMetricsTable(vA, vB, comp);

    // Comments Dual List
    renderCompareComments(vA.sample_comments, vB.sample_comments);

    // Smooth scroll down to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
}

function renderCompareGroupedChart(vA, vB) {
    const canvas = document.getElementById("compareGroupedCanvas");
    if (!canvas) return;

    if (compGroupedChart) {
        compGroupedChart.destroy();
    }

    const isDark = document.body.classList.contains("dark-mode");
    const textColor = isDark ? "#e5e7eb" : "#374151";
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)";

    const labelA = vA.title.length > 22 ? vA.title.slice(0, 20) + "..." : vA.title;
    const labelB = vB.title.length > 22 ? vB.title.slice(0, 20) + "..." : vB.title;

    compGroupedChart = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: ["Positive (%)", "Negative (%)", "Neutral (%)"],
            datasets: [
                {
                    label: `Video A (${labelA})`,
                    data: [vA.pct_positive, vA.pct_negative, vA.pct_neutral],
                    backgroundColor: "#8b5cf6",
                    borderColor: "#7c3aed",
                    borderWidth: 1,
                    borderRadius: 8
                },
                {
                    label: `Video B (${labelB})`,
                    data: [vB.pct_positive, vB.pct_negative, vB.pct_neutral],
                    backgroundColor: "#06b6d4",
                    borderColor: "#0891b2",
                    borderWidth: 1,
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: textColor, font: { family: 'Inter', weight: 500 } }
                },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            return `${ctx.dataset.label}: ${ctx.raw}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: textColor, font: { family: 'Inter' } },
                    grid: { color: gridColor }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: textColor,
                        font: { family: 'Inter' },
                        callback: val => val + "%"
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

function renderCompareDonutCharts(vA, vB) {
    const canvasA = document.getElementById("compDonutA");
    const canvasB = document.getElementById("compDonutB");
    if (!canvasA || !canvasB) return;

    if (compDonutAChart) compDonutAChart.destroy();
    if (compDonutBChart) compDonutBChart.destroy();

    const isDark = document.body.classList.contains("dark-mode");
    const textColor = isDark ? "#e5e7eb" : "#374151";

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom",
                labels: { color: textColor, font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 12 }
            }
        },
        cutout: "65%"
    };

    compDonutAChart = new Chart(canvasA.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Positive", "Negative", "Neutral"],
            datasets: [{
                data: [vA.positive, vA.negative, vA.neutral],
                backgroundColor: ["#10b981", "#ef4444", "#f59e0b"],
                borderWidth: 0
            }]
        },
        options: commonOptions
    });

    compDonutBChart = new Chart(canvasB.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Positive", "Negative", "Neutral"],
            datasets: [{
                data: [vB.positive, vB.negative, vB.neutral],
                backgroundColor: ["#10b981", "#ef4444", "#f59e0b"],
                borderWidth: 0
            }]
        },
        options: commonOptions
    });
}

function renderCompareMetricsTable(vA, vB, comp) {
    const tbody = document.getElementById("compareTableBody");
    if (!tbody) return;

    const diffComments = vA.total_comments - vB.total_comments;
    const diffPosSign = comp.diff_positive > 0 ? "+" : "";
    const diffNegSign = comp.diff_negative > 0 ? "+" : "";
    const diffNeuSign = comp.diff_neutral > 0 ? "+" : "";

    const dominantA = vA.pct_positive >= vA.pct_negative && vA.pct_positive >= vA.pct_neutral ? "Positive" :
        vA.pct_negative >= vA.pct_neutral ? "Negative" : "Neutral";
    const dominantB = vB.pct_positive >= vB.pct_negative && vB.pct_positive >= vB.pct_neutral ? "Positive" :
        vB.pct_negative >= vB.pct_neutral ? "Negative" : "Neutral";

    tbody.innerHTML = `
        <tr>
            <td><strong><i class="fas fa-comments"></i> Total Comments Analyzed</strong></td>
            <td>${vA.total_comments.toLocaleString()}</td>
            <td>${vB.total_comments.toLocaleString()}</td>
            <td><span class="diff-tag ${diffComments >= 0 ? 'pos' : 'neg'}">${diffComments > 0 ? '+' : ''}${diffComments}</span></td>
        </tr>
        <tr>
            <td><strong><i class="fas fa-smile" style="color: #10b981;"></i> Positive Sentiment</strong></td>
            <td>${vA.positive} (${vA.pct_positive}%)</td>
            <td>${vB.positive} (${vB.pct_positive}%)</td>
            <td><span class="diff-tag ${comp.diff_positive >= 0 ? 'pos' : 'neg'}">${diffPosSign}${comp.diff_positive}%</span></td>
        </tr>
        <tr>
            <td><strong><i class="fas fa-frown" style="color: #ef4444;"></i> Negative Sentiment</strong></td>
            <td>${vA.negative} (${vA.pct_negative}%)</td>
            <td>${vB.negative} (${vB.pct_negative}%)</td>
            <td><span class="diff-tag ${comp.diff_negative <= 0 ? 'pos' : 'neg'}">${diffNegSign}${comp.diff_negative}%</span></td>
        </tr>
        <tr>
            <td><strong><i class="fas fa-meh" style="color: #f59e0b;"></i> Neutral Sentiment</strong></td>
            <td>${vA.neutral} (${vA.pct_neutral}%)</td>
            <td>${vB.neutral} (${vB.pct_neutral}%)</td>
            <td><span class="diff-tag neutral">${diffNeuSign}${comp.diff_neutral}%</span></td>
        </tr>
        <tr>
            <td><strong><i class="fas fa-crown" style="color: #8b5cf6;"></i> Dominant Sentiment</strong></td>
            <td><span class="sentiment-badge badge-${dominantA.toLowerCase()}">${dominantA}</span></td>
            <td><span class="sentiment-badge badge-${dominantB.toLowerCase()}">${dominantB}</span></td>
            <td><strong>${comp.lead_video} Lead</strong></td>
        </tr>
    `;
}

function renderCompareComments(commentsA, commentsB) {
    const listA = document.getElementById("compCommentsListA");
    const listB = document.getElementById("compCommentsListB");
    if (!listA || !listB) return;

    listA.innerHTML = "";
    listB.innerHTML = "";

    if (!commentsA || commentsA.length === 0) {
        listA.innerHTML = '<div class="no-comments"><p>No comments available</p></div>';
    } else {
        commentsA.forEach(item => {
            listA.innerHTML += `
                <div class="comp-comment-item">
                    <span class="sentiment-badge badge-${item.sentiment.toLowerCase()}">${item.sentiment}</span>
                    <p class="comp-comment-text">${item.comment}</p>
                </div>
            `;
        });
    }

    if (!commentsB || commentsB.length === 0) {
        listB.innerHTML = '<div class="no-comments"><p>No comments available</p></div>';
    } else {
        commentsB.forEach(item => {
            listB.innerHTML += `
                <div class="comp-comment-item">
                    <span class="sentiment-badge badge-${item.sentiment.toLowerCase()}">${item.sentiment}</span>
                    <p class="comp-comment-text">${item.comment}</p>
                </div>
            `;
        });
    }
}

function resetComparisonForm() {
    const form = document.getElementById("compareForm");
    if (form) form.reset();
    const resultsContainer = document.getElementById("compareResultsContainer");
    const emptyState = document.getElementById("compareEmptyState");
    if (resultsContainer) resultsContainer.style.display = "none";
    if (emptyState) emptyState.style.display = "flex";
    compareData = null;
    showToast("Comparison cleared.", "info");
}

function exportComparisonJSON() {
    if (!compareData) {
        showToast("Please run a video comparison first.", "warning");
        return;
    }
    const jsonStr = JSON.stringify(compareData, null, 4);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "YouTube_Video_Comparison.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function exportComparisonCSV() {
    if (!compareData) {
        showToast("Please run a video comparison first.", "warning");
        return;
    }
    const vA = compareData.video_a;
    const vB = compareData.video_b;
    const comp = compareData.comparison;

    let csv = "Metric,Video A,Video B,Difference\n";
    csv += `"Title","${vA.title.replace(/"/g, '""')}","${vB.title.replace(/"/g, '""')}",""\n`;
    csv += `"URL","${vA.url}","${vB.url}",""\n`;
    csv += `"Total Comments",${vA.total_comments},${vB.total_comments},${vA.total_comments - vB.total_comments}\n`;
    csv += `"Positive Comments",${vA.positive},${vB.positive},${comp.diff_positive}%\n`;
    csv += `"Negative Comments",${vA.negative},${vB.negative},${comp.diff_negative}%\n`;
    csv += `"Neutral Comments",${vA.neutral},${vB.neutral},${comp.diff_neutral}%\n`;
    csv += `"Positive Percentage",${vA.pct_positive}%,${vB.pct_positive}%,${comp.diff_positive}%\n`;
    csv += `"Negative Percentage",${vA.pct_negative}%,${vB.pct_negative}%,${comp.diff_negative}%\n`;
    csv += `"Neutral Percentage",${vA.pct_neutral}%,${vB.pct_neutral}%,${comp.diff_neutral}%\n`;
    csv += `"Dominant Lead","","","${comp.lead_video}"\n`;

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "YouTube_Video_Comparison.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function exportComparisonTXT() {
    if (!compareData) {
        showToast("Please run a video comparison first.", "warning");
        return;
    }
    const vA = compareData.video_a;
    const vB = compareData.video_b;
    const comp = compareData.comparison;

    let txt = "========================================================\n";
    txt += "       YOUTUBE VIDEO SENTIMENT COMPARISON REPORT        \n";
    txt += "========================================================\n\n";
    txt += "SUMMARY:\n";
    txt += comp.summary + "\n\n";
    txt += "--------------------------------------------------------\n";
    txt += "VIDEO A:\n";
    txt += `Title: ${vA.title}\n`;
    txt += `URL: ${vA.url}\n`;
    txt += `Total Comments: ${vA.total_comments}\n`;
    txt += `Positive: ${vA.positive} (${vA.pct_positive}%)\n`;
    txt += `Negative: ${vA.negative} (${vA.pct_negative}%)\n`;
    txt += `Neutral:  ${vA.neutral} (${vA.pct_neutral}%)\n\n`;
    txt += "--------------------------------------------------------\n";
    txt += "VIDEO B:\n";
    txt += `Title: ${vB.title}\n`;
    txt += `URL: ${vB.url}\n`;
    txt += `Total Comments: ${vB.total_comments}\n`;
    txt += `Positive: ${vB.positive} (${vB.pct_positive}%)\n`;
    txt += `Negative: ${vB.negative} (${vB.pct_negative}%)\n`;
    txt += `Neutral:  ${vB.neutral} (${vB.pct_neutral}%)\n\n`;
    txt += "--------------------------------------------------------\n";
    txt += "DELTA ANALYSIS:\n";
    txt += `Positive Margin: ${comp.diff_positive > 0 ? '+' : ''}${comp.diff_positive}%\n`;
    txt += `Negative Margin: ${comp.diff_negative > 0 ? '+' : ''}${comp.diff_negative}%\n`;
    txt += `Neutral Margin:  ${comp.diff_neutral > 0 ? '+' : ''}${comp.diff_neutral}%\n`;
    txt += `Dominant Positive Lead: ${comp.lead_video} (${comp.lead_title})\n\n`;

    const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "YouTube_Video_Comparison_Report.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


// =========================================================================
//                   ANALYTICS DASHBOARD LOGIC
// =========================================================================

async function loadDashboard() {
    try {
        const response = await fetch("/dashboard-data");
        if (response.status !== 200) {
            renderDashboardEmpty();
            return;
        }

        const data = await response.json();
        window.dashData = data;

        // Update Stat Cards with animated counters
        animateNumber(document.getElementById("dashTotalAnalyses"), data.total_analyses || 0);
        animateNumber(document.getElementById("dashTotalComments"), data.total_comments || 0);

        const avgPosEl = document.getElementById("dashAvgPositive");
        if (avgPosEl) {
            avgPosEl.textContent = (data.avg_positive || 0) + "%";
        }

        const healthEl = document.getElementById("dashHealthScore");
        if (healthEl) {
            if (!data.authenticated || data.total_analyses === 0) {
                healthEl.textContent = "--";
                healthEl.style.color = "inherit";
            } else if (data.avg_positive >= 65) {
                healthEl.textContent = "Positive (" + data.avg_positive + "%)";
                healthEl.style.color = "#10b981";
            } else if (data.avg_positive >= 45) {
                healthEl.textContent = "Balanced (" + data.avg_positive + "%)";
                healthEl.style.color = "#3b82f6";
            } else {
                healthEl.textContent = "Mixed (" + data.avg_positive + "%)";
                healthEl.style.color = "#f59e0b";
            }
        }

        // Recent analyses list
        const recentList = document.getElementById("dashRecentList");
        const emptyState = document.getElementById("dashEmptyState");
        const chartEmpty = document.getElementById("dashChartEmpty");
        const chartContainer = document.getElementById("dashChartContainer");

        if (!data.recent_analyses || data.recent_analyses.length === 0) {
            renderDashboardEmpty();
            return;
        }

        if (emptyState) emptyState.style.display = "none";
        if (chartEmpty) chartEmpty.style.display = "none";
        if (chartContainer) chartContainer.style.display = "block";

        if (recentList) {
            recentList.innerHTML = "";
            data.recent_analyses.forEach(item => {
                const thumb = item.thumbnail ? `<img src="${item.thumbnail}" alt="Thumbnail" class="dash-thumb" onerror="this.style.display='none'">` : '<div class="dash-thumb-fallback"><i class="fab fa-youtube"></i></div>';
                recentList.innerHTML += `
                    <div class="dash-recent-row">
                        <div class="dash-recent-thumb-col">
                            ${thumb}
                        </div>
                        <div class="dash-recent-info">
                            <h4 class="dash-recent-title" title="${item.title}">${item.title}</h4>
                            <div class="dash-recent-meta">
                                <span><i class="far fa-calendar-alt"></i> ${item.date}</span>
                                <span><i class="far fa-comment-dots"></i> ${item.total_comments} comments</span>
                            </div>
                        </div>
                        <div class="dash-recent-row-right">
                            <div class="dash-recent-stat">
                                <span class="dash-pos-tag"><i class="fas fa-smile"></i> ${item.pct_positive}%</span>
                            </div>
                            <div class="dash-recent-actions">
                                <button class="dash-row-btn" onclick="viewAnalysis(${item.id})">
                                    <i class="fas fa-eye"></i> View
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
        }

        // Render Aggregate Sentiment Donut Chart
        renderDashAggregateChart(data.total_positive || 0, data.total_negative || 0, data.total_neutral || 0);

    } catch (e) {
        console.error("Dashboard error:", e);
        renderDashboardEmpty();
    }
}

function renderDashboardEmpty() {
    const emptyState = document.getElementById("dashEmptyState");
    const recentList = document.getElementById("dashRecentList");
    const chartEmpty = document.getElementById("dashChartEmpty");
    const chartContainer = document.getElementById("dashChartContainer");

    if (emptyState) emptyState.style.display = "flex";
    if (recentList) recentList.innerHTML = "";
    if (chartEmpty) chartEmpty.style.display = "flex";
    if (chartContainer) chartContainer.style.display = "none";
}

function renderDashAggregateChart(pos, neg, neu) {
    const canvas = document.getElementById("dashAggregateChart");
    if (!canvas) return;

    if (dashAggregateChart) dashAggregateChart.destroy();

    const isDark = document.body.classList.contains("dark-mode");
    const textColor = isDark ? "#e5e7eb" : "#374151";

    dashAggregateChart = new Chart(canvas.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Positive", "Negative", "Neutral"],
            datasets: [{
                data: [pos, neg, neu],
                backgroundColor: ["#10b981", "#ef4444", "#f59e0b"],
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: textColor,
                        font: { family: "Inter", size: 11, weight: "500" },
                        padding: 12,
                        boxWidth: 12
                    }
                }
            },
            cutout: "60%"
        }
    });
}


