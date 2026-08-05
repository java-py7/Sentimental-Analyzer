const b = document.body;
function toggleMode() {
  b.classList.toggle('dark-mode');
   const themeIcon = document.querySelector(".darklight-ModeToggle");

    if (document.body.classList.contains("dark-mode")) {
        themeIcon.classList.remove("fa-sun");
        themeIcon.classList.add("fa-moon");
    } else {
        themeIcon.classList.remove("fa-moon");
        themeIcon.classList.add("fa-sun");
    }
  m = b.classList.contains('dark') ? 'dark' : 'light';
}

const menu = document.querySelector(".menu");
const toggle = document.querySelector(".toggle");
const menuIcon = document.querySelector(".menu-icon");

toggle.addEventListener("click", () => {
    menu.classList.toggle("active");

    if (menu.classList.contains("active")) {
        menuIcon.classList.remove("fa-bars");
        menuIcon.classList.add("fa-times");
    } else {
        menuIcon.classList.remove("fa-times");
        menuIcon.classList.add("fa-bars");
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

if(params.get("login")==="success"){
    showToast("Logged in successfully.", "success");
    window.history.replaceState({}, "", window.location.pathname);
}

if(params.get("logout")==="success"){
    showToast("Logged out successfully.", "success");
    window.history.replaceState({}, "", window.location.pathname);
}

if(params.get("toast")==="login_cancelled"){
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

function showPreloader() {
    const preloader = document.getElementById('preloader');
    const textElement = document.getElementById('preloader-text');
    const subtextElement = document.getElementById('preloader-subtext');

    preloader.classList.add('active');
    document.body.style.overflow = 'hidden';

    let messageIndex = 0;

    // Start with first message
    if (PRELOADER_MESSAGES[0]) {
        textElement.textContent = PRELOADER_MESSAGES[0].text;
        subtextElement.textContent = PRELOADER_MESSAGES[0].subtext;
        messageIndex = 1;
    }

    // Change messages during preloader
    const messageInterval = setInterval(() => {
        if (messageIndex < PRELOADER_MESSAGES.length) {
            textElement.textContent = PRELOADER_MESSAGES[messageIndex].text;
            subtextElement.textContent = PRELOADER_MESSAGES[messageIndex].subtext;
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

document.querySelectorAll(".export-btn").forEach(btn=>{
    btn.disabled = true;
});

// =============== FORM SUBMISSION ===============
document.addEventListener("DOMContentLoaded", () => {

    document.addEventListener("submit", function (e) {

        if (!e.target.matches('form[action="/analyze"]')) return;

        e.preventDefault();

        const form = e.target;
        const formData = new FormData(form);

        if (!formData.get("video_url")) {
            showToast("Please enter a YouTube URL","warning");
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
                showToast(data.message);
                return;
            }

            window.chartData = data.summary;
            window.commentData = data.comments;
            exportData = data;

            document.getElementById("resultsContainer").style.display = "block";

            document.getElementById("results-title").style.display = "block";

            const analysisVideo = document.getElementById("analysisVideo");

            analysisVideo.textContent = data.video_url;
            analysisVideo.href = data.video_url;
            analysisVideo.target = "_blank";

            renderSummary(data.summary);
            renderComments(data.comments);
            renderSentimentChart();
            renderWordCloud(data.comments);

            // ========================================
            // Export Page
            // ========================================

            document.getElementById("exportVideo").textContent =
                data.video_title;

            document.getElementById("exportComments").textContent =
                data.comments.length;

            document.getElementById("exportStatus").textContent =
                "Ready";

            document.querySelectorAll(".export-btn").forEach(btn=>{
                btn.disabled = false;
            });

            openTab("home");

            form.reset();

        })
        .catch(err => {

            clearInterval(messageInterval);
            hidePreloader();

            console.error(err);

            alert("Something went wrong.");

        });

    });

});

// =============== TAB FUNCTIONALITY ===============
function openTab(tabName) {

    document.querySelectorAll(".tab-content").forEach(tab => {
        tab.classList.remove("active");
    });

    const currentTab = document.getElementById(tabName);

    if(currentTab){
        currentTab.classList.add("active");
    }

    // Bottom navigation active state
    document.querySelectorAll(".tab-button").forEach(btn=>{
        btn.classList.remove("active");
    });

    const bottomBtn =
        document.querySelector(`.tab-button[onclick="openTab('${tabName}')"]`);

    if(bottomBtn){
        bottomBtn.classList.add("active");
    }

    // Close radial menu
    menu.classList.remove("active");
    menuIcon.classList.remove("fa-times");
    menuIcon.classList.add("fa-bars");

    switch(tabName){

        case "history":

            if(!isLoggedIn){
                showToast("Please sign in to access your analysis history.","warning");
                openTab("about");
                return;
            }
            loadHistory();
            break;
        case "export":
            // load exports later
            break;

        case "settings":
            // load settings later
            break;

        case "youtube":
            // load shortcuts later
            break;

        case "about":
            // nothing for now
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
                    "#4facfe",
                    "#ff4d4d",
                    "#43e97b"
                ],

                borderRadius: 12
            }]
        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                }

            },

            scales: {

                y: {
                    beginAtZero: true
                }

            }

        }

    });

}

document.addEventListener("DOMContentLoaded", () => {

    renderSentimentChart();

});


function renderSummary(summary){

    document.querySelector(".total-comments").textContent =
        summary.positive + summary.negative + summary.neutral;

    document.querySelector(".positive-card .stat-number").textContent =
        summary.positive;

    document.querySelector(".negative-card .stat-number").textContent =
        summary.negative;

    document.querySelector(".neutral-card .stat-number").textContent =
        summary.neutral;
}


// =============== COMMENTS ===============

let allComments = [];

function renderComments(comments){

    allComments = comments;

    const search = document.getElementById("commentSearch");
    const filter = document.getElementById("sentimentFilter");

    if(search) search.value = "";
    if(filter) filter.value = "all";

    displayComments(comments);

    if(search){
        search.oninput = filterComments;
    }

    if(filter){
        filter.onchange = filterComments;
    }

}

function displayComments(comments){

    const list = document.querySelector(".comments-list");

    if(!list) return;

    list.innerHTML = "";

    if(comments.length === 0){

        list.innerHTML = `

            <div class="no-comments">

                <i class="far fa-folder-open"></i>

                <p>No comments found.</p>

            </div>

        `;

        return;

    }

    comments.forEach(item => {

        list.innerHTML += `

            <div class="comment-item">

                <div class="comment-text">

                    ${item.comment}

                </div>

                <div class="sentiment-badge badge-${item.sentiment.toLowerCase()}">

                    ${item.sentiment}

                </div>

            </div>

        `;

    });

}

function filterComments(){

    const searchInput = document.getElementById("commentSearch");
    const filterSelect = document.getElementById("sentimentFilter");

    if(!searchInput || !filterSelect) return;

    const search = searchInput.value.trim().toLowerCase();

    const filter = filterSelect.value;

    const filtered = allComments.filter(item => {

        const matchText =
            item.comment.toLowerCase().includes(search);

        const matchSentiment =
            filter === "all" ||
            item.sentiment === filter;

        return matchText && matchSentiment;

    });

    displayComments(filtered);

}


function renderWordCloud(comments){

    const canvas = document.getElementById("wordCloud");

    if(!canvas) return;

    const stopWords = new Set([
        "the","a","an","and","or","to","of","is","are","was","were",
        "i","you","he","she","it","they","we","this","that","in","on",
        "for","with","my","your","our","their","be","have","has","had",
        "at","as","from","by","but","if","so","not","me","do","did",
        "does","can","will","just","very"
    ]);

    const words = {};

    comments.forEach(item=>{

        item.comment
            .toLowerCase()
            .replace(/[^\w\s]/g,"")
            .split(/\s+/)
            .forEach(word=>{

                if(word.length < 3) return;

                if(stopWords.has(word)) return;

                words[word] = (words[word] || 0) + 1;

            });

    });

    const list = Object.entries(words);

    const ctx = canvas.getContext("2d");

    ctx.clearRect(0,0,canvas.width,canvas.height);

    WordCloud(canvas,{

        list,

        gridSize:10,

        weightFactor:12,

        rotateRatio:0.35,

        backgroundColor:"transparent",

        drawOutOfBound:false,

        shuffle:true

    });

    document.getElementById("wordCloud").style.display = "block";
    document.getElementById("wordCloudEmpty").style.display = "none";

    setTimeout(()=>{

        saveAnalysisImages();

    },1000);

}

async function saveAnalysisImages(){

    if(!exportData.database_id){
        return;
    }

    const chartCanvas=document.getElementById("sentimentCanvas");
    const wordCanvas=document.getElementById("wordCloud");

    if(!chartCanvas||!wordCanvas){
        return;
    }

    const response=await fetch("/save-images",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({

            analysis_id:exportData.database_id,

            chart_image:chartCanvas.toDataURL("image/png"),

            wordcloud_image:wordCanvas.toDataURL("image/png")

        })

    });

    const result=await response.json();

    console.log(result);

}



/*============================================
                EXPORT DATA
============================================*/

let exportData = null;

/*============================================
            EXPORT FUNCTIONS
============================================*/

function downloadTXT(){

    if(!exportData){

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

    exportData.comments.forEach((item,index)=>{

        txt += `${index+1}. [${item.sentiment}]\n`;
        txt += item.comment + "\n\n";

    });

    const blob = new Blob([txt],{
        type:"text/plain;charset=utf-8"
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


function downloadCSV(){

    if(!exportData){

        alert("Please analyze a YouTube video first.");

        return;

    }

    let csv = "";

    csv += "Sentiment,Comment\n";

    exportData.comments.forEach(item=>{

        const comment = item.comment
            .replace(/"/g,'""')
            .replace(/\n/g," ");

        csv += `"${item.sentiment}","${comment}"\n`;

    });

    const blob = new Blob(
        [csv],
        {
            type:"text/csv;charset=utf-8;"
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


function downloadJSON(){

    if(!exportData){

        alert("Please analyze a YouTube video first.");

        return;

    }

    const json = JSON.stringify(exportData,null,4);

    const blob = new Blob(
        [json],
        {
            type:"application/json"
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


function downloadChart(){

    const link = document.createElement("a");

    link.download = "Sentiment_Chart.png";

    if(exportData && exportData.chart){

        link.href = exportData.chart;

    }else{

        if(!window.chartData){

            alert("Please analyze a YouTube video first.");

            return;

        }

        const canvas = document.getElementById("sentimentCanvas");

        if(!canvas){

            alert("Chart not found.");

            return;

        }

        link.href = canvas.toDataURL("image/png");

    }

    link.click();

}


function downloadWordCloud(){

    const link = document.createElement("a");

    link.download = "Word_Cloud.png";

    if(exportData && exportData.wordcloud){

        link.href = exportData.wordcloud;

    }else{

        const canvas = document.getElementById("wordCloud");

        if(!canvas){

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
async function downloadPDF(){
    if(!exportData){
        alert("Please analyze a YouTube video first.");
        return;
    }
    const doc = {
        pageSize:"A4",
        pageMargins:[40,45,40,45],
        content:[
            {
                text:"YouTube Sentiment Analysis Report",
                style:"title",
                alignment:"center"
            },
            {
                canvas:[
                    {
                        type:"line",
                        x1:0,
                        y1:0,
                        x2:515,
                        y2:0,
                        lineWidth:1.5,
                        lineColor:"#7c3aed"
                    }
                ],
                margin:[0,12,0,18]
            },
            {
                table:{
                    widths:["*"],
                    body:[
                        [
                            {
                                stack:[
                                    {
                                        columns:[
                                            { text:"Video Title:", width:80, bold:true, color:"#7c3aed" },
                                            { text:exportData.video_title, width:"*" }
                                        ],
                                        margin:[0,0,0,6]
                                    },
                                    {
                                        columns:[
                                            { text:"Video URL:", width:80, bold:true, color:"#7c3aed" },
                                            { text:exportData.video_url, color:"blue", width:"*" }
                                        ],
                                        margin:[0,0,0,6]
                                    },
                                    {
                                        columns:[
                                            { text:"Generated:", width:80, bold:true, color:"#7c3aed" },
                                            { text:new Date().toLocaleString(), width:"*" }
                                        ]
                                    }
                                ],
                                fillColor:"#f8fafc",
                                margin:[12,12,12,12]
                            }
                        ]
                    ]
                },
                layout:{
                    hLineWidth:function(){ return 0.5; },
                    vLineWidth:function(){ return 0.5; },
                    hLineColor:function(){ return "#e2e8f0"; },
                    vLineColor:function(){ return "#e2e8f0"; }
                },
                margin:[0,0,0,15]
            },
            {
                text:"Analysis Summary",
                style:"section"
            },
            {
                columns:[
                    {
                        width:"23%",
                        table:{
                            widths:["*"],
                            body:[
                                [
                                    {
                                        text:[
                                            { text:"POSITIVE\n", fontSize:9, bold:true, color:"#ffffff" },
                                            { text:exportData.summary.positive.toString(), fontSize:20, bold:true, color:"#ffffff" }
                                        ],
                                        fillColor:"#22c55e",
                                        alignment:"center",
                                        margin:[0,10,0,10]
                                    }
                                ]
                            ]
                        },
                        layout:"noBorders"
                    },
                    { width:"2.6%", text:"" },
                    {
                        width:"23%",
                        table:{
                            widths:["*"],
                            body:[
                                [
                                    {
                                        text:[
                                            { text:"NEUTRAL\n", fontSize:9, bold:true, color:"#ffffff" },
                                            { text:exportData.summary.neutral.toString(), fontSize:20, bold:true, color:"#ffffff" }
                                        ],
                                        fillColor:"#3b82f6",
                                        alignment:"center",
                                        margin:[0,10,0,10]
                                    }
                                ]
                            ]
                        },
                        layout:"noBorders"
                    },
                    { width:"2.6%", text:"" },
                    {
                        width:"23%",
                        table:{
                            widths:["*"],
                            body:[
                                [
                                    {
                                        text:[
                                            { text:"NEGATIVE\n", fontSize:9, bold:true, color:"#ffffff" },
                                            { text:exportData.summary.negative.toString(), fontSize:20, bold:true, color:"#ffffff" }
                                        ],
                                        fillColor:"#ef4444",
                                        alignment:"center",
                                        margin:[0,10,0,10]
                                    }
                                ]
                            ]
                        },
                        layout:"noBorders"
                    },
                    { width:"2.6%", text:"" },
                    {
                        width:"23%",
                        table:{
                            widths:["*"],
                            body:[
                                [
                                    {
                                        text:[
                                            { text:"TOTAL\n", fontSize:9, bold:true, color:"#ffffff" },
                                            { text:exportData.comments.length.toString(), fontSize:20, bold:true, color:"#ffffff" }
                                        ],
                                        fillColor:"#7c3aed",
                                        alignment:"center",
                                        margin:[0,10,0,10]
                                    }
                                ]
                            ]
                        },
                        layout:"noBorders"
                    }
                ],
                margin:[0,0,0,20]
            },
            {
                text:"Sentiment Chart",
                style:"section"
            },
            {
                image:document.getElementById("sentimentCanvas").toDataURL("image/png"),
                width:480,
                alignment:"center",
                margin:[0,5,0,15]
            },
            {
                text:"Word Cloud",
                style:"section"
            },
            {
                image:document.getElementById("wordCloud").toDataURL("image/png"),
                width:480,
                alignment:"center",
                margin:[0,5,0,15]
            },
            {
                text:"Comments Analysis",
                style:"section",
                pageBreak:"before"
            },
            {
                table:{
                    headerRows:1,
                    widths:[35,75,"*"],
                    body:[
                        [
                            { text:"No.", style:"tableHeader" },
                            { text:"Sentiment", style:"tableHeader" },
                            { text:"Comment", style:"tableHeader" }
                        ],
                        ...exportData.comments.map((item,index)=>{
                            let sentimentColor="#3b82f6";
                            if(item.sentiment==="Positive"){
                                sentimentColor="#22c55e";
                            }
                            else if(item.sentiment==="Negative"){
                                sentimentColor="#ef4444";
                            }
                            return [
                                {
                                    text:String(index+1),
                                    alignment:"center",
                                    margin:[0,5,0,5]
                                },
                                {
                                    text:item.sentiment,
                                    bold:true,
                                    color:"white",
                                    fillColor:sentimentColor,
                                    alignment:"center",
                                    margin:[0,5,0,5]
                                },
                                {
                                    text:item.comment,
                                    margin:[5,5,5,5]
                                }
                            ];
                        })
                    ]
                },
                layout:{
                    fillColor:function(row){
                        if(row===0){
                            return "#111827";
                        }
                        return row % 2 === 0 ? "#f9fafb" : null;
                    },
                    hLineColor:function(){ return "#e5e7eb"; },
                    vLineColor:function(){ return "#e5e7eb"; },
                    hLineWidth:function(){ return 0.5; },
                    vLineWidth:function(){ return 0.5; }
                }
            }
        ],
        styles:{
            title:{
                fontSize:22,
                bold:true,
                color:"#111827"
            },
            heading:{
                fontSize:12,
                bold:true,
                color:"#7c3aed"
            },
            section:{
                fontSize:16,
                bold:true,
                color:"#1f2937",
                margin:[0,15,0,8]
            },
            tableHeader:{
                bold:true,
                color:"white",
                alignment:"center",
                fontSize:11,
                margin:[0,6,0,6]
            }
        }
    };
    doc.footer = function(currentPage,pageCount){
        return{
            columns:[
                {
                    text:"YouTube Sentiment Analyzer",
                    margin:[40,0]
                },
                {
                    text:new Date().toLocaleDateString(),
                    alignment:"center"
                },
                {
                    text:"Page "+currentPage+" of "+pageCount,
                    alignment:"right",
                    margin:[0,0,40,0]
                }
            ],
            fontSize:9,
            color:"#9ca3af"
        };
    };
    pdfMake.createPdf(doc).download("YouTube_Sentiment_Report.pdf");
}


function searchYouTube(){
    const input = document.getElementById("youtubeSearchInput");
    const query = input.value.trim();
    
    if(!query){
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
document.getElementById("historySearch").addEventListener("input", function () {

    const value = this.value.toLowerCase();

    const filtered = historyData.filter(item =>

        item.title.toLowerCase().includes(value) ||
        item.video_id.toLowerCase().includes(value)

    );

    renderHistory(filtered);

});

// Sort
document.getElementById("historySort").addEventListener("change", function () {

    let sorted = [...historyData];

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

// View Analysis
async function viewAnalysis(id){

    const response = await fetch(`/get-analysis/${id}`);

    const data = await response.json();

    if(!data.success){
        return;
    }

    currentHistoryAnalysis = data.analysis;

    /*========================================
                HEADER
    ========================================*/

    document.getElementById("historyModalTitle").textContent = data.analysis.video_title;

    const videoLink = document.getElementById("historyVideoLink");

    videoLink.href = data.analysis.video_url;
    videoLink.textContent = data.analysis.video_url;

    /*========================================
                SUMMARY
    ========================================*/

    document.getElementById("historyPositive").textContent = data.analysis.summary.positive;
    document.getElementById("historyNegative").textContent = data.analysis.summary.negative;
    document.getElementById("historyNeutral").textContent = data.analysis.summary.neutral;
    document.getElementById("historyComments").textContent = data.analysis.comments.length;

    /*========================================
                IMAGES
    ========================================*/

    document.getElementById("historyChart").src =
        data.analysis.chart;

    document.getElementById("historyWordCloud").src =
        data.analysis.wordcloud;

    /*========================================
                COMMENTS
    ========================================*/

    historyComments = data.analysis.comments;
    renderHistoryComments(historyComments);
    document.getElementById("historyCommentSearch").value = "";
    document.getElementById("historyCommentFilter").value = "all";
    document.getElementById("historyCommentSearch").oninput = filterHistoryComments;
    document.getElementById("historyCommentFilter").onchange = filterHistoryComments;
    document.getElementById("historyModal").style.display = "flex";

}


let historyComments = [];

function renderHistoryComments(comments){

    const list =
        document.getElementById("historyCommentList");

    list.innerHTML = "";

    if(comments.length===0){

        list.innerHTML=`
            <div class="no-comments">
                No comments found.
            </div>
        `;

        return;
    }

    comments.forEach(item=>{

        list.innerHTML+=`

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

function filterHistoryComments(){

    const search = document.getElementById("historyCommentSearch").value.toLowerCase();
    const filter = document.getElementById("historyCommentFilter").value;
    const filtered = historyComments.filter(item=>{
        const matchText = item.comment.toLowerCase().includes(search);
        const matchSentiment = filter==="all" || item.sentiment===filter;
        return matchText && matchSentiment;

    });

    renderHistoryComments(filtered);

}


// Close Modal
function closeHistoryModal() {

    document.getElementById("historyModal").style.display = "none";

}


function exportHistoryAnalysis(){

    if(!currentHistoryAnalysis){
        return;
    }

    exportData = structuredClone(currentHistoryAnalysis);
    document.getElementById("exportVideo").textContent = exportData.video_title;
    document.getElementById("exportComments").textContent = exportData.comments.length;
    document.getElementById("exportStatus").textContent = "Loaded From History";
    document.querySelectorAll(".export-btn").forEach(btn=>{
        btn.disabled=false;
    });

    closeHistoryModal();

    openTab("export");

}


function showDeleteModal(id){
    deleteAnalysisId = id;
    document.getElementById("deleteHistoryModal").style.display="flex";
}

function closeDeleteModal(){
    deleteAnalysisId = null;
    document.getElementById("deleteHistoryModal").style.display="none";
}

async function confirmDeleteAnalysis(){

    const response = await fetch(

        `/delete-analysis/${deleteAnalysisId}`,

        {
            method:"DELETE"
        }

    );

    const result = await response.json();

    if(result.success){
        closeDeleteModal();
        closeHistoryModal();
        loadHistory();
    }

}

// ================= CLEAR HISTORY =================

const clearHistoryModal = document.getElementById("clearHistoryModal");

document.getElementById("clearHistoryBtn").onclick = function(){
    clearHistoryModal.style.display = "flex";
};

document.getElementById("cancelClearHistory").onclick = function(){
    clearHistoryModal.style.display = "none";
};

window.addEventListener("click", function(e){

    if(e.target === clearHistoryModal){
        clearHistoryModal.style.display = "none";
    }

});

document.getElementById("confirmClearHistory").onclick = async function(){

    if(historyData.length === 0){
        clearHistoryModal.style.display = "none";
        showToast("No history found.", "warning");
        return;
    }

    const response = await fetch("/clear-history", {

        method: "DELETE"

    });

    const result = await response.json();

    clearHistoryModal.style.display = "none";

    if(result.success){
        historyData = [];
        loadHistory();
        showToast("History cleared successfully.", "success");
    }else{
        showToast("Failed to clear history.", "error");
    }

};

        
