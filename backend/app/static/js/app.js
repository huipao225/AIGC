(function () {
    var form = document.getElementById("detect-form");
    var textInput = document.getElementById("text-input");
    var charCount = document.getElementById("char-count");
    var resultsDiv = document.getElementById("results");
    var clearBtn = document.getElementById("clear-btn");
    var submitBtn = document.getElementById("submit-btn");
    var loading = document.getElementById("loading");

    // Tab elements
    var tabText = document.getElementById("tab-text");
    var tabFile = document.getElementById("tab-file");
    var textPanel = document.getElementById("text-panel");
    var filePanel = document.getElementById("file-panel");

    // File elements
    var dropZone = document.getElementById("drop-zone");
    var fileInput = document.getElementById("file-input");
    var filePreview = document.getElementById("file-preview");
    var fileName = document.getElementById("file-name");
    var fileSize = document.getElementById("file-size");
    var fileRemove = document.getElementById("file-remove");

    var currentMode = "text"; // "text" or "file"
    var selectedFile = null;

    // === Tab switching ===
    tabText.addEventListener("click", function () {
        currentMode = "text";
        tabText.classList.add("active");
        tabFile.classList.remove("active");
        textPanel.classList.remove("hidden");
        filePanel.classList.add("hidden");
        resultsDiv.innerHTML = "";
    });

    tabFile.addEventListener("click", function () {
        currentMode = "file";
        tabFile.classList.add("active");
        tabText.classList.remove("active");
        filePanel.classList.remove("hidden");
        textPanel.classList.add("hidden");
        resultsDiv.innerHTML = "";
    });

    // === Character counter ===
    textInput.addEventListener("input", function () {
        var len = this.value.length;
        var max = parseInt(this.getAttribute("maxlength"), 10);
        charCount.textContent = len.toLocaleString() + " / " + max.toLocaleString();
        if (len > max * 0.9) {
            charCount.classList.add("text-red-400");
        } else {
            charCount.classList.remove("text-red-400");
        }
    });

    // === Clear button ===
    clearBtn.addEventListener("click", function () {
        textInput.value = "";
        charCount.textContent = "0 / 50,000";
        charCount.classList.remove("text-red-400");
        resultsDiv.innerHTML = "";
        textInput.focus();
    });

    // === File upload: click ===
    dropZone.addEventListener("click", function () {
        fileInput.click();
    });

    fileInput.addEventListener("change", function () {
        handleFileSelect(this.files[0]);
    });

    // === File upload: drag & drop ===
    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        handleFileSelect(e.dataTransfer.files[0]);
    });

    function handleFileSelect(file) {
        if (!file) return;
        var allowed = [".txt", ".docx", ".pdf"];
        var ext = "." + file.name.split(".").pop().toLowerCase();
        if (allowed.indexOf(ext) === -1) {
            alert("不支持的文件格式，请选择 .txt / .docx / .pdf 文件");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert("文件大小不能超过 10MB");
            return;
        }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        filePreview.classList.remove("hidden");
        dropZone.classList.add("hidden");
    }

    fileRemove.addEventListener("click", function () {
        selectedFile = null;
        fileInput.value = "";
        filePreview.classList.add("hidden");
        dropZone.classList.remove("hidden");
    });

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // === Form submission ===
    form.addEventListener("submit", function (evt) {
        evt.preventDefault();

        if (currentMode === "file" && !selectedFile) {
            alert("请先选择要上传的文件");
            return;
        }
        if (currentMode === "text" && !textInput.value.trim()) {
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = "检测中...";
        loading.classList.remove("hidden");
        resultsDiv.innerHTML = "";

        var request;
        if (currentMode === "text") {
            request = fetch("/api/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: textInput.value.trim() }),
            });
        } else {
            var formData = new FormData();
            formData.append("file", selectedFile);
            request = fetch("/api/detect/file", {
                method: "POST",
                body: formData,
            });
        }

        request
            .then(function (resp) {
                return resp.json().then(function (data) {
                    return { status: resp.status, body: data };
                });
            })
            .then(function (result) {
                renderResult(result.body, result.status);
            })
            .catch(function () {
                resultsDiv.innerHTML =
                    '<div class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">网络错误，请重试。</div>';
            })
            .finally(function () {
                submitBtn.disabled = false;
                submitBtn.textContent = "开始检测";
                loading.classList.add("hidden");
            });
    });

    // === Result rendering ===
    function renderResult(payload, statusCode) {
        if (statusCode !== 200 || payload.status === "error") {
            var err = payload.error || {};
            resultsDiv.innerHTML =
                '<div class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">' +
                '<p class="font-medium">错误: ' +
                escapeHtml(err.code || "未知") +
                "</p>" +
                "<p>" +
                escapeHtml(err.message || "发生未知错误") +
                "</p>" +
                "</div>";
            return;
        }

        var d = payload.data;
        var scorePercent = Math.round(d.overall_score * 100);
        var isAI = d.classification === "AI-generated";
        var barColor =
            scorePercent > 70
                ? "bg-red-500"
                : scorePercent > 50
                ? "bg-orange-400"
                : "bg-green-500";
        var badgeColor = isAI
            ? "bg-red-100 text-red-700"
            : "bg-green-100 text-green-700";
        var badgeText = isAI ? "AI 生成" : "人类写作";
        var confidenceText = isAI ? "置信度" : "可信度";
        var scoreLabel = isAI ? "AI 生成概率" : "人类写作概率";

        var html =
            '<div class="result-card bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-5">';

        // Overall result
        html += '<div class="text-center">';
        html +=
            '<span class="inline-block px-3 py-1 rounded-full text-sm font-medium ' +
            badgeColor +
            '">' +
            badgeText +
            "</span>";
        html +=
            '<p class="text-4xl font-bold mt-3 ' +
            (isAI ? "text-red-600" : "text-green-600") +
            '">' +
            scorePercent +
            "%</p>";
        html +=
            '<p class="text-sm text-gray-400">' +
            confidenceText +
            ": " +
            Math.round(d.confidence * 100) +
            "%</p>";
        html += "</div>";

        // Score bar
        html += '<div class="bg-gray-200 rounded-full h-3 overflow-hidden">';
        html +=
            '<div class="' +
            barColor +
            ' h-full rounded-full score-bar" style="width: ' +
            scorePercent +
            '%"></div>';
        html +=
            '<div class="flex justify-between text-xs text-gray-400 mt-1"><span>人类</span><span>AI</span></div>';
        html += "</div>";

        // Breakdown cards — 4 columns
        html += '<div class="grid grid-cols-4 gap-2 text-sm">';
        html +=
            '<div class="bg-gray-50 rounded-lg p-2 text-center"><p class="text-gray-500 text-xs">RoBERTa</p><p class="font-semibold text-sm">' +
            Math.round(d.breakdown.roberta.score * 100) +
            "%</p></div>";
        html +=
            '<div class="bg-gray-50 rounded-lg p-2 text-center"><p class="text-gray-500 text-xs">困惑度</p><p class="font-semibold text-sm">' +
            (d.breakdown.perplexity ? d.breakdown.perplexity.perplexity : "-") +
            "</p></div>";
        html +=
            '<div class="bg-gray-50 rounded-lg p-2 text-center"><p class="text-gray-500 text-xs">句长变异</p><p class="font-semibold text-sm">' +
            d.breakdown.burstiness.cv.toFixed(3) +
            "</p></div>";
        html +=
            '<div class="bg-gray-50 rounded-lg p-2 text-center"><p class="text-gray-500 text-xs">中文特征</p><p class="font-semibold text-sm">' +
            (d.breakdown.chinese_features
                ? Math.round(d.breakdown.chinese_features.score * 100) + "%"
                : "-") +
            "</p></div>";
        html += "</div>";

        // Chinese feature details
        if (d.breakdown.chinese_features) {
            var cf = d.breakdown.chinese_features;
            html +=
                '<div class="bg-amber-50 rounded-lg p-3 text-xs space-y-1">' +
                '<p class="font-medium text-amber-800">中文语言特征分析</p>' +
                '<div class="grid grid-cols-2 gap-1 text-amber-700">' +
                "<span>连接词密度: " + cf.marker_density.toFixed(2) + " /句</span>" +
                "<span>开篇模式词: " + (cf.opener_ratio * 100).toFixed(0) + "%</span>" +
                "<span>术语密度: " + cf.buzzword_density.toFixed(2) + " /百字</span>" +
                "<span>段落均句数: " + cf.sentences_per_para.toFixed(1) + "</span>" +
                "<span>论证标记: " + cf.evidence_density.toFixed(2) + " /句</span>" +
                "<span>连词密度: " + cf.conj_density.toFixed(2) + " /句</span>" +
                "</div>" +
                '<p class="text-amber-600 mt-1 font-medium">' +
                (cf.score > 0.6
                    ? "强烈提示：文本呈现典型的 AI 写作用语特征（高频连接词 + 模式化开篇 + 术语密集 + 论证标记多）"
                    : cf.score > 0.4
                    ? "提示：文本存在一定程度的 AI 写作特征"
                    : "文本在语言特征上较接近人类写作风格") +
                "</p></div>";
        }

        // Per-segment analysis
        if (d.segments.length > 1) {
            html +=
                '<div class="space-y-2"><p class="text-sm text-gray-500">分段分析（共 ' +
                d.segments.length +
                " 段）</p>";
            for (var i = 0; i < d.segments.length; i++) {
                var seg = d.segments[i];
                var segPct = Math.round(seg.score * 100);
                var segColor =
                    segPct > 70
                        ? "bg-red-400"
                        : segPct > 50
                        ? "bg-orange-300"
                        : "bg-green-400";
                html +=
                    '<div class="flex items-center gap-2"><span class="text-xs text-gray-400 w-8">#' +
                    (i + 1) +
                    '</span><div class="bg-gray-100 rounded-full h-2 flex-1"><div class="' +
                    segColor +
                    ' h-full rounded-full" style="width:' +
                    segPct +
                    '%"></div></div><span class="text-xs w-10 text-right">' +
                    segPct +
                    "%</span></div>";
            }
            html += "</div>";
        }

        // Metadata
        html +=
            '<div class="flex justify-between text-xs text-gray-400 pt-2 border-t border-gray-100">';
        html +=
            "<span>文本长度: " +
            d.metadata.text_length.toLocaleString() +
            " 字符</span>";
        html +=
            "<span>分析耗时: " + d.metadata.processing_time_ms + " ms</span>";
        html += "<span>分析段落: " + d.metadata.chunks_analyzed + " 段</span>";
        html += "</div>";

        html += "</div>";
        resultsDiv.innerHTML = html;
        resultsDiv.scrollIntoView({ behavior: "smooth" });
    }

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
})();
