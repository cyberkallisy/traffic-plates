// =====================================================
// Traffic Plate Detector — Frontend Logic
// Image mode + Video mode (mode toggle in header)
// =====================================================

const $ = (id) => document.getElementById(id);

const state = {
    mode: "image",          // "image" | "video"
    file: null,
    busy: false,
};

const VEHICLE_EMOJI = {
    bicycle:    "🚲",
    car:        "🚗",
    motorcycle: "🏍️",
    bus:        "🚌",
    truck:      "🚚",
};

// --- Lightbox (click any plate crop or annotated image to zoom) ---
function setupLightbox() {
    const box = $("lightbox");
    const img = $("lightbox-img");
    const close = box.querySelector(".lightbox-close");

    // Delegate clicks on images with data-zoom
    document.addEventListener("click", (e) => {
        const target = e.target.closest("[data-zoom]");
        if (!target) return;
        img.src = target.dataset.zoom;
        box.classList.add("open");
        box.setAttribute("aria-hidden", "false");
    });

    const shut = () => {
        box.classList.remove("open");
        box.setAttribute("aria-hidden", "true");
        img.src = "";
    };
    box.addEventListener("click", shut);
    close.addEventListener("click", (e) => { e.stopPropagation(); shut(); });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && box.classList.contains("open")) shut();
    });
}

// --- Status check on load ---
async function checkHealth() {
    try {
        const r = await fetch("/health");
        const d = await r.json();
        $("status-dot").classList.add("ready");
        const label = (d.model || "model").split(/[/\\]/).pop();
        $("status-text").textContent =
            `Plates: ${label} · YOLO ${d.yolo_loaded ? "✓" : "…"} · OCR ${d.awiros_loaded ? "✓" : "…"}`;
    } catch (e) {
        $("status-dot").classList.add("error");
        $("status-text").textContent = "Server unreachable";
    }
}

// --- Mode toggle ---
function setupModeToggle() {
    document.querySelectorAll(".mode-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const m = btn.dataset.mode;
            if (m === state.mode) return;
            setMode(m);
        });
    });
}

function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".mode-btn").forEach((b) => {
        const active = b.dataset.mode === mode;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
    });
    $("image-upload-section").classList.toggle("hidden", mode !== "image");
    $("video-upload-section").classList.toggle("hidden", mode !== "video");
    hideResults();
    hideError();
}

// --- File selection (image) ---
function setupImageDropzone() {
    const dz = $("dropzone");
    const input = $("file-input");

    dz.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        input.click();
    });
    input.addEventListener("change", (e) => {
        if (e.target.files[0]) handleImageFile(e.target.files[0]);
    });
    ["dragenter", "dragover"].forEach((ev) => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            dz.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach((ev) => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            dz.classList.remove("dragover");
        });
    });
    dz.addEventListener("drop", (e) => {
        const f = e.dataTransfer.files[0];
        if (f) handleImageFile(f);
    });
    $("change-image-btn").addEventListener("click", (e) => { e.stopPropagation(); resetImage(); });
    $("reset-btn").addEventListener("click", resetImage);
    $("detect-btn").addEventListener("click", runImageDetection);
}

function handleImageFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("Please upload an image file (JPG, PNG, etc.)");
        return;
    }
    if (file.size > 200 * 1024 * 1024) {
        showError("File too large. Max 200 MB.");
        return;
    }
    state.file = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        $("preview").src = e.target.result;
        $("preview").classList.remove("hidden");
        $("dropzone-content").classList.add("hidden");
        $("change-image-btn").classList.remove("hidden");
    };
    reader.readAsDataURL(file);
    $("detect-btn").disabled = false;
    $("reset-btn").disabled = false;
    hideError();
    hideResults();
}

function resetImage() {
    state.file = null;
    $("file-input").value = "";
    $("preview").classList.add("hidden");
    $("dropzone-content").classList.remove("hidden");
    $("change-image-btn").classList.add("hidden");
    $("detect-btn").disabled = true;
    $("reset-btn").disabled = true;
    hideResults();
    hideError();
}

// --- File selection (video) ---
function setupVideoDropzone() {
    const dz = $("video-dropzone");
    const input = $("video-file-input");

    dz.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        input.click();
    });
    input.addEventListener("change", (e) => {
        if (e.target.files[0]) handleVideoFile(e.target.files[0]);
    });
    ["dragenter", "dragover"].forEach((ev) => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            dz.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach((ev) => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            dz.classList.remove("dragover");
        });
    });
    dz.addEventListener("drop", (e) => {
        const f = e.dataTransfer.files[0];
        if (f) handleVideoFile(f);
    });
    $("change-video-btn").addEventListener("click", (e) => { e.stopPropagation(); resetVideo(); });
    $("reset-video-btn").addEventListener("click", resetVideo);
    $("detect-video-btn").addEventListener("click", runVideoDetection);
}

function handleVideoFile(file) {
    if (!file.type.startsWith("video/")) {
        showError("Please upload a video file (MP4, MOV, AVI, etc.)");
        return;
    }
    if (file.size > 200 * 1024 * 1024) {
        showError("File too large. Max 200 MB.");
        return;
    }
    state.file = file;
    const url = URL.createObjectURL(file);
    const v = $("video-preview");
    v.src = url;
    v.classList.remove("hidden");
    $("video-dropzone-content").classList.add("hidden");
    $("change-video-btn").classList.remove("hidden");
    $("detect-video-btn").disabled = false;
    $("reset-video-btn").disabled = false;
    hideError();
    hideResults();
}

function resetVideo() {
    state.file = null;
    $("video-file-input").value = "";
    const v = $("video-preview");
    if (v.src) URL.revokeObjectURL(v.src);
    v.removeAttribute("src");
    v.load();
    v.classList.add("hidden");
    $("video-dropzone-content").classList.remove("hidden");
    $("change-video-btn").classList.add("hidden");
    $("detect-video-btn").disabled = true;
    $("reset-video-btn").disabled = true;
    hideResults();
    hideError();
}

// --- Detection (image) ---
async function runImageDetection() {
    if (!state.file || state.busy) return;
    state.busy = true;
    setBusy(true, "Running YOLO (plates + vehicles) + OCR pipeline…",
                  "Detecting plates + vehicles, then OCR on each crop.");
    hideError();
    hideResults();

    const form = new FormData();
    form.append("image", state.file);
    form.append("conf", "0.25");

    try {
        const r = await fetch("/api/detect", { method: "POST", body: form });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || `Server error ${r.status}`);
        renderImageResults(data);
    } catch (e) {
        showError(e.message || "Detection failed");
    } finally {
        state.busy = false;
        setBusy(false);
    }
}

// --- Detection (video) ---
async function runVideoDetection() {
    if (!state.file || state.busy) return;
    state.busy = true;
    setBusy(true,
        `Processing video with ${$("tracker-select").value}…`,
        "This can take 1–5 minutes depending on video length. Watch the server log for progress.");
    hideError();
    hideResults();

    const form = new FormData();
    form.append("video", state.file);
    form.append("tracker", $("tracker-select").value);
    const stride = parseInt($("frame-stride").value, 10);
    form.append("frame_stride", isNaN(stride) || stride < 1 ? "1" : String(stride));
    const maxFrames = parseInt($("max-frames").value, 10);
    form.append("max_frames", isNaN(maxFrames) || maxFrames <= 0 ? "" : String(maxFrames));
    form.append("write_video", $("write-video-toggle").checked ? "1" : "0");

    try {
        const r = await fetch("/api/detect_video", { method: "POST", body: form });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || `Server error ${r.status}`);
        renderVideoResults(data);
    } catch (e) {
        showError(e.message || "Video processing failed");
    } finally {
        state.busy = false;
        setBusy(false);
    }
}

function setBusy(busy, text, hint) {
    $("loading").classList.toggle("hidden", !busy);
    if (text) $("loading-text").textContent = text;
    if (hint) $("loading-hint").textContent = hint;
    $("detect-btn").disabled = busy || !state.file || state.mode !== "image";
    $("reset-btn").disabled = busy;
    $("detect-video-btn").disabled = busy || !state.file || state.mode !== "video";
    $("reset-video-btn").disabled = busy;
    document.querySelectorAll(".dropzone").forEach((d) => {
        d.style.pointerEvents = busy ? "none" : "";
    });
}

// ============ IMAGE RESULTS ============
function renderImageResults(d) {
    // Summary cards
    $("sum-total").textContent    = d.num_plates ?? 0;
    $("sum-valid").textContent    = d.num_valid ?? 0;
    $("sum-vehicles").textContent = d.num_vehicles ?? 0;
    $("sum-time").textContent     = `${(d.elapsed_seconds ?? 0).toFixed(2)}s`;

    // Annotated image — click to zoom
    const ann = $("annotated-img");
    if (d.annotated_url) {
        ann.src = d.annotated_url + "?t=" + Date.now();
        ann.setAttribute("data-zoom", d.annotated_url);
    } else {
        ann.removeAttribute("src");
        ann.removeAttribute("data-zoom");
    }

    // Plates list
    const list = $("plates-list");
    list.innerHTML = "";
    const plates = d.plates || [];
    $("plates-count-hint").textContent = plates.length ? `· ${plates.length} found` : "";
    if (plates.length === 0) {
        list.innerHTML = '<p class="empty">No plates detected. Try a clearer image with visible plates.</p>';
    } else {
        plates.forEach((p) => list.appendChild(renderPlateCard(p)));
    }

    // Vehicle breakdown chips
    const breakdown = $("vehicle-breakdown");
    breakdown.innerHTML = "";
    if (d.vehicle_counts && Object.keys(d.vehicle_counts).length) {
        const present = Object.entries(d.vehicle_counts).filter(([, n]) => n > 0);
        if (present.length > 0) {
            present.forEach(([cls, n]) => {
                const chip = document.createElement("span");
                chip.className = `vehicle-chip ${cls}`;
                const emoji = VEHICLE_EMOJI[cls] || "🚙";
                chip.innerHTML = `${emoji} ${cls} <b>${n}</b>`;
                breakdown.appendChild(chip);
            });
            breakdown.classList.remove("hidden");
        } else {
            breakdown.classList.add("hidden");
        }
    } else {
        breakdown.classList.add("hidden");
    }

    // Vehicles list
    const vlist = $("vehicles-list");
    vlist.innerHTML = "";
    const vehicles = d.vehicles || [];
    $("vehicles-count-hint").textContent = vehicles.length ? `· ${vehicles.length} found` : "";
    if (vehicles.length === 0) {
        vlist.innerHTML = '<p class="empty">No vehicles detected.</p>';
    } else {
        vehicles.forEach((v, i) => vlist.appendChild(renderVehicleCard(v, i)));
    }

    $("results").classList.remove("hidden");
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPlateCard(p) {
    const card = document.createElement("div");
    card.className = "plate-card " + (p.valid_format ? "valid" : "invalid");

    // Crop thumbnail (clickable for lightbox)
    const crop = document.createElement("img");
    crop.className = "plate-crop";
    crop.alt = p.text || "plate";
    if (p.crop_url) {
        crop.src = p.crop_url + "?t=" + Date.now();
        crop.setAttribute("data-zoom", p.crop_url);
        crop.title = "Click to zoom";
    }
    card.appendChild(crop);

    // Info column
    const info = document.createElement("div");
    info.className = "plate-info";

    const text = document.createElement("div");
    text.className = "plate-text";
    text.textContent = p.text || "(unreadable)";
    info.appendChild(text);

    const meta = document.createElement("div");
    meta.className = "plate-meta";
    const confDet = document.createElement("span");
    confDet.className = "det";
    confDet.textContent = `Det ${Math.round((p.detection_confidence || 0) * 100)}%`;
    meta.appendChild(confDet);
    if (p.ocr_confidence > 0) {
        const confOcr = document.createElement("span");
        confOcr.className = "ocr";
        confOcr.textContent = `OCR ${Math.round(p.ocr_confidence * 100)}%`;
        meta.appendChild(confOcr);
    }
    if (p.state_code) {
        const st = document.createElement("span");
        st.className = "state";
        st.textContent = p.state_code;
        meta.appendChild(st);
    }
    if (p.format_type && p.format_type !== "standard") {
        const fmt = document.createElement("span");
        fmt.textContent = p.format_type;
        meta.appendChild(fmt);
    }
    if (p.engine) {
        const eng = document.createElement("span");
        eng.textContent = p.engine;
        eng.title = "OCR engine used for this plate";
        meta.appendChild(eng);
    }
    info.appendChild(meta);
    card.appendChild(info);

    // Raw OCR line (when different from final text)
    if (p.raw_ocr && p.raw_ocr !== p.text) {
        const raw = document.createElement("div");
        raw.className = "plate-raw";
        const fixNote = (p.fixes_applied && p.fixes_applied.length)
            ? ` <span class="fixes">[fixed: ${escapeHtml(p.fixes_applied.join(", "))}]</span>` : "";
        raw.innerHTML = `OCR raw: <code>${escapeHtml(p.raw_ocr)}</code>${fixNote}`;
        card.appendChild(raw);
    }

    // Badge (always last on the right)
    const badge = document.createElement("div");
    badge.className = "badge " + (p.valid_format ? "valid" : "invalid");
    badge.textContent = p.valid_format ? "VALID" : "INVALID";
    card.appendChild(badge);

    return card;
}

// ============ VIDEO RESULTS TABS ============
//
// Two full-width tabs below the Annotated Video:
//   1) "Detected Number Plates" — per-track cards with OCR voting + clickable
//      plate text (opens track-modal showing every frame the plate was seen)
//   2) "Detected Vehicles" — one summary card per unique vehicle (one vehicle
//      per tracked plate), each clickable to open the same frame-modal
function setupVideoResultsTabs() {
    document.querySelectorAll(".results-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            const t = btn.dataset.tab;
            if (!t) return;
            // Toggle tab buttons
            document.querySelectorAll(".results-tab").forEach((b) => {
                const active = b.dataset.tab === t;
                b.classList.toggle("active", active);
                b.setAttribute("aria-selected", active ? "true" : "false");
            });
            // Toggle tab panels
            document.querySelectorAll(".tab-panel").forEach((p) => {
                p.classList.toggle("active", p.dataset.tab === t);
            });
        });
    });
}

// ============ VIDEO RESULTS ============
function renderVideoResults(d) {
    $("vsum-tracks").textContent = d.n_tracks ?? 0;
    $("vsum-valid").textContent  = d.n_valid_plates ?? 0;
    $("vsum-frames").textContent = `${d.n_frames_processed ?? 0}/${d.n_total_frames ?? 0}`;
    $("vsum-time").textContent   = `${(d.elapsed_seconds ?? 0).toFixed(2)}s`;

    // Annotated video
    const v = $("annotated-video");
    if (d.annotated_video_url) {
        v.src = d.annotated_video_url + "?t=" + Date.now();
        v.classList.remove("hidden");
        $("video-download").href = d.annotated_video_url;
        $("video-download").classList.remove("hidden");
    } else {
        v.removeAttribute("src");
        v.classList.add("hidden");
        $("video-download").classList.add("hidden");
    }
    if (d.report_url) {
        $("report-link").href = d.report_url;
        $("report-link").classList.remove("hidden");
    } else {
        $("report-link").classList.add("hidden");
    }
    if (d.tracks_json_url) {
        $("tracks-json-link").href = d.tracks_json_url;
        $("tracks-json-link").classList.remove("hidden");
    } else {
        $("tracks-json-link").classList.add("hidden");
    }

    // Meta strip
    const meta = $("video-meta-strip");
    meta.innerHTML = "";
    const chips = [
        ["Tracker",      d.tracker],
        ["Stride",       d.stride],
        ["Processed",    `${d.n_frames_processed ?? 0} / ${d.n_total_frames ?? 0}`],
        ["Pipeline FPS", d.fps_processed ? d.fps_processed.toFixed(2) : "—"],
    ];
    chips.forEach(([k, val]) => {
        if (val === undefined || val === null || val === "") return;
        const chip = document.createElement("span");
        chip.className = "vmeta-chip";
        chip.innerHTML = `${k}: <b>${escapeHtml(String(val))}</b>`;
        meta.appendChild(chip);
    });

    // Tracks list (PLATES tab — full width)
    const list = $("video-tracks-list");
    list.innerHTML = "";
    const tracks = d.tracks || [];
    if (tracks.length === 0) {
        list.innerHTML = '<p class="empty">No plates detected in this video. Try a clip with clearer plates.</p>';
    } else {
        tracks.forEach((t) => list.appendChild(renderTrackCard(t)));
    }

    // Vehicles list (VEHICLES tab — full width)
    renderVideoVehicles(tracks);

    // Reset to PLATES tab so users see plate cards first (matches user's request
    // "show detected number plates, then detected vehicle numbers below")
    resetVideoResultsTabs();

    $("video-results").classList.remove("hidden");
    $("video-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// Reset tabs to "plates" active on each new result
function resetVideoResultsTabs() {
    document.querySelectorAll(".results-tab").forEach((b) => {
        const active = b.dataset.tab === "plates";
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.dataset.tab === "plates");
    });
}

// Render vehicles tab — one card per track (= one vehicle per unique plate)
function renderVideoVehicles(tracks) {
    const list = $("video-vehicles-list");
    list.innerHTML = "";
    if (!tracks || tracks.length === 0) {
        list.innerHTML = '<p class="empty">No vehicles detected in this video.</p>';
        return;
    }
    tracks.forEach((t) => list.appendChild(renderVideoVehicleCard(t)));
}

function renderVideoVehicleCard(t) {
    const card = document.createElement("div");
    card.className = "vveh-card " + (t.valid_indian ? "valid" : "invalid");

    // Head: vehicle ID + plate number (clickable) + badge
    const head = document.createElement("div");
    head.className = "vveh-head";

    const idBox = document.createElement("div");
    idBox.className = "vveh-id";
    idBox.textContent = `Vehicle #${t.track_id}`;
    head.appendChild(idBox);

    const plate = document.createElement("div");
    plate.className = "vveh-plate";
    plate.textContent = t.final_text || "(unreadable)";
    plate.title = `Click to see all ${t.n_frames} frames where vehicle #${t.track_id} was tracked`;
    plate.addEventListener("click", () => openTrackDetails(t));
    head.appendChild(plate);

    const badge = document.createElement("div");
    badge.className = "badge " + (t.valid_indian ? "valid" : "invalid");
    badge.textContent = t.valid_indian ? "VALID" : "INVALID";
    head.appendChild(badge);

    card.appendChild(head);

    // ── Detected Vehicles tab: show the SOURCE FRAME images (full annotated
    //    video frames where the plate was detected) — not just OCR crops.
    //    This is what the user wants: "the frames from which the plate was
    //    cropped" alongside the plate number. Click any image to zoom in
    //    the lightbox; click the plate text to open the per-frame modal. ─
    if (t.best_annotated_url) {
        const frameRow = document.createElement("div");
        frameRow.className = "vveh-frames";

        const srcImg = document.createElement("img");
        srcImg.className = "vveh-frame-img";
        srcImg.src = t.best_annotated_url + "?t=" + Date.now();
        srcImg.alt = `Frame #${t.best_frame} where vehicle #${t.track_id} was tracked`;
        srcImg.title = "Click to zoom — actual video frame where this plate was detected";
        srcImg.setAttribute("data-zoom", t.best_annotated_url);
        srcImg.loading = "lazy";
        frameRow.appendChild(srcImg);

        const frameLabel = document.createElement("div");
        frameLabel.className = "vveh-frame-label";
        frameLabel.innerHTML =
            `<b>📷 Frame #${t.best_frame ?? "?"}</b> — the actual video frame this plate was cropped from ` +
            `<span class="hint">(bbox around plate · click image to zoom)</span>`;
        frameRow.appendChild(frameLabel);

        card.appendChild(frameRow);
    } else if (t.best_crop_url) {
        // Fallback: no source frame available (old data) — show OCR crop
        const cropRow = document.createElement("div");
        cropRow.className = "vveh-crop";
        const cropImg = document.createElement("img");
        cropImg.className = "vveh-crop-img";
        cropImg.src = t.best_crop_url + "?t=" + Date.now();
        cropImg.alt = `OCR crop for vehicle #${t.track_id}`;
        cropImg.title = "Click to zoom — close-up crop that OCR read";
        cropImg.setAttribute("data-zoom", t.best_crop_url);
        cropRow.appendChild(cropImg);
        const cropLabel = document.createElement("div");
        cropLabel.className = "vveh-crop-label";
        cropLabel.innerHTML = `<b>🔍 OCR crop</b> — ${Math.round((t.best_conf || 0) * 100)}%`;
        cropRow.appendChild(cropLabel);
        card.appendChild(cropRow);
    }

    // Stats line
    const stats = document.createElement("div");
    stats.className = "vveh-stats";
    stats.innerHTML = [
        `<span><b>${t.n_frames ?? 0}</b>&nbsp;frames seen</span>`,
        `<span>first→last: <b>${t.first_seen ?? "?"}→${t.last_seen ?? "?"}</b></span>`,
        `<span>YOLO: <b>${Math.round((t.avg_yolo_conf || 0) * 100)}%</b></span>`,
        `<span>OCR: <b>${Math.round((t.final_conf || 0) * 100)}%</b></span>`,
        `<span>best frame: <b>#${t.best_frame ?? "?"}</b></span>`,
    ].join("");
    card.appendChild(stats);

    // Hint that plate is clickable
    const hint = document.createElement("div");
    hint.className = "vveh-hint";
    hint.textContent = "👆 click plate number or image → all frames for this vehicle";
    card.appendChild(hint);

    return card;
}

function renderTrackCard(t) {
    const card = document.createElement("div");
    card.className = "track-card " + (t.valid_indian ? "valid" : "invalid");

    // Head: ID box + plate text + badge
    const head = document.createElement("div");
    head.className = "track-head";

    const idBox = document.createElement("div");
    idBox.className = "track-id-box";
    idBox.textContent = `#${t.track_id}`;
    head.appendChild(idBox);

    const plate = document.createElement("div");
    plate.className = "plate-text plate-text-clickable";
    plate.textContent = t.final_text || "(unreadable)";
    plate.title = `Click to see all ${t.n_frames} frames where this plate was tracked`;
    plate.addEventListener("click", () => openTrackDetails(t));
    head.appendChild(plate);

    const badge = document.createElement("div");
    badge.className = "badge " + (t.valid_indian ? "valid" : "invalid");
    badge.textContent = t.valid_indian ? "VALID" : "INVALID";
    head.appendChild(badge);

    card.appendChild(head);

    // ── Comparison row: source annotated frame + best crop side by side.
//    Clicking either image opens the lightbox. The plate text in the head
//    still opens the track-modal (per-frame reads). ─────────────
    if (t.best_crop_url || t.best_annotated_url) {
        const cmp = document.createElement("div");
        cmp.className = "track-comparison";

        // Source frame (full annotated video frame with bbox around plate)
        if (t.best_annotated_url) {
            const srcRow = document.createElement("div");
            srcRow.className = "track-source";

            const srcImg = document.createElement("img");
            srcImg.className = "track-source-img";
            srcImg.src = t.best_annotated_url + "?t=" + Date.now();
            srcImg.alt = `Source frame #${t.best_frame} for track #${t.track_id}`;
            srcImg.title = "Click to zoom — full video frame with bbox around the plate (compare OCR with real photo)";
            srcImg.setAttribute("data-zoom", t.best_annotated_url);
            srcRow.appendChild(srcImg);

            const srcLabel = document.createElement("div");
            srcLabel.className = "track-source-label";
            srcLabel.innerHTML =
                `<b>📷 Source frame #${t.best_frame ?? "?"}</b> — annotated with bbox around plate (click image to zoom · compare OCR text with real photo)`;
            srcRow.appendChild(srcLabel);

            cmp.appendChild(srcRow);
        }

        // Best crop (close-up of the plate that OCR read)
        if (t.best_crop_url) {
            const cropRow = document.createElement("div");
            cropRow.className = "track-crop";

            const cropImg = document.createElement("img");
            cropImg.className = "track-crop-img";
            cropImg.src = t.best_crop_url + "?t=" + Date.now();
            cropImg.alt = `Best crop for track #${t.track_id}`;
            cropImg.title = "Click to zoom — close-up crop that OCR read";
            cropImg.setAttribute("data-zoom", t.best_crop_url);
            cropRow.appendChild(cropImg);

            const cropLabel = document.createElement("div");
            cropLabel.className = "track-crop-label";
            cropLabel.innerHTML =
                `<b>🔍 OCR crop</b> — ${Math.round((t.best_conf || 0) * 100)}% per-frame · text: <code>${escapeHtml(t.best_text || "?")}</code>`;
            cropRow.appendChild(cropLabel);

            cmp.appendChild(cropRow);
        }

        card.appendChild(cmp);
    }

    // Meta line — uses correct field names from backend (first_seen, last_seen, etc.)
    const meta = document.createElement("div");
    meta.className = "track-meta";

    const span = (html) => {
        const s = document.createElement("span");
        s.innerHTML = html;
        return s;
    };

    meta.appendChild(span(`<b>${t.n_frames ?? 0}</b>&nbsp;frames`));
    meta.appendChild(span(`seen&nbsp;<b>${t.first_seen ?? "?"} → ${t.last_seen ?? "?"}</b>`));
    meta.appendChild(span(`OCR&nbsp;<b>${Math.round((t.final_conf || 0) * 100)}%</b>`));
    meta.appendChild(span(`YOLO&nbsp;<b>${Math.round((t.avg_yolo_conf || 0) * 100)}%</b>`));
    meta.appendChild(span(`unique reads&nbsp;<b>${t.n_unique_reads ?? 0}</b>`));
    card.appendChild(meta);

    // Per-character voting strip (the magic of video tracking)
    if (t.votes_per_pos && Object.keys(t.votes_per_pos).length) {
        const strip = document.createElement("div");
        strip.className = "votes-strip";
        const label = document.createElement("span");
        label.className = "votes-label";
        label.textContent = "Per-position votes:";
        strip.appendChild(label);

        const positions = Object.keys(t.votes_per_pos)
            .map((k) => parseInt(k, 10))
            .filter((n) => !isNaN(n))
            .sort((a, b) => a - b);

        positions.forEach((pos) => {
            const cell = document.createElement("span");
            cell.className = "vote-cell";
            const v = t.votes_per_pos[pos];
            // v is {char: count, ...}; pick the winning char
            let bestChar = "?";
            let bestCount = 0;
            Object.entries(v).forEach(([ch, n]) => {
                if (n > bestCount) { bestChar = ch; bestCount = n; }
            });
            cell.textContent = `[${pos}]${bestChar}`;
            cell.title = `Position ${pos}: ${JSON.stringify(v)}`;
            strip.appendChild(cell);
        });
        card.appendChild(strip);
    }

    return card;
}

// ============ Track-details modal (NEW) ============
//
// Click on a track's best-frame thumbnail → opens this modal listing every
// frame where ByteTrack fetched this plate. Each row shows the frame #, the
// crop thumbnail (click to zoom), the OCR text, and the confidence.
function openTrackDetails(t) {
    const modal = $("track-modal");
    const title = $("track-modal-title");
    const sub   = $("track-modal-sub");
    const grid  = $("track-modal-grid");
    const summary = $("track-modal-summary");

    title.textContent = `Track #${t.track_id} — all ${t.n_frames} frames fetched by ByteTrack`;
    sub.innerHTML = `Final voted text: <code>${escapeHtml(t.final_text || "(unreadable)")}</code>` +
        ` &nbsp;·&nbsp; <b>${Math.round((t.final_conf || 0) * 100)}%</b> confidence` +
        ` &nbsp;·&nbsp; frames <b>${t.first_seen} → ${t.last_seen}</b>`;

    // Summary chips
    const reads = t.per_frame_reads || [];
    const reads_with_text = reads.filter((r) => r.text && r.text.trim()).length;
    const avg_yolo = reads.length ? (reads.reduce((s, r) => s + (r.yolo_conf || 0), 0) / reads.length) : 0;
    const avg_ocr  = reads.length ? (reads.reduce((s, r) => s + (r.ocr_conf || 0), 0) / reads.length) : 0;
    summary.innerHTML = "";
    [
        ["Frames fetched", t.n_frames],
        ["Read succeeded", reads_with_text],
        ["Avg YOLO conf",  Math.round(avg_yolo * 100) + "%"],
        ["Avg OCR conf",   reads_with_text ? Math.round(avg_ocr * 100) + "%" : "—"],
        ["Unique OCR",    t.n_unique_reads ?? 0],
    ].forEach(([k, v]) => {
        const chip = document.createElement("span");
        chip.className = "vmeta-chip";
        chip.innerHTML = `${k}: <b>${escapeHtml(String(v))}</b>`;
        summary.appendChild(chip);
    });

    // Grid: one card per frame (sorted by frame number asc)
    grid.innerHTML = "";
    const sorted = [...reads].sort((a, b) => a.frame - b.frame);
    sorted.forEach((r) => {
        const row = document.createElement("div");
        row.className = "track-modal-row";
        if (r.frame === t.best_frame) row.classList.add("is-best");

        const head = document.createElement("div");
        head.className = "track-modal-row-head";
        head.innerHTML = `Frame <b>#${r.frame}</b>` +
            (r.frame === t.best_frame ? ` <span class="best-tag">BEST</span>` : "") +
            ` <span class="conf-tag yolo">YOLO ${Math.round((r.yolo_conf || 0) * 100)}%</span>` +
            ` <span class="conf-tag ocr">OCR ${Math.round((r.ocr_conf || 0) * 100)}%</span>`;
        row.appendChild(head);

        if (r.crop_url) {
            const img = document.createElement("img");
            img.className = "track-modal-thumb";
            img.src = r.crop_url + "?t=" + Date.now();
            img.alt = `Frame ${r.frame}`;
            img.loading = "lazy";
            img.addEventListener("click", () => {
                const lb = $("lightbox");
                $("lightbox-img").src = img.src;
                lb.classList.add("open");
                lb.setAttribute("aria-hidden", "false");
            });
            row.appendChild(img);
        }

        const textRow = document.createElement("div");
        textRow.className = "track-modal-text";
        if (r.text) {
            textRow.innerHTML = `<code>${escapeHtml(r.text)}</code>`;
        } else {
            textRow.innerHTML = `<span class="hint">(no readable text)</span>`;
        }
        row.appendChild(textRow);

        grid.appendChild(row);
    });

    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
    // Scroll to top of modal
    modal.scrollTop = 0;
}

function closeTrackDetails() {
    const modal = $("track-modal");
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
}

function setupTrackModal() {
    const modal = $("track-modal");
    // Close on backdrop click
    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeTrackDetails();
    });
    // Close on ESC
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !modal.classList.contains("hidden")) closeTrackDetails();
    });
    // Close button
    $("track-modal-close").addEventListener("click", closeTrackDetails);
}

function renderVehicleCard(v, idx) {
    const card = document.createElement("div");
    card.className = `vehicle-card ${v.class_name || ""}`;
    if (v.color && v.color.length === 3) {
        card.style.borderLeftColor = `rgb(${v.color[2]},${v.color[1]},${v.color[0]})`;
    }

    const emoji = v.emoji || VEHICLE_EMOJI[v.class_name] || "🚙";
    const name = document.createElement("div");
    name.className = "vehicle-name";
    name.textContent = `${emoji} ${v.class_name || "vehicle"}`;
    card.appendChild(name);

    const meta = document.createElement("div");
    meta.className = "vehicle-meta";
    const conf = document.createElement("span");
    conf.className = "vehicle-conf";
    conf.textContent = `${Math.round((v.confidence || 0) * 100)}%`;
    meta.appendChild(conf);
    const id_ = document.createElement("span");
    id_.className = "vehicle-idx";
    id_.textContent = `#${idx + 1}`;
    meta.appendChild(id_);
    card.appendChild(meta);

    if (v.bbox && v.bbox.length === 4) {
        const bbox = document.createElement("div");
        bbox.className = "vehicle-bbox";
        const [x1, y1, x2, y2] = v.bbox;
        bbox.textContent = `${x2 - x1}×${y2 - y1}px @ (${x1},${y1})`;
        card.appendChild(bbox);
    }

    return card;
}

function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

function showError(msg) {
    const e = $("error");
    e.textContent = "⚠ " + msg;
    e.classList.remove("hidden");
}
function hideError() { $("error").classList.add("hidden"); }

function hideResults() {
    $("results").classList.add("hidden");
    $("video-results").classList.add("hidden");
}

// --- Init ---
document.addEventListener("DOMContentLoaded", () => {
    setupLightbox();
    setupTrackModal();
    setupVideoResultsTabs();
    checkHealth();
    setupModeToggle();
    setupImageDropzone();
    setupVideoDropzone();
});