import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { getProbableContributingFactors, renderContributingFactorsHtml } from "./clinical_patterns.js";

/* ============================================================
   DOM REFERENCES
   ============================================================ */
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const statusUI = document.getElementById("status");
const switchCameraBtn = document.getElementById("switch-camera");

const startScreen = document.getElementById("start-screen");
const beginBtn = document.getElementById("begin-btn");

const phaseBanner = document.getElementById("phase-banner");
const phaseInstructionUI = document.getElementById("phase-instruction");
const phaseCountdownUI = document.getElementById("phase-countdown");
const stopBtn = document.getElementById("stop-btn");

const reportScreen = document.getElementById("report-screen");
const reportListUI = document.getElementById("report-list");
const restartBtn = document.getElementById("restart-btn");
const modeInstructionsUI = document.getElementById("mode-instructions");
const viewModeBtns = document.querySelectorAll(".view-mode-btn");

const MODE_INSTRUCTIONS = {
    side: "Position the camera to the side, perpendicular to a walking path with at least 4-5 steps of space. Stand back so your full body stays in frame as you walk across it — one or two passes is plenty.",
    front: "Position the camera facing you, at chest-to-hip height. Walk toward the camera for several steps, then turn around and walk away — one pass in each direction is plenty.",
    rear: "Position the camera behind you, framing your legs and heels. Walk away from the camera for several steps, keeping your lower legs clearly visible.",
};

let selectedMode = "side";
viewModeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
        selectedMode = btn.dataset.mode;
        viewModeBtns.forEach((b) => b.classList.toggle("active", b === btn));
        modeInstructionsUI.innerText = MODE_INSTRUCTIONS[selectedMode];
    });
});

/* ============================================================
   MEDIAPIPE / CAMERA SETUP (same pattern as index.html's app.js)
   ============================================================ */
let poseLandmarker = undefined;
let lastVideoTime = -1;
let currentFacingMode = "environment";
let currentStream = null;
let latestLandmarks = null;

async function initializeTracker() {
    try {
        statusUI.innerText = "Loading WebAssembly Components...";
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });
        statusUI.innerText = "Ready. Requesting Camera Access...";
        startCamera();
    } catch (error) {
        statusUI.innerText = "Initialization Failed.";
        console.error("MediaPipe Initialization Error:", error);
    }
}

function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusUI.innerText = "Camera API not supported in this browser.";
        return;
    }
    if (currentStream) {
        currentStream.getTracks().forEach((track) => track.stop());
    }
    const constraints = {
        video: { facingMode: currentFacingMode, width: 640, height: 480, frameRate: { ideal: 30 } }
    };
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            currentStream = stream;
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam, { once: true });
            statusUI.innerText = "Scanning Active";
        })
        .catch((err) => {
            statusUI.innerText = "Camera Access Denied by User or System.";
            console.error("Camera Error:", err);
        });
}

switchCameraBtn.addEventListener("click", () => {
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    lastVideoTime = -1;
    startCamera();
});

async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime && poseLandmarker) {
        lastVideoTime = video.currentTime;
        const results = poseLandmarker.detectForVideo(video, startTimeMs);
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            latestLandmarks = landmarks;
            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#00ffcc", lineWidth: 2 });
            drawingUtils.drawLandmarks(landmarks, { color: "#ff3333", radius: 3 });

            if (gait.state === "capturing") {
                gait.buffer.push({ t: performance.now(), lm: snapshotLandmarks(landmarks) });
            }
        } else {
            latestLandmarks = null;
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

/* ============================================================
   GAIT CAPTURE STATE MACHINE
   ============================================================ */
const NEEDED_INDICES = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

function snapshotLandmarks(landmarks) {
    const snap = {};
    for (const i of NEEDED_INDICES) {
        const lm = landmarks[i];
        snap[i] = lm ? { x: lm.x, y: lm.y, v: typeof lm.visibility === "number" ? lm.visibility : null } : null;
    }
    return snap;
}

const COUNTDOWN_SEC = 5;
const MAX_CAPTURE_MS = 25000; // hard cap; "Stop & Analyze" ends it earlier

const gait = {
    state: "idle", // idle | countdown | capturing | done
    buffer: [],
    countdownTimer: null,
    captureTimer: null,
};

beginBtn.addEventListener("click", startCapture);
stopBtn.addEventListener("click", () => { if (gait.state === "capturing") endCapture(); });
restartBtn.addEventListener("click", () => {
    reportScreen.classList.add("hidden");
    startCapture();
});

function startCapture() {
    gait.buffer = [];
    gait.mode = selectedMode;
    startScreen.classList.add("hidden");
    phaseBanner.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    phaseInstructionUI.innerText = "Get ready — " + MODE_INSTRUCTIONS[selectedMode];

    let remaining = COUNTDOWN_SEC;
    gait.state = "countdown";
    phaseCountdownUI.innerText = remaining;
    clearInterval(gait.countdownTimer);
    gait.countdownTimer = setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
            phaseCountdownUI.innerText = remaining;
        } else {
            clearInterval(gait.countdownTimer);
            beginRecording();
        }
    }, 1000);
}

function beginRecording() {
    gait.state = "capturing";
    phaseInstructionUI.innerText = "Walking — capture in progress";
    phaseCountdownUI.innerText = "Walk now";
    stopBtn.classList.remove("hidden");
    clearTimeout(gait.captureTimer);
    gait.captureTimer = setTimeout(endCapture, MAX_CAPTURE_MS);
}

function endCapture() {
    clearTimeout(gait.captureTimer);
    gait.state = "done";
    phaseBanner.classList.add("hidden");
    const analysis = gait.mode === "front" ? analyzeFrontGait(gait.buffer)
        : gait.mode === "rear" ? analyzeRearGait(gait.buffer)
        : analyzeGait(gait.buffer);
    showReport(analysis, gait.mode);
}

/* ============================================================
   ANALYSIS
   ============================================================ */
function avg(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null; }

function calculateAngle(p1, p2, p3) {
    let radians = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) angle = 360.0 - angle;
    return angle;
}

function smoothSeries(series, windowSize) {
    const out = new Array(series.length).fill(null);
    const half = Math.floor(windowSize / 2);
    for (let i = 0; i < series.length; i++) {
        let sum = 0, count = 0;
        for (let j = Math.max(0, i - half); j <= Math.min(series.length - 1, i + half); j++) {
            if (series[j] !== null) { sum += series[j]; count++; }
        }
        out[i] = count > 0 ? sum / count : null;
    }
    return out;
}

// Simple local-maxima peak finder with a minimum-spacing constraint, so
// per-frame noise doesn't get counted as extra strides.
function findPeaks(series, minSpacingFrames, minProminence) {
    const peaks = [];
    for (let i = 1; i < series.length - 1; i++) {
        if (series[i] === null) continue;
        if (series[i] > series[i - 1] && series[i] >= series[i + 1]) {
            if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minSpacingFrames) {
                peaks.push(i);
            } else if (series[i] > series[peaks[peaks.length - 1]]) {
                peaks[peaks.length - 1] = i; // keep the taller nearby peak
            }
        }
    }
    // Prominence filter: drop peaks that don't stand out from the local baseline
    const validVals = series.filter((v) => v !== null);
    const range = validVals.length ? Math.max(...validVals) - Math.min(...validVals) : 0;
    const minProm = range * minProminence;
    return peaks.filter((i) => {
        const lo = Math.max(0, i - minSpacingFrames);
        const hi = Math.min(series.length - 1, i + minSpacingFrames);
        const localMin = Math.min(...series.slice(lo, hi + 1).filter((v) => v !== null));
        return series[i] - localMin >= minProm;
    });
}

function avgVisibility(buffer, indices) {
    const scores = [];
    for (const frame of buffer) {
        for (const i of indices) {
            const lm = frame.lm[i];
            if (lm && lm.v !== null) scores.push(lm.v);
        }
    }
    return scores.length ? avg(scores) : null;
}

function analyzeGait(buffer) {
    if (buffer.length < 30) {
        return { valid: false, note: "Not enough tracking data captured. Try a longer walk or better lighting." };
    }

    // Pick the higher-confidence side as "near leg" (the side facing the camera).
    const leftVis = avgVisibility(buffer, [23, 25, 27]);
    const rightVis = avgVisibility(buffer, [24, 26, 28]);
    const useLeft = (leftVis || 0) >= (rightVis || 0);
    const side = useLeft ? "Left" : "Right";
    const HIP = useLeft ? 23 : 24, KNEE = useLeft ? 25 : 26, ANKLE = useLeft ? 27 : 28;
    const SHOULDER_L = 11, SHOULDER_R = 12;

    const fps = buffer.length > 1 ? 1000 / ((buffer[buffer.length - 1].t - buffer[0].t) / (buffer.length - 1)) : 30;

    // Ankle position relative to hip (horizontal) — oscillates as the leg
    // swings forward/back through the gait cycle. Peaks approximate the
    // near-leg's initial-contact ("heel strike") events.
    const ankleRelX = buffer.map((f) => {
        const hip = f.lm[HIP], ankle = f.lm[ANKLE];
        return (hip && ankle) ? (ankle.x - hip.x) : null;
    });
    const smoothed = smoothSeries(ankleRelX, 5);
    const minSpacingFrames = Math.round(fps * 0.35); // strides shouldn't be faster than ~0.35s apart
    const peaks = findPeaks(smoothed, minSpacingFrames, 0.15);

    if (peaks.length < 2) {
        return {
            valid: false,
            note: "Could not detect a clear repeating stride pattern. Make sure you walk fully across the frame, side-on to the camera, with your near leg clearly visible.",
        };
    }

    const strideTimesS = [];
    for (let i = 1; i < peaks.length; i++) {
        strideTimesS.push((buffer[peaks[i]].t - buffer[peaks[i - 1]].t) / 1000);
    }
    const avgStrideTimeS = avg(strideTimesS);
    const cadenceStepsPerMin = avgStrideTimeS ? Math.round(120 / avgStrideTimeS) : null; // 2 steps per stride

    // Stance/swing ratio (approximate): within each stride, treat frames
    // where the ankle is in the lowest 40% of its vertical range for that
    // stride as "stance" (foot near ground), rest as "swing". A rough,
    // clearly-approximate proxy — real stance/swing needs ground-contact
    // sensing, which a single camera doesn't have.
    let stanceRatios = [];
    for (let i = 0; i < peaks.length - 1; i++) {
        const seg = buffer.slice(peaks[i], peaks[i + 1]);
        const ys = seg.map((f) => f.lm[ANKLE] ? f.lm[ANKLE].y : null).filter((v) => v !== null);
        if (ys.length < 3) continue;
        const maxY = Math.max(...ys); // largest y = lowest in image = closest to ground
        const minY = Math.min(...ys);
        const threshold = maxY - 0.4 * (maxY - minY);
        const stanceFrames = ys.filter((y) => y >= threshold).length;
        stanceRatios.push(stanceFrames / ys.length);
    }
    const avgStancePct = stanceRatios.length ? Math.round(avg(stanceRatios) * 100) : null;

    // Joint ROM during gait (near leg), plus trunk vertical oscillation.
    const hipAngles = [], kneeAngles = [], ankleAngles = [], shoulderYs = [];
    for (const frame of buffer) {
        const s = frame.lm;
        const hip = s[HIP], knee = s[KNEE], ankle = s[ANKLE];
        const shoulder = useLeft ? s[SHOULDER_L] : s[SHOULDER_R];
        if (shoulder && hip && knee) {
            hipAngles.push(calculateAngle(shoulder, hip, knee));
        }
        if (hip && knee && ankle) {
            kneeAngles.push(calculateAngle(hip, knee, ankle));
        }
        if (knee && ankle) {
            const virtualToe = { x: ankle.x + 0.05, y: ankle.y };
            ankleAngles.push(calculateAngle(knee, ankle, virtualToe));
        }
        if (s[SHOULDER_L] && s[SHOULDER_R]) {
            shoulderYs.push((s[SHOULDER_L].y + s[SHOULDER_R].y) / 2);
        }
    }
    const rom = (arr) => (arr.length ? Math.round(Math.max(...arr) - Math.min(...arr)) : null);
    const trunkOscillationIdx = shoulderYs.length ? Math.round((Math.max(...shoulderYs) - Math.min(...shoulderYs)) * 100) : null;

    return {
        valid: true,
        side,
        trackingConfidence: Math.round((useLeft ? leftVis : rightVis) * 100),
        stridesDetected: peaks.length - 1,
        avgStrideTimeS: avgStrideTimeS ? Math.round(avgStrideTimeS * 100) / 100 : null,
        cadenceStepsPerMin,
        avgStancePct,
        hipRomDeg: rom(hipAngles),
        kneeRomDeg: rom(kneeAngles),
        ankleRomDeg: rom(ankleAngles),
        trunkOscillationIdx,
        captureDurationS: Math.round((buffer[buffer.length - 1].t - buffer[0].t) / 100) / 10,
    };
}

/* ============================================================
   STANCE-PHASE DETECTION (shared by front & rear view analysis)
   ============================================================
   Comparative heuristic: whichever ankle sits lower in the image frame
   (larger normalized y) at a given moment is treated as the stance
   (weight-bearing) leg, the other as mid-swing (lifted, smaller y).
   This is intentionally comparative rather than an absolute threshold,
   so it stays robust even as the subject changes apparent size while
   walking toward/away from the camera. */
function detectStanceWindows(buffer, minDurationMs = 150) {
    const leftY = smoothSeries(buffer.map((f) => f.lm[27] ? f.lm[27].y : null), 5);
    const rightY = smoothSeries(buffer.map((f) => f.lm[28] ? f.lm[28].y : null), 5);

    const windows = [];
    let currentSide = null, startIdx = null;

    for (let i = 0; i < buffer.length; i++) {
        if (leftY[i] === null || rightY[i] === null) continue;
        const side = leftY[i] > rightY[i] ? "left" : "right";
        if (side !== currentSide) {
            if (currentSide !== null) {
                const durationMs = buffer[i - 1].t - buffer[startIdx].t;
                if (durationMs >= minDurationMs) windows.push({ side: currentSide, startIdx, endIdx: i - 1 });
            }
            currentSide = side;
            startIdx = i;
        }
    }
    if (currentSide !== null) {
        const durationMs = buffer[buffer.length - 1].t - buffer[startIdx].t;
        if (durationMs >= minDurationMs) windows.push({ side: currentSide, startIdx, endIdx: buffer.length - 1 });
    }
    return windows;
}

// Signed deviation of a joint from the straight line between two other
// joints, positive = toward the body midline. Same technique already used
// in app.js for frontal-plane knee valgus and sagittal knee hyperextension.
function signedDeviationFromLine(proximal, joint, distal, midlineX) {
    if (proximal.y === distal.y) return null;
    const t = (joint.y - proximal.y) / (distal.y - proximal.y);
    const expectedX = proximal.x + (distal.x - proximal.x) * t;
    const inwardSign = Math.sign(midlineX - proximal.x) || 1;
    return (joint.x - expectedX) * inwardSign;
}

/* ============================================================
   FRONT-VIEW ANALYSIS: dynamic knee valgus + Trendelenburg (hip-drop) pattern
   ============================================================ */
function analyzeFrontGait(buffer) {
    if (buffer.length < 30) {
        return { valid: false, mode: "front", note: "Not enough tracking data captured. Try a longer walk toward/away from the camera." };
    }
    const windows = detectStanceWindows(buffer);
    if (windows.length < 2) {
        return { valid: false, mode: "front", note: "Could not detect clear alternating stance phases. Make sure both legs are visible as you walk toward/away from the camera." };
    }

    const HIP = { left: 23, right: 24 }, KNEE = { left: 25, right: 26 }, ANKLE = { left: 27, right: 28 };
    const trackingConfidence = Math.round((avgVisibility(buffer, [23, 24, 25, 26, 27, 28]) || 0) * 100);

    const results = { left: { valgusDevs: [], hipDropDevs: [] }, right: { valgusDevs: [], hipDropDevs: [] } };

    for (const w of windows) {
        const other = w.side === "left" ? "right" : "left";
        for (let i = w.startIdx; i <= w.endIdx; i++) {
            const s = buffer[i].lm;
            const hip = s[HIP[w.side]], knee = s[KNEE[w.side]], ankle = s[ANKLE[w.side]];
            const hipOther = s[HIP[other]];
            const hipL = s[23], hipR = s[24];
            if (hip && knee && ankle && hipL && hipR) {
                const midlineX = (hipL.x + hipR.x) / 2;
                const dev = signedDeviationFromLine(hip, knee, ankle, midlineX);
                if (dev !== null) results[w.side].valgusDevs.push(dev);
            }
            if (hip && hipOther) {
                results[w.side].hipDropDevs.push(hipOther.y - hip.y); // positive = contralateral hip lower (dropped)
            }
        }
    }

    const summarize = (side) => ({
        avgValgusDevIdx: results[side].valgusDevs.length ? Math.round(avg(results[side].valgusDevs) * 100) : null,
        avgHipDropIdx: results[side].hipDropDevs.length ? Math.round(avg(results[side].hipDropDevs) * 100) : null,
    });

    return {
        valid: true, mode: "front",
        trackingConfidence,
        stancePhasesDetected: windows.length,
        left: summarize("left"),
        right: summarize("right"),
    };
}

/* ============================================================
   REAR-VIEW ANALYSIS: shank/heel frontal-plane alignment during stance
   ============================================================ */
function analyzeRearGait(buffer) {
    if (buffer.length < 30) {
        return { valid: false, mode: "rear", note: "Not enough tracking data captured. Try a longer walk away from the camera." };
    }
    const windows = detectStanceWindows(buffer);
    if (windows.length < 2) {
        return { valid: false, mode: "rear", note: "Could not detect clear alternating stance phases. Make sure both legs and heels are visible as you walk away from the camera." };
    }

    const HIP = { left: 23, right: 24 }, KNEE = { left: 25, right: 26 }, ANKLE = { left: 27, right: 28 };
    const trackingConfidence = Math.round((avgVisibility(buffer, [25, 26, 27, 28, 29, 30]) || 0) * 100);

    const shankDevs = { left: [], right: [] };
    for (const w of windows) {
        for (let i = w.startIdx; i <= w.endIdx; i++) {
            const s = buffer[i].lm;
            const knee = s[KNEE[w.side]], ankle = s[ANKLE[w.side]];
            const hipL = s[23], hipR = s[24];
            if (knee && ankle && hipL && hipR) {
                const midlineX = (hipL.x + hipR.x) / 2;
                // Shank vector (knee->ankle) horizontal offset, signed toward
                // the midline (medial/pronation-associated) or away from it
                // (lateral/supination-associated) — a simple frontal-plane
                // lean proxy since we don't have a true vertical reference
                // in-frame to compare the shank against.
                const verticalDev = (ankle.x - knee.x);
                const inwardSign = Math.sign(midlineX - knee.x) || 1;
                shankDevs[w.side].push(verticalDev * inwardSign);
            }
        }
    }

    const summarize = (side) => {
        const devs = shankDevs[side];
        if (!devs.length) return null;
        const meanDev = avg(devs);
        return {
            devIdx: Math.round(meanDev * 100),
            direction: meanDev > 0 ? "medial" : "lateral",
        };
    };

    return {
        valid: true, mode: "rear",
        trackingConfidence,
        stancePhasesDetected: windows.length,
        left: summarize("left"),
        right: summarize("right"),
    };
}

/* ============================================================
   REPORT RENDERING
   ============================================================ */
function flagClass(level) {
    return { good: "flag-good", moderate: "flag-moderate", flag: "flag-alert" }[level] || "flag-good";
}

function showReport(result, mode) {
    if (mode === "front") return showFrontReport(result);
    if (mode === "rear") return showRearReport(result);
    return showSideReport(result);
}

function showSideReport(result) {
    const rows = [];
    const push = (section, row) => rows.push({ section, ...row });

    if (!result.valid) {
        push("Gait Assessment", { label: "Analysis", value: "Incomplete", level: "moderate", note: result.note });
    } else {
        push("Capture Summary", {
            label: "Near-Side Leg Tracked", value: result.side, level: "good",
            note: "Automatically selected as the leg with higher average landmark visibility — the side facing the camera."
        });
        push("Capture Summary", {
            label: "Tracking Confidence", value: `${result.trackingConfidence}%`,
            level: result.trackingConfidence < 60 ? "flag" : result.trackingConfidence < 80 ? "moderate" : "good",
            note: "Average landmark visibility for the tracked leg across the whole capture."
        });
        push("Capture Summary", {
            label: "Strides Detected", value: `${result.stridesDetected}`, level: "good",
            note: `Over a ${result.captureDurationS}s capture. More strides (multiple walking passes) improve reliability.`
        });

        push("Temporal Parameters", {
            label: "Cadence (est.)", value: `${result.cadenceStepsPerMin} steps/min`, level: "good",
            note: "Estimated assuming a roughly symmetric gait (2 steps per detected stride of the tracked leg). Typical adult walking cadence is often cited around 100-120 steps/min, but varies widely by age, height, and pace."
        });
        push("Temporal Parameters", {
            label: "Stride Time", value: `${result.avgStrideTimeS} s`, level: "good",
            note: "Average time for the tracked leg to complete one full gait cycle."
        });
        if (result.avgStancePct != null) {
            push("Temporal Parameters", {
                label: "Stance Phase (approx.)", value: `${result.avgStancePct}%`,
                level: (result.avgStancePct < 50 || result.avgStancePct > 70) ? "moderate" : "good",
                note: "Rough proxy from ankle height within each stride, not true ground-contact sensing. Typical walking stance phase is roughly 60% of the gait cycle — treat deviations as a prompt to look more closely, not a diagnosis."
            });
        }

        if (result.hipRomDeg != null) {
            push("Joint Range of Motion (Near Leg)", {
                label: "Hip ROM During Gait", value: `${result.hipRomDeg}°`, level: "good",
                note: "Total hip flexion/extension range observed across the capture."
            });
        }
        if (result.kneeRomDeg != null) {
            push("Joint Range of Motion (Near Leg)", {
                label: "Knee ROM During Gait", value: `${result.kneeRomDeg}°`, level: "good",
                note: "Total knee flexion/extension range observed across the capture."
            });
        }
        if (result.ankleRomDeg != null) {
            push("Joint Range of Motion (Near Leg)", {
                label: "Ankle ROM During Gait", value: `${result.ankleRomDeg}°`, level: "good",
                note: "Total ankle range observed across the capture. Ankle/foot landmarks are the least reliable in pose estimation — treat this value with more caution."
            });
        }
        if (result.trunkOscillationIdx != null) {
            push("Trunk", {
                label: "Vertical Trunk Oscillation", value: `${result.trunkOscillationIdx} idx`,
                level: result.trunkOscillationIdx > 8 ? "moderate" : "good",
                note: "Vertical bob of the shoulder midpoint during walking — normalized index, not a physical distance (no calibration reference in frame)."
            });
        }
    }

    let html = "";
    let lastSection = null;
    for (const r of rows) {
        if (r.section !== lastSection) {
            html += `<div class="report-section-header">${r.section}</div>`;
            lastSection = r.section;
        }
        html += `
        <div class="report-row ${flagClass(r.level)}">
            <div class="report-row-top">
                <span class="report-label">${r.label}</span>
                <span class="report-value">${r.value}</span>
            </div>
            <div class="report-note">${r.note}</div>
        </div>`;
    }
    reportListUI.innerHTML = html;
    reportScreen.classList.remove("hidden");
}

function renderRows(rows) {
    let html = "";
    let lastSection = null;
    for (const r of rows) {
        if (r.section !== lastSection) {
            html += `<div class="report-section-header">${r.section}</div>`;
            lastSection = r.section;
        }
        html += `
        <div class="report-row ${flagClass(r.level)}">
            <div class="report-row-top">
                <span class="report-label">${r.label}</span>
                <span class="report-value">${r.value}</span>
            </div>
            <div class="report-note">${r.note}</div>
        </div>`;
    }
    return html;
}

function showFrontReport(result) {
    const rows = [];
    const push = (section, row) => rows.push({ section, ...row });

    if (!result.valid) {
        push("Front-View Gait Assessment", { label: "Analysis", value: "Incomplete", level: "moderate", note: result.note });
    } else {
        push("Capture Summary", {
            label: "Tracking Confidence", value: `${result.trackingConfidence}%`,
            level: result.trackingConfidence < 60 ? "flag" : result.trackingConfidence < 80 ? "moderate" : "good",
            note: "Average visibility across both hips, knees, and ankles during the capture."
        });
        push("Capture Summary", {
            label: "Stance Phases Detected", value: `${result.stancePhasesDetected}`, level: "good",
            note: "Alternating left/right weight-bearing phases identified from comparative ankle height."
        });

        for (const side of ["left", "right"]) {
            const r = result[side];
            if (!r) continue;
            const label = side === "left" ? "Left" : "Right";
            if (r.avgValgusDevIdx != null) {
                const interp = r.avgValgusDevIdx > 3 ? "Valgus tendency" : r.avgValgusDevIdx < -3 ? "Varus tendency" : "Within normal range";
                push(`${label} Leg — Stance Phase`, {
                    label: "Dynamic Knee Alignment", value: `${r.avgValgusDevIdx} idx — ${interp}`,
                    level: Math.abs(r.avgValgusDevIdx) > 6 ? "flag" : Math.abs(r.avgValgusDevIdx) > 3 ? "moderate" : "good",
                    note: `Knee position relative to the hip-ankle line during ${label.toLowerCase()}-leg stance phases of walking. Positive = drifted toward the midline (valgus).`
                });
            }
            if (r.avgHipDropIdx != null) {
                const dropping = r.avgHipDropIdx > 2;
                push(`${label} Leg — Stance Phase`, {
                    label: "Contralateral Hip Drop", value: `${r.avgHipDropIdx} idx${dropping ? " — drop observed" : ""}`,
                    level: r.avgHipDropIdx > 5 ? "flag" : r.avgHipDropIdx > 2 ? "moderate" : "good",
                    note: `How much the opposite hip drops during ${label.toLowerCase()}-leg single-limb stance — the classic Trendelenburg-test signal.`
                });
            }
        }
    }

    let html = renderRows(rows);

    if (result.valid) {
        let allPatterns = [];
        for (const side of ["left", "right"]) {
            const r = result[side];
            const findings = {};
            if (r && r.avgValgusDevIdx != null && r.avgValgusDevIdx > 3) {
                findings.frontGaitKneeValgus = { side: side === "left" ? "Left" : "Right" };
            }
            if (r && r.avgHipDropIdx != null && r.avgHipDropIdx > 5) {
                findings.trendelenburgPattern = { side: side === "left" ? "Left" : "Right" };
            }
            allPatterns = allPatterns.concat(getProbableContributingFactors(findings));
        }
        html += renderContributingFactorsHtml(allPatterns);
    }

    reportListUI.innerHTML = html;
    reportScreen.classList.remove("hidden");
}

function showRearReport(result) {
    const rows = [];
    const push = (section, row) => rows.push({ section, ...row });

    if (!result.valid) {
        push("Rear-View Gait Assessment", { label: "Analysis", value: "Incomplete", level: "moderate", note: result.note });
    } else {
        push("Capture Summary", {
            label: "Tracking Confidence", value: `${result.trackingConfidence}%`,
            level: result.trackingConfidence < 60 ? "flag" : result.trackingConfidence < 80 ? "moderate" : "good",
            note: "Average visibility across knees, ankles, and heels during the capture — the lowest-confidence landmarks in pose estimation, so weight this report accordingly."
        });
        push("Capture Summary", {
            label: "Stance Phases Detected", value: `${result.stancePhasesDetected}`, level: "good",
            note: "Alternating left/right weight-bearing phases identified from comparative ankle height."
        });

        for (const side of ["left", "right"]) {
            const r = result[side];
            if (!r) continue;
            const label = side === "left" ? "Left" : "Right";
            if (r) {
                push(`${label} Leg — Stance Phase`, {
                    label: "Shank Frontal-Plane Lean", value: `${Math.abs(r.devIdx)} idx (${r.direction})`,
                    level: Math.abs(r.devIdx) > 6 ? "flag" : Math.abs(r.devIdx) > 3 ? "moderate" : "good",
                    note: `Lower-leg lean during ${label.toLowerCase()}-leg stance, viewed from behind. A medial lean is the classic visual proxy for a dynamic pronation-associated pattern; lateral lean is supination-associated. This is the roughest measurement in this whole tool — heel/ankle landmarks are the least reliable from any single camera angle.`
                });
            }
        }
    }

    let html = renderRows(rows);

    if (result.valid) {
        let allPatterns = [];
        for (const side of ["left", "right"]) {
            const r = result[side];
            const findings = {};
            if (r && Math.abs(r.devIdx) > 3) {
                findings.rearShankAlignment = { side: side === "left" ? "Left" : "Right", direction: r.direction };
            }
            allPatterns = allPatterns.concat(getProbableContributingFactors(findings));
        }
        html += renderContributingFactorsHtml(allPatterns);
    }

    reportListUI.innerHTML = html;
    reportScreen.classList.remove("hidden");
}

/* ============================================================
   BOOT
   ============================================================ */
initializeTracker();
