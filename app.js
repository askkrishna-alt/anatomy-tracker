import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

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
const phaseProgressUI = document.getElementById("phase-progress");
const phaseInstructionUI = document.getElementById("phase-instruction");
const phaseCountdownUI = document.getElementById("phase-countdown");
const nextPhaseBtn = document.getElementById("next-phase-btn");

const metricsPanel = document.getElementById("metrics-panel");
const headAngleUI = document.getElementById("head-angle");
const pelvicTiltUI = document.getElementById("pelvic-tilt");
const shoulderSymUI = document.getElementById("shoulder-sym");

const reportScreen = document.getElementById("report-screen");
const reportListUI = document.getElementById("report-list");
const restartBtn = document.getElementById("restart-btn");

/* ============================================================
   MEDIAPIPE / CAMERA SETUP
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

            drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
                color: "#00ffcc",
                lineWidth: 2
            });
            drawingUtils.drawLandmarks(landmarks, { color: "#ff3333", radius: 3 });

            updateLiveMetrics(landmarks);

            if (assessment.state === "capturing") {
                assessment.buffer.push({ t: performance.now(), lm: snapshotLandmarks(landmarks) });
            }
        } else {
            latestLandmarks = null;
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

/* ============================================================
   LIVE METRIC READOUT (shown continuously behind the flow)
   ============================================================ */
function calculateAngle(p1, p2, p3) {
    let radians = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) angle = 360.0 - angle;
    return Math.round(angle);
}

function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function updateLiveMetrics(landmarks) {
    const leftEar = landmarks[7];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    if (leftEar && leftShoulder) {
        const virtualVerticalPoint = { x: leftShoulder.x, y: leftShoulder.y - 0.2 };
        const forwardHeadAngle = calculateAngle(virtualVerticalPoint, leftShoulder, leftEar);
        headAngleUI.innerText = `${forwardHeadAngle}°`;
        headAngleUI.style.color = forwardHeadAngle > 25 ? "#ff3333" : "#00ffcc";
    }
    if (leftHip && rightHip) {
        const hipHeightDifference = Math.abs(leftHip.y - rightHip.y);
        pelvicTiltUI.innerText = `${Math.round(hipHeightDifference * 100)} idx`;
    }
    if (leftShoulder && rightShoulder) {
        const shoulderHeightDifference = Math.abs(leftShoulder.y - rightShoulder.y);
        const score = Math.round(shoulderHeightDifference * 100);
        shoulderSymUI.innerText = `${score} idx`;
        shoulderSymUI.style.color = score > 3 ? "#ff3333" : "#00ffcc";
    }
}

/* ============================================================
   GUIDED ASSESSMENT STATE MACHINE
   ============================================================ */
const NEEDED_INDICES = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

function snapshotLandmarks(landmarks) {
    const snap = {};
    for (const i of NEEDED_INDICES) {
        const lm = landmarks[i];
        snap[i] = lm ? { x: lm.x, y: lm.y } : null;
    }
    return snap;
}

const PHASES = [
    {
        key: "front",
        label: "Front Stance",
        instruction: "Stand facing the camera, feet hip-width apart, arms relaxed at your sides.",
        countdownSec: 4,
        captureMs: 2500
    },
    {
        key: "profile",
        label: "Side Stance",
        instruction: "Turn 90° so your side faces the camera. Stand naturally.",
        countdownSec: 4,
        captureMs: 2500
    },
    {
        key: "squat",
        label: "Squats",
        instruction: "Face the camera again. Perform 3 slow, controlled squats.",
        countdownSec: 3,
        captureMs: 18000
    }
];

const assessment = {
    state: "idle", // idle | countdown | capturing | done
    phaseIndex: 0,
    buffer: [],
    countdownTimer: null,
    captureTimer: null,
    results: {}
};

beginBtn.addEventListener("click", startAssessment);
restartBtn.addEventListener("click", () => {
    reportScreen.classList.add("hidden");
    metricsPanel.classList.remove("hidden");
    startAssessment();
});
nextPhaseBtn.addEventListener("click", () => {
    if (assessment.state === "capturing") endPhaseCapture();
});

function startAssessment() {
    assessment.phaseIndex = 0;
    assessment.results = {};
    startScreen.classList.add("hidden");
    beginPhase(0);
}

function beginPhase(index) {
    if (index >= PHASES.length) {
        showReport();
        return;
    }
    assessment.phaseIndex = index;
    assessment.buffer = [];
    const phase = PHASES[index];

    phaseBanner.classList.remove("hidden");
    nextPhaseBtn.classList.add("hidden");
    phaseProgressUI.innerText = PHASES.map((p, i) => (i === index ? "●" : "○")).join(" ");
    phaseInstructionUI.innerText = `${phase.label}: ${phase.instruction}`;

    let remaining = phase.countdownSec;
    assessment.state = "countdown";
    phaseCountdownUI.innerText = remaining;
    clearInterval(assessment.countdownTimer);
    assessment.countdownTimer = setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
            phaseCountdownUI.innerText = remaining;
        } else {
            clearInterval(assessment.countdownTimer);
            beginCapture(phase);
        }
    }, 1000);
}

function beginCapture(phase) {
    assessment.state = "capturing";
    phaseCountdownUI.innerText = phase.key === "squat" ? "Go — squat 3x" : "Hold still";
    nextPhaseBtn.classList.remove("hidden");
    clearTimeout(assessment.captureTimer);
    assessment.captureTimer = setTimeout(endPhaseCapture, phase.captureMs);
}

function endPhaseCapture() {
    clearTimeout(assessment.captureTimer);
    const phase = PHASES[assessment.phaseIndex];
    assessment.results[phase.key] = analyzePhase(phase.key, assessment.buffer);
    beginPhase(assessment.phaseIndex + 1);
}

/* ============================================================
   PER-PHASE ANALYSIS
   ============================================================ */
function analyzePhase(key, buffer) {
    if (key === "front") return analyzeFrontHold(buffer);
    if (key === "profile") return analyzeProfileHold(buffer);
    if (key === "squat") return analyzeSquat(buffer);
    return {};
}

function avg(arr) {
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
}

function analyzeFrontHold(buffer) {
    const shoulderDeltas = [], hipDeltas = [], headTilts = [];
    for (const frame of buffer) {
        const s = frame.lm;
        if (s[11] && s[12]) shoulderDeltas.push(Math.abs(s[11].y - s[12].y));
        if (s[23] && s[24]) hipDeltas.push(Math.abs(s[23].y - s[24].y));
        if (s[7] && s[8]) headTilts.push(Math.abs(s[7].y - s[8].y));
    }
    return {
        shoulderSymIdx: avg(shoulderDeltas) !== null ? Math.round(avg(shoulderDeltas) * 100) : null,
        hipSymIdx: avg(hipDeltas) !== null ? Math.round(avg(hipDeltas) * 100) : null,
        headTiltIdx: avg(headTilts) !== null ? Math.round(avg(headTilts) * 100) : null,
        frameCount: buffer.length
    };
}

function analyzeProfileHold(buffer) {
    const headAngles = [], trunkAngles = [];
    for (const frame of buffer) {
        const s = frame.lm;
        const ear = s[7] || s[8];
        const shoulder = s[11] || s[12];
        const hip = s[23] || s[24];
        if (ear && shoulder) {
            const virtualVerticalPoint = { x: shoulder.x, y: shoulder.y - 0.2 };
            headAngles.push(calculateAngle(virtualVerticalPoint, shoulder, ear));
        }
        if (shoulder && hip) {
            const dx = shoulder.x - hip.x;
            const dy = shoulder.y - hip.y;
            const angleFromVertical = Math.round(Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI));
            trunkAngles.push(angleFromVertical);
        }
    }
    return {
        forwardHeadAngle: avg(headAngles) !== null ? Math.round(avg(headAngles)) : null,
        trunkLeanAngle: avg(trunkAngles) !== null ? Math.round(avg(trunkAngles)) : null,
        frameCount: buffer.length
    };
}

function analyzeSquat(buffer) {
    const hipYSeries = buffer.map((f) => {
        const lh = f.lm[23], rh = f.lm[24];
        if (!lh || !rh) return null;
        return (lh.y + rh.y) / 2;
    });

    const validYs = hipYSeries.filter((y) => y !== null);
    if (validYs.length < 10) {
        return { repCount: 0, note: "Not enough tracking data captured during squats." };
    }

    const baselineCount = Math.max(3, Math.floor(validYs.length * 0.1));
    const standingBaselineY = avg(validYs.slice(0, baselineCount));
    const DESCENT_THRESHOLD = 0.05; // normalized frame-height units

    const reps = [];
    let inSquat = false;
    let bottomIndex = -1;
    let bottomY = -Infinity;

    for (let i = 0; i < buffer.length; i++) {
        const y = hipYSeries[i];
        if (y === null) continue;
        const descended = y - standingBaselineY;
        if (!inSquat && descended > DESCENT_THRESHOLD) {
            inSquat = true;
            bottomY = y;
            bottomIndex = i;
        } else if (inSquat) {
            if (y > bottomY) { bottomY = y; bottomIndex = i; }
            if (descended < DESCENT_THRESHOLD * 0.5) {
                reps.push(bottomIndex);
                inSquat = false;
                bottomY = -Infinity;
            }
        }
    }
    if (inSquat && bottomIndex >= 0) reps.push(bottomIndex);

    if (reps.length === 0) {
        return { repCount: 0, note: "No squat depth detected — try bending your knees further." };
    }

    const valgusRatios = [], depthPercents = [], kneeAngleDeltas = [], trunkLeans = [];
    let heelLiftDetected = false;

    for (const idx of reps) {
        const s = buffer[idx].lm;
        const lHip = s[23], rHip = s[24], lKnee = s[25], rKnee = s[26], lAnkle = s[27], rAnkle = s[28];
        const lHeel = s[29], rHeel = s[30];
        const lShoulder = s[11], rShoulder = s[12];

        if (lKnee && rKnee && lAnkle && rAnkle) {
            const kneeSep = dist(lKnee, rKnee);
            const ankleSep = dist(lAnkle, rAnkle);
            if (ankleSep > 0.01) valgusRatios.push(kneeSep / ankleSep);
        }
        if (lHip && lKnee && rHip && rKnee) {
            const bottomHipY = (lHip.y + rHip.y) / 2;
            const thighLen = (dist(lHip, lKnee) + dist(rHip, rKnee)) / 2;
            if (thighLen > 0.01) depthPercents.push(((bottomHipY - standingBaselineY) / thighLen) * 100);
        }
        if (lHip && lKnee && lAnkle && rHip && rKnee && rAnkle) {
            const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
            const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
            kneeAngleDeltas.push(Math.abs(leftKneeAngle - rightKneeAngle));
        }
        if (lShoulder && rShoulder && lHip && rHip) {
            const shoulderMid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
            const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
            const dx = shoulderMid.x - hipMid.x;
            const dy = shoulderMid.y - hipMid.y;
            trunkLeans.push(Math.round(Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI)));
        }
        if (lAnkle && lHeel && rAnkle && rHeel) {
            const leftGap = lHeel.y - lAnkle.y;
            const rightGap = rHeel.y - rAnkle.y;
            if (leftGap < 0.01 || rightGap < 0.01) heelLiftDetected = true;
        }
    }

    return {
        repCount: reps.length,
        avgKneeValgusRatio: avg(valgusRatios) !== null ? Math.round(avg(valgusRatios) * 100) / 100 : null,
        avgDepthPercent: avg(depthPercents) !== null ? Math.round(avg(depthPercents)) : null,
        avgKneeSymmetryDelta: avg(kneeAngleDeltas) !== null ? Math.round(avg(kneeAngleDeltas)) : null,
        avgTrunkLean: avg(trunkLeans) !== null ? Math.round(avg(trunkLeans)) : null,
        heelLiftDetected
    };
}

/* ============================================================
   REPORT RENDERING
   ============================================================ */
function flagClass(level) {
    return { good: "flag-good", moderate: "flag-moderate", flag: "flag-alert" }[level] || "flag-good";
}

function buildReportRows() {
    const rows = [];
    const front = assessment.results.front || {};
    const profile = assessment.results.profile || {};
    const squat = assessment.results.squat || {};

    if (front.shoulderSymIdx !== undefined && front.shoulderSymIdx !== null) {
        rows.push({
            label: "Shoulder Symmetry (Front)",
            value: `${front.shoulderSymIdx} idx`,
            level: front.shoulderSymIdx > 5 ? "flag" : front.shoulderSymIdx > 2 ? "moderate" : "good",
            note: "Measures whether one shoulder sits visibly higher than the other while standing relaxed."
        });
    }
    if (front.hipSymIdx !== undefined && front.hipSymIdx !== null) {
        rows.push({
            label: "Hip Symmetry (Front)",
            value: `${front.hipSymIdx} idx`,
            level: front.hipSymIdx > 5 ? "flag" : front.hipSymIdx > 2 ? "moderate" : "good",
            note: "Measures left/right hip height difference — can reflect pelvic obliquity or standing habit."
        });
    }
    if (profile.forwardHeadAngle !== undefined && profile.forwardHeadAngle !== null) {
        rows.push({
            label: "Forward Head Posture Angle",
            value: `${profile.forwardHeadAngle}°`,
            level: profile.forwardHeadAngle > 30 ? "flag" : profile.forwardHeadAngle > 20 ? "moderate" : "good",
            note: "Angle of the ear relative to the shoulder from the side. Larger angles suggest a more forward-jutted head position (\"text neck\")."
        });
    }
    if (profile.trunkLeanAngle !== undefined && profile.trunkLeanAngle !== null) {
        rows.push({
            label: "Trunk Lean (Standing, Profile)",
            value: `${profile.trunkLeanAngle}°`,
            level: profile.trunkLeanAngle > 15 ? "flag" : profile.trunkLeanAngle > 8 ? "moderate" : "good",
            note: "How far the torso leans from vertical while standing relaxed, viewed from the side."
        });
    }

    if (squat.repCount === 0) {
        rows.push({
            label: "Squat Analysis",
            value: "No reps detected",
            level: "moderate",
            note: squat.note || "Try bending your knees further next time so the app can detect squat depth."
        });
    } else if (squat.repCount) {
        rows.push({
            label: "Squat Reps Detected",
            value: `${squat.repCount}`,
            level: "good",
            note: "Number of squat cycles automatically identified from hip movement."
        });
        if (squat.avgKneeValgusRatio !== null) {
            rows.push({
                label: "Knee Tracking (Valgus Ratio)",
                value: `${squat.avgKneeValgusRatio}`,
                level: squat.avgKneeValgusRatio < 0.7 ? "flag" : squat.avgKneeValgusRatio < 0.85 ? "moderate" : "good",
                note: "Knee width vs. ankle width at the bottom of the squat. Values well below 1.0 suggest the knees are drifting inward relative to the feet — often linked to hip/glute control."
            });
        }
        if (squat.avgDepthPercent !== null) {
            rows.push({
                label: "Squat Depth",
                value: `${squat.avgDepthPercent}% of thigh length`,
                level: "good",
                note: "Descriptive measure of how deep the squat went. Not inherently good or bad — depends on your training goals and joint comfort."
            });
        }
        if (squat.avgKneeSymmetryDelta !== null) {
            rows.push({
                label: "Left/Right Squat Symmetry",
                value: `${squat.avgKneeSymmetryDelta}° knee angle delta`,
                level: squat.avgKneeSymmetryDelta > 10 ? "flag" : squat.avgKneeSymmetryDelta > 5 ? "moderate" : "good",
                note: "Difference between left and right knee bend at the bottom of each rep — a proxy for uneven loading between sides."
            });
        }
        if (squat.avgTrunkLean !== null) {
            rows.push({
                label: "Trunk Lean During Squat",
                value: `${squat.avgTrunkLean}°`,
                level: squat.avgTrunkLean > 40 ? "flag" : squat.avgTrunkLean > 25 ? "moderate" : "good",
                note: "Forward torso lean at the bottom of the squat. Excessive lean can shift load onto the lower back."
            });
        }
        if (squat.heelLiftDetected) {
            rows.push({
                label: "Heel Lift",
                value: "Detected",
                level: "moderate",
                note: "Heels appeared to rise off the ground during the squat — often linked to limited ankle mobility."
            });
        }
    }

    return rows;
}

function showReport() {
    phaseBanner.classList.add("hidden");
    metricsPanel.classList.add("hidden");
    const rows = buildReportRows();
    reportListUI.innerHTML = rows.map((r) => `
        <div class="report-row ${flagClass(r.level)}">
            <div class="report-row-top">
                <span class="report-label">${r.label}</span>
                <span class="report-value">${r.value}</span>
            </div>
            <div class="report-note">${r.note}</div>
        </div>
    `).join("");
    reportScreen.classList.remove("hidden");
}

/* ============================================================
   BOOT
   ============================================================ */
initializeTracker();
