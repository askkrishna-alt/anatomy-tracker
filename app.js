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
const NEEDED_INDICES = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

function snapshotLandmarks(landmarks) {
    const snap = {};
    for (const i of NEEDED_INDICES) {
        const lm = landmarks[i];
        // visibility (0-1) is MediaPipe's own per-landmark confidence score,
        // kept here to surface as a tracking-confidence metric later.
        snap[i] = lm ? { x: lm.x, y: lm.y, v: typeof lm.visibility === "number" ? lm.visibility : null } : null;
    }
    return snap;
}

const PHASES = [
    {
        key: "front",
        label: "Front Stance",
        instruction: "Stand facing the camera, feet hip-width apart, arms relaxed at your sides.",
        countdownSec: 5,
        captureMs: 2500
    },
    {
        key: "profile",
        label: "Side Stance",
        instruction: "Turn 90° so your side faces the camera. Stand naturally.",
        countdownSec: 5,
        captureMs: 2500
    },
    {
        key: "squat",
        label: "Squats",
        instruction: "Face the camera again. Perform 3 slow, controlled squats.",
        countdownSec: 5,
        captureMs: 20000
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
    const shoulderDeltas = [], hipDeltas = [], headTilts = [], headShifts = [];
    const clavicleAngles = [], hipShifts = [], kneeAlignRatios = [], poplitealDeltas = [];

    for (const frame of buffer) {
        const s = frame.lm;
        const nose = s[0], lEar = s[7], rEar = s[8];
        const lShoulder = s[11], rShoulder = s[12];
        const lHip = s[23], rHip = s[24];
        const lKnee = s[25], rKnee = s[26];
        const lAnkle = s[27], rAnkle = s[28];

        if (lShoulder && rShoulder) {
            shoulderDeltas.push(Math.abs(lShoulder.y - rShoulder.y));
            // Clavicle Angle: true angle of the shoulder line from horizontal
            const dx = rShoulder.x - lShoulder.x;
            const dy = rShoulder.y - lShoulder.y;
            clavicleAngles.push(Math.round(Math.abs((Math.atan2(dy, dx) * 180) / Math.PI)));
        }
        if (lHip && rHip) {
            hipDeltas.push(Math.abs(lHip.y - rHip.y));
        }
        if (lEar && rEar) {
            headTilts.push(Math.abs(lEar.y - rEar.y));
        }
        if (nose && lShoulder && rShoulder) {
            const shoulderMidX = (lShoulder.x + rShoulder.x) / 2;
            headShifts.push(Math.abs(nose.x - shoulderMidX));
        }
        if (lHip && rHip && lAnkle && rAnkle) {
            const hipMidX = (lHip.x + rHip.x) / 2;
            const ankleMidX = (lAnkle.x + rAnkle.x) / 2;
            hipShifts.push(Math.abs(hipMidX - ankleMidX));
        }
        if (lKnee && rKnee && lAnkle && rAnkle) {
            const kneeSep = dist(lKnee, rKnee);
            const ankleSep = dist(lAnkle, rAnkle);
            if (ankleSep > 0.01) kneeAlignRatios.push(kneeSep / ankleSep);
        }
        if (lKnee && rKnee) {
            poplitealDeltas.push(Math.abs(lKnee.y - rKnee.y));
        }
    }

    return {
        shoulderSymIdx: avg(shoulderDeltas) !== null ? Math.round(avg(shoulderDeltas) * 100) : null,
        clavicleAngle: avg(clavicleAngles) !== null ? Math.round(avg(clavicleAngles)) : null,
        hipSymIdx: avg(hipDeltas) !== null ? Math.round(avg(hipDeltas) * 100) : null,
        headTiltIdx: avg(headTilts) !== null ? Math.round(avg(headTilts) * 100) : null,
        headShiftIdx: avg(headShifts) !== null ? Math.round(avg(headShifts) * 100) : null,
        hipShiftIdx: avg(hipShifts) !== null ? Math.round(avg(hipShifts) * 100) : null,
        kneeAlignRatio: avg(kneeAlignRatios) !== null ? Math.round(avg(kneeAlignRatios) * 100) / 100 : null,
        poplitealDeltaIdx: avg(poplitealDeltas) !== null ? Math.round(avg(poplitealDeltas) * 100) : null,
        trackingConfidence: avgVisibility(buffer),
        frameCount: buffer.length
    };
}

// Average MediaPipe per-landmark visibility score across all captured frames
// and tracked joints — a rough proxy for how reliable the rest of this
// phase's measurements are (low light, partial occlusion, out-of-frame limbs).
function avgVisibility(buffer) {
    const scores = [];
    for (const frame of buffer) {
        for (const i of NEEDED_INDICES) {
            const lm = frame.lm[i];
            if (lm && lm.v !== null) scores.push(lm.v);
        }
    }
    return scores.length ? Math.round(avg(scores) * 100) : null;
}

function analyzeProfileHold(buffer) {
    const headAngles = [], trunkAngles = [], plumbDeviations = [], kneeLineOffsets = [];

    for (const frame of buffer) {
        const s = frame.lm;
        const nose = s[0];
        const ear = s[7] || s[8];
        const shoulder = s[11] || s[12];
        const hip = s[23] || s[24];
        const knee = s[25] || s[26];
        const ankle = s[27] || s[28];

        if (ear && shoulder) {
            const virtualVerticalPoint = { x: shoulder.x, y: shoulder.y - 0.2 };
            headAngles.push(calculateAngle(virtualVerticalPoint, shoulder, ear));
        }
        if (shoulder && hip) {
            const dx = shoulder.x - hip.x;
            const dy = shoulder.y - hip.y;
            trunkAngles.push(Math.round(Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI)));
        }
        // Plumb Line Deviation: how far ear/shoulder/hip stray horizontally
        // from the ankle (ground reference), the classic postural plumb-line check.
        if (ear && shoulder && hip && ankle) {
            const ref = ankle.x;
            const dev = (Math.abs(ear.x - ref) + Math.abs(shoulder.x - ref) + Math.abs(hip.x - ref)) / 3;
            plumbDeviations.push(dev);
        }
        // Knee Hyperextension Screen: expected knee x if hip-ankle were a
        // straight line, compared to actual knee x, signed by the body's
        // own anterior direction (nose is anterior to the ear).
        if (hip && knee && ankle && nose && ear && ankle.y !== hip.y) {
            const t = (knee.y - hip.y) / (ankle.y - hip.y);
            const expectedX = hip.x + (ankle.x - hip.x) * t;
            const rawOffset = knee.x - expectedX;
            const anteriorSign = Math.sign(nose.x - ear.x) || 1;
            kneeLineOffsets.push(rawOffset * anteriorSign);
        }
    }

    return {
        forwardHeadAngle: avg(headAngles) !== null ? Math.round(avg(headAngles)) : null,
        trunkLeanAngle: avg(trunkAngles) !== null ? Math.round(avg(trunkAngles)) : null,
        plumbLineDeviationIdx: avg(plumbDeviations) !== null ? Math.round(avg(plumbDeviations) * 100) : null,
        kneeLineOffsetIdx: avg(kneeLineOffsets) !== null ? Math.round(avg(kneeLineOffsets) * 100) : null,
        trackingConfidence: avgVisibility(buffer),
        frameCount: buffer.length
    };
}

// Simple centered moving-average smoother — reduces per-frame landmark jitter
// that would otherwise cause the threshold crossing to bounce and double-count reps.
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

function analyzeSquat(buffer) {
    const rawHipYSeries = buffer.map((f) => {
        const lh = f.lm[23], rh = f.lm[24];
        if (!lh || !rh) return null;
        return (lh.y + rh.y) / 2;
    });

    const validYs = rawHipYSeries.filter((y) => y !== null);
    if (validYs.length < 10) {
        return { repCount: 0, note: "Not enough tracking data captured during squats." };
    }

    const hipYSeries = smoothSeries(rawHipYSeries, 5);
    const baselineCount = Math.max(3, Math.floor(validYs.length * 0.1));
    const standingBaselineY = avg(validYs.slice(0, baselineCount));

    const ENTER_THRESHOLD = 0.06;   // must descend this far (normalized) to register as squatting
    const EXIT_THRESHOLD = 0.03;    // must rise back above this to register as standing again (hysteresis gap prevents bounce)
    const MIN_REP_DURATION_MS = 350; // filters out brief jitter blips that aren't real reps

    const reps = [];
    let inSquat = false;
    let bottomIndex = -1;
    let bottomY = -Infinity;
    let entryTimeMs = 0;

    for (let i = 0; i < buffer.length; i++) {
        const y = hipYSeries[i];
        if (y === null) continue;
        const descended = y - standingBaselineY;
        if (!inSquat && descended > ENTER_THRESHOLD) {
            inSquat = true;
            bottomY = y;
            bottomIndex = i;
            entryTimeMs = buffer[i].t;
        } else if (inSquat) {
            if (y > bottomY) { bottomY = y; bottomIndex = i; }
            if (descended < EXIT_THRESHOLD) {
                const durationMs = buffer[i].t - entryTimeMs;
                if (durationMs >= MIN_REP_DURATION_MS) {
                    reps.push(bottomIndex);
                }
                inSquat = false;
                bottomY = -Infinity;
            }
        }
    }
    if (inSquat && bottomIndex >= 0) {
        const durationMs = buffer[buffer.length - 1].t - entryTimeMs;
        if (durationMs >= MIN_REP_DURATION_MS) reps.push(bottomIndex);
    }

    if (reps.length === 0) {
        return { repCount: 0, note: "No squat depth detected — try bending your knees further." };
    }

    // --- Standing baseline reference values (from the pre-squat frames) ---
    const baselineFrames = buffer.slice(0, baselineCount);
    const baseAvg = (getter) => {
        const vals = baselineFrames.map(getter).filter((v) => v !== null);
        return avg(vals);
    };
    const standingHipMidX = baseAvg((f) => (f.lm[23] && f.lm[24]) ? (f.lm[23].x + f.lm[24].x) / 2 : null);
    const standingAnkleMidX = baseAvg((f) => (f.lm[27] && f.lm[28]) ? (f.lm[27].x + f.lm[28].x) / 2 : null);
    const standingKneeMidX = baseAvg((f) => (f.lm[25] && f.lm[26]) ? (f.lm[25].x + f.lm[26].x) / 2 : null);
    const standingAnkleSep = baseAvg((f) => (f.lm[27] && f.lm[28]) ? dist(f.lm[27], f.lm[28]) : null);
    const standingShoulderSep = baseAvg((f) => (f.lm[11] && f.lm[12]) ? dist(f.lm[11], f.lm[12]) : null);
    const stanceWidthRatio = (standingAnkleSep !== null && standingShoulderSep > 0.01)
        ? Math.round((standingAnkleSep / standingShoulderSep) * 100) / 100 : null;
    const footFlareVals = baselineFrames.map((f) => {
        const s = f.lm;
        const flares = [];
        if (s[27] && s[31]) flares.push(Math.abs(s[31].x - s[27].x));
        if (s[28] && s[32]) flares.push(Math.abs(s[32].x - s[28].x));
        return flares.length ? avg(flares) : null;
    }).filter((v) => v !== null);
    const footFlareIdx = footFlareVals.length ? Math.round(avg(footFlareVals) * 100) : null;

    // --- Continuous (whole-movement) tracking across every captured frame ---
    let peakLateralShift = 0, sumLateralShift = 0, lateralShiftCount = 0;
    let minValgusRatio = Infinity;
    let shoulderXs = [];
    let peakPelvicDeviation = 0;
    let leftPatellarXs = [], rightPatellarXs = [];
    let kneeAngleSeries = [];

    for (const frame of buffer) {
        const s = frame.lm;
        const lHip = s[23], rHip = s[24], lKnee = s[25], rKnee = s[26], lAnkle = s[27], rAnkle = s[28];
        const lFoot = s[31], rFoot = s[32];
        const lShoulder = s[11], rShoulder = s[12];

        if (lHip && rHip && lAnkle && rAnkle) {
            const hipMidX = (lHip.x + rHip.x) / 2;
            const ankleMidX = (lAnkle.x + rAnkle.x) / 2;
            const shift = Math.abs(hipMidX - ankleMidX);
            peakLateralShift = Math.max(peakLateralShift, shift);
            sumLateralShift += shift;
            lateralShiftCount++;
        }
        if (lKnee && rKnee && lAnkle && rAnkle) {
            const ankleSep = dist(lAnkle, rAnkle);
            if (ankleSep > 0.01) minValgusRatio = Math.min(minValgusRatio, dist(lKnee, rKnee) / ankleSep);
        }
        if (lShoulder && rShoulder) shoulderXs.push((lShoulder.x + rShoulder.x) / 2);
        if (lHip && rHip) peakPelvicDeviation = Math.max(peakPelvicDeviation, Math.abs(lHip.y - rHip.y));
        if (lKnee && lFoot) leftPatellarXs.push(lKnee.x - lFoot.x);
        if (rKnee && rFoot) rightPatellarXs.push(rKnee.x - rFoot.x);
        if (lHip && lKnee && lAnkle && rHip && rKnee && rAnkle) {
            kneeAngleSeries.push({ t: frame.t, angle: (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2 });
        }
    }

    const range = (arr) => arr.length ? Math.max(...arr) - Math.min(...arr) : null;
    const shoulderDriftIdx = range(shoulderXs) !== null ? Math.round(range(shoulderXs) * 100) : null;
    const patellarTrackIdx = (range(leftPatellarXs) !== null && range(rightPatellarXs) !== null)
        ? Math.round(((range(leftPatellarXs) + range(rightPatellarXs)) / 2) * 100) : null;

    let peakAngularVelocity = 0;
    for (let i = 1; i < kneeAngleSeries.length; i++) {
        const dtSec = (kneeAngleSeries[i].t - kneeAngleSeries[i - 1].t) / 1000;
        if (dtSec > 0) {
            const v = Math.abs(kneeAngleSeries[i].angle - kneeAngleSeries[i - 1].angle) / dtSec;
            peakAngularVelocity = Math.max(peakAngularVelocity, v);
        }
    }

    // --- Per-rep (bottom-of-squat) metrics ---
    const valgusRatios = [], depthPercents = [], kneeAngleDeltas = [], trunkLeans = [];
    const hipFlexionAngles = [], torsoShinDeltas = [], forwardKneeTravels = [], lateralHipShiftsAtDepth = [];
    const asymmetryPcts = [];
    const depthCategories = [];
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
        let bottomHipY = null, thighLen = null;
        if (lHip && lKnee && rHip && rKnee) {
            bottomHipY = (lHip.y + rHip.y) / 2;
            thighLen = (dist(lHip, lKnee) + dist(rHip, rKnee)) / 2;
            if (thighLen > 0.01) depthPercents.push(((bottomHipY - standingBaselineY) / thighLen) * 100);
        }
        let leftKneeAngle = null, rightKneeAngle = null;
        if (lHip && lKnee && lAnkle && rHip && rKnee && rAnkle) {
            leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
            rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
            const delta = Math.abs(leftKneeAngle - rightKneeAngle);
            kneeAngleDeltas.push(delta);
            const angleAvg = (leftKneeAngle + rightKneeAngle) / 2;
            if (angleAvg > 0) asymmetryPcts.push((delta / angleAvg) * 100);
        }
        let shoulderMid = null, hipMid = null, trunkAngleAtBottom = null;
        if (lShoulder && rShoulder && lHip && rHip) {
            shoulderMid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
            hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
            const dx = shoulderMid.x - hipMid.x;
            const dy = shoulderMid.y - hipMid.y;
            trunkAngleAtBottom = Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI);
            trunkLeans.push(Math.round(trunkAngleAtBottom));
        }
        if (lAnkle && lHeel && rAnkle && rHeel) {
            const leftGap = lHeel.y - lAnkle.y;
            const rightGap = rHeel.y - rAnkle.y;
            if (leftGap < 0.01 || rightGap < 0.01) heelLiftDetected = true;
        }
        // Hip flexion angle at bottom (shoulder-hip-knee)
        if (lShoulder && lHip && lKnee && rShoulder && rHip && rKnee) {
            const leftHipFlex = calculateAngle(lShoulder, lHip, lKnee);
            const rightHipFlex = calculateAngle(rShoulder, rHip, rKnee);
            hipFlexionAngles.push((leftHipFlex + rightHipFlex) / 2);
        }
        // Torso-vs-shin parallelism: shin angle from vertical vs torso angle from vertical
        if (lKnee && lAnkle && trunkAngleAtBottom !== null) {
            const shinDx = lKnee.x - lAnkle.x;
            const shinDy = lKnee.y - lAnkle.y;
            const shinAngle = Math.abs((Math.atan2(shinDx, -shinDy) * 180) / Math.PI);
            torsoShinDeltas.push(Math.round(Math.abs(shinAngle - trunkAngleAtBottom)));
        }
        // Forward knee travel relative to standing baseline (ankle dorsiflexion proxy)
        if (lKnee && rKnee && lAnkle && rAnkle && standingKneeMidX !== null && standingAnkleMidX !== null) {
            const kneeMidXBottom = (lKnee.x + rKnee.x) / 2;
            const ankleMidXBottom = (lAnkle.x + rAnkle.x) / 2;
            const travel = (kneeMidXBottom - ankleMidXBottom) - (standingKneeMidX - standingAnkleMidX);
            forwardKneeTravels.push(Math.abs(travel));
        }
        // Lateral hip shift at this rep's deepest point, relative to standing
        if (lHip && rHip && standingHipMidX !== null) {
            const hipMidXBottom = (lHip.x + rHip.x) / 2;
            lateralHipShiftsAtDepth.push(Math.abs(hipMidXBottom - standingHipMidX));
        }
        // Squat depth category
        if (bottomHipY !== null && lKnee && rKnee) {
            const kneeYAvg = (lKnee.y + rKnee.y) / 2;
            const band = 0.02;
            if (bottomHipY < kneeYAvg - band) depthCategories.push("Above Parallel");
            else if (bottomHipY > kneeYAvg + band) depthCategories.push("Below Parallel (Deep)");
            else depthCategories.push("Parallel");
        }
    }

    const modeOf = (arr) => {
        if (!arr.length) return null;
        const counts = {};
        arr.forEach((v) => { counts[v] = (counts[v] || 0) + 1; });
        return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    };

    return {
        repCount: reps.length,
        avgKneeValgusRatio: avg(valgusRatios) !== null ? Math.round(avg(valgusRatios) * 100) / 100 : null,
        peakKneeValgusRatio: minValgusRatio !== Infinity ? Math.round(minValgusRatio * 100) / 100 : null,
        avgDepthPercent: avg(depthPercents) !== null ? Math.round(avg(depthPercents)) : null,
        depthCategory: modeOf(depthCategories),
        avgKneeSymmetryDelta: avg(kneeAngleDeltas) !== null ? Math.round(avg(kneeAngleDeltas)) : null,
        avgTrunkLean: avg(trunkLeans) !== null ? Math.round(avg(trunkLeans)) : null,
        avgHipFlexionAngle: avg(hipFlexionAngles) !== null ? Math.round(avg(hipFlexionAngles)) : null,
        avgTorsoShinDelta: avg(torsoShinDeltas) !== null ? Math.round(avg(torsoShinDeltas)) : null,
        avgForwardKneeTravelIdx: avg(forwardKneeTravels) !== null ? Math.round(avg(forwardKneeTravels) * 100) : null,
        avgLateralHipShiftIdx: avg(lateralHipShiftsAtDepth) !== null ? Math.round(avg(lateralHipShiftsAtDepth) * 100) : null,
        asymmetryPercentIndex: avg(asymmetryPcts) !== null ? Math.round(avg(asymmetryPcts)) : null,
        stanceWidthRatio,
        footFlareIdx,
        peakLateralShiftIdx: Math.round(peakLateralShift * 100),
        avgLateralShiftIdx: lateralShiftCount ? Math.round((sumLateralShift / lateralShiftCount) * 100) : null,
        shoulderDriftIdx,
        patellarTrackIdx,
        peakPelvicDeviationIdx: Math.round(peakPelvicDeviation * 100),
        peakAngularVelocity: Math.round(peakAngularVelocity),
        trackingConfidence: avgVisibility(buffer),
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
    const push = (section, row) => rows.push({ section, ...row });

    /* ---------- STATIC POSTURE — FRONT ---------- */
    if (front.trackingConfidence != null) {
        push("Static Posture — Front", {
            label: "Tracking Confidence",
            value: `${front.trackingConfidence}%`,
            level: front.trackingConfidence < 60 ? "flag" : front.trackingConfidence < 80 ? "moderate" : "good",
            note: "Average MediaPipe landmark visibility score for this phase. Low values mean poor lighting, occlusion, or partial framing — weight this phase's other numbers accordingly."
        });
    }
    if (front.headTiltIdx != null) {
        push("Static Posture — Front", {
            label: "Head Tilt (Lateral)",
            value: `${front.headTiltIdx} idx`,
            level: front.headTiltIdx > 5 ? "flag" : front.headTiltIdx > 2 ? "moderate" : "good",
            note: "Left/right ear height difference — lateral cranial tilt relative to the cervical spine."
        });
    }
    if (front.headShiftIdx != null) {
        push("Static Posture — Front", {
            label: "Head Shift (Lateral)",
            value: `${front.headShiftIdx} idx`,
            level: front.headShiftIdx > 5 ? "flag" : front.headShiftIdx > 2 ? "moderate" : "good",
            note: "Horizontal offset of the nose from the shoulder midline — lateral head translation, distinct from tilt."
        });
    }
    if (front.shoulderSymIdx != null) {
        push("Static Posture — Front", {
            label: "Shoulder Symmetry (AC Joint Height)",
            value: `${front.shoulderSymIdx} idx`,
            level: front.shoulderSymIdx > 5 ? "flag" : front.shoulderSymIdx > 2 ? "moderate" : "good",
            note: "Height differential between left/right shoulder landmarks (acromioclavicular joint proxy)."
        });
    }
    if (front.clavicleAngle != null) {
        push("Static Posture — Front", {
            label: "Clavicle / Shoulder Girdle Angle",
            value: `${front.clavicleAngle}°`,
            level: front.clavicleAngle > 8 ? "flag" : front.clavicleAngle > 4 ? "moderate" : "good",
            note: "Angle of the shoulder line from horizontal — shoulder girdle elevation or depression on one side."
        });
    }
    if (front.hipSymIdx != null) {
        push("Static Posture — Front", {
            label: "Pelvic Leveling",
            value: `${front.hipSymIdx} idx`,
            level: front.hipSymIdx > 5 ? "flag" : front.hipSymIdx > 2 ? "moderate" : "good",
            note: "Left/right hip landmark height difference (proxy for ASIS/iliac crest leveling — see accuracy note below)."
        });
    }
    if (front.hipShiftIdx != null) {
        push("Static Posture — Front", {
            label: "Hip Shift",
            value: `${front.hipShiftIdx} idx`,
            level: front.hipShiftIdx > 5 ? "flag" : front.hipShiftIdx > 2 ? "moderate" : "good",
            note: "Lateral translation of the pelvis midpoint relative to the midline of the feet."
        });
    }
    if (front.kneeAlignRatio != null) {
        push("Static Posture — Front", {
            label: "Knee Alignment (Standing)",
            value: `${front.kneeAlignRatio}`,
            level: front.kneeAlignRatio < 0.85 || front.kneeAlignRatio > 1.15 ? "moderate" : "good",
            note: "Inter-knee vs. inter-ankle distance ratio while standing. ~1.0 is neutral; well below suggests static varus/valgus tendency."
        });
    }
    if (front.poplitealDeltaIdx != null) {
        push("Static Posture — Front", {
            label: "Popliteal Crease Alignment",
            value: `${front.poplitealDeltaIdx} idx`,
            level: front.poplitealDeltaIdx > 5 ? "flag" : front.poplitealDeltaIdx > 2 ? "moderate" : "good",
            note: "Left/right knee-crease height symmetry — a rough proxy since the true crease isn't a distinct landmark in this model."
        });
    }

    /* ---------- STATIC POSTURE — PROFILE ---------- */
    if (profile.trackingConfidence != null) {
        push("Static Posture — Profile", {
            label: "Tracking Confidence",
            value: `${profile.trackingConfidence}%`,
            level: profile.trackingConfidence < 60 ? "flag" : profile.trackingConfidence < 80 ? "moderate" : "good",
            note: "Average MediaPipe landmark visibility score for the side-view phase."
        });
    }
    if (profile.forwardHeadAngle != null) {
        push("Static Posture — Profile", {
            label: "Forward Head Posture Angle",
            value: `${profile.forwardHeadAngle}°`,
            level: profile.forwardHeadAngle > 30 ? "flag" : profile.forwardHeadAngle > 20 ? "moderate" : "good",
            note: "Ear-to-shoulder angle from the side (craniovertebral angle proxy). Smaller angles indicate more forward head carriage (\"text neck\")."
        });
    }
    if (profile.trunkLeanAngle != null) {
        push("Static Posture — Profile", {
            label: "Trunk Lean (Standing)",
            value: `${profile.trunkLeanAngle}°`,
            level: profile.trunkLeanAngle > 15 ? "flag" : profile.trunkLeanAngle > 8 ? "moderate" : "good",
            note: "Shoulder-to-hip line angle from vertical while standing relaxed, viewed from the side."
        });
    }
    if (profile.plumbLineDeviationIdx != null) {
        push("Static Posture — Profile", {
            label: "Plumb Line Deviation",
            value: `${profile.plumbLineDeviationIdx} idx`,
            level: profile.plumbLineDeviationIdx > 6 ? "flag" : profile.plumbLineDeviationIdx > 3 ? "moderate" : "good",
            note: "Average horizontal deviation of ear/shoulder/hip from the ankle (ground reference) — the classic postural plumb-line check, averaged across those landmarks."
        });
    }
    if (profile.kneeLineOffsetIdx != null) {
        push("Static Posture — Profile", {
            label: "Knee Alignment (Sagittal) — Hyperextension Screen",
            value: `${profile.kneeLineOffsetIdx} idx`,
            level: profile.kneeLineOffsetIdx < -4 ? "flag" : profile.kneeLineOffsetIdx < -2 ? "moderate" : "good",
            note: "Knee position relative to the straight hip-ankle line, signed by facing direction. Negative values mean the knee sits posterior to that line — a screening flag for possible genu recurvatum, worth visual confirmation."
        });
    }

    /* ---------- DYNAMIC SQUAT ASSESSMENT ---------- */
    if (squat.repCount === 0) {
        push("Dynamic Squat Assessment", {
            label: "Squat Analysis",
            value: "No reps detected",
            level: "moderate",
            note: squat.note || "Try bending your knees further next time so the app can detect squat depth."
        });
    } else if (squat.repCount) {
        push("Dynamic Squat Assessment", {
            label: "Squat Reps Detected",
            value: `${squat.repCount}`,
            level: "good",
            note: "Number of squat cycles automatically identified from smoothed hip-height movement."
        });
        if (squat.trackingConfidence != null) {
            push("Dynamic Squat Assessment", {
                label: "Tracking Confidence",
                value: `${squat.trackingConfidence}%`,
                level: squat.trackingConfidence < 60 ? "flag" : squat.trackingConfidence < 80 ? "moderate" : "good",
                note: "Average MediaPipe landmark visibility score across the whole squat capture."
            });
        }
        if (squat.stanceWidthRatio != null) {
            push("Dynamic Squat Assessment", {
                label: "Stance Width (vs. Shoulder Width)",
                value: `${squat.stanceWidthRatio}x`,
                level: "good",
                note: "Baseline heel-to-heel stance width divided by shoulder width, measured before descent began."
            });
        }
        if (squat.footFlareIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Foot Flare (Front-View Proxy)",
                value: `${squat.footFlareIdx} idx`,
                level: "good",
                note: "Approximate toe-out indicator from a front camera view — horizontal offset of forefoot from ankle. Lower confidence than a top-down or rear view."
            });
        }
        if (squat.avgKneeValgusRatio != null) {
            push("Dynamic Squat Assessment", {
                label: "Knee Valgus/Varus (avg at depth)",
                value: `${squat.avgKneeValgusRatio}`,
                level: squat.avgKneeValgusRatio < 0.7 ? "flag" : squat.avgKneeValgusRatio < 0.85 ? "moderate" : "good",
                note: "Knee width vs. ankle width, averaged across rep bottoms. Below 1.0 suggests knees drifting inward relative to the feet."
            });
        }
        if (squat.peakKneeValgusRatio != null) {
            push("Dynamic Squat Assessment", {
                label: "Knee Valgus/Varus (worst moment, full rep)",
                value: `${squat.peakKneeValgusRatio}`,
                level: squat.peakKneeValgusRatio < 0.7 ? "flag" : squat.peakKneeValgusRatio < 0.85 ? "moderate" : "good",
                note: "Lowest valgus ratio seen at any point during descent or ascent, not just at the bottom — often where the worst collapse actually happens."
            });
        }
        if (squat.patellarTrackIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Patellar Tracking Consistency",
                value: `${squat.patellarTrackIdx} idx`,
                level: squat.patellarTrackIdx > 8 ? "moderate" : "good",
                note: "Range of knee-to-forefoot horizontal offset through the whole rep. Larger values mean the knee's path over the foot varied more rep-to-rep."
            });
        }
        if (squat.depthCategory) {
            push("Dynamic Squat Assessment", {
                label: "Squat Depth Category",
                value: squat.depthCategory,
                level: "good",
                note: "Most common category across reps, based on hip crease vs. knee height at the bottom. Descriptive only — depth targets depend on the individual's goals."
            });
        }
        if (squat.avgDepthPercent != null) {
            push("Dynamic Squat Assessment", {
                label: "Squat Depth (% of thigh length)",
                value: `${squat.avgDepthPercent}%`,
                level: "good",
                note: "Hip descent expressed as a percentage of thigh length, for tracking depth trends over time."
            });
        }
        if (squat.avgHipFlexionAngle != null) {
            push("Dynamic Squat Assessment", {
                label: "Hip Flexion Angle (at depth)",
                value: `${squat.avgHipFlexionAngle}°`,
                level: "good",
                note: "Shoulder-hip-knee angle at the bottom of the squat, averaged left/right."
            });
        }
        if (squat.avgKneeSymmetryDelta != null) {
            push("Dynamic Squat Assessment", {
                label: "Left/Right Knee Flexion Symmetry",
                value: `${squat.avgKneeSymmetryDelta}°`,
                level: squat.avgKneeSymmetryDelta > 10 ? "flag" : squat.avgKneeSymmetryDelta > 5 ? "moderate" : "good",
                note: "Difference between left and right knee bend at rep bottoms — a proxy for uneven loading between sides."
            });
        }
        if (squat.asymmetryPercentIndex != null) {
            push("Dynamic Squat Assessment", {
                label: "Asymmetry Percentage Index",
                value: `${squat.asymmetryPercentIndex}%`,
                level: squat.asymmetryPercentIndex > 15 ? "flag" : squat.asymmetryPercentIndex > 8 ? "moderate" : "good",
                note: "Left-vs-right knee flexion difference as a percentage of the average, a normalized asymmetry ratio."
            });
        }
        if (squat.avgTrunkLean != null) {
            push("Dynamic Squat Assessment", {
                label: "Forward Lean Angle (at depth)",
                value: `${squat.avgTrunkLean}°`,
                level: squat.avgTrunkLean > 40 ? "flag" : squat.avgTrunkLean > 25 ? "moderate" : "good",
                note: "Trunk angle from vertical at the bottom of the squat. Excessive lean can shift load onto the lower back."
            });
        }
        if (squat.avgTorsoShinDelta != null) {
            push("Dynamic Squat Assessment", {
                label: "Torso vs. Shin Parallelism",
                value: `${squat.avgTorsoShinDelta}°`,
                level: squat.avgTorsoShinDelta > 25 ? "moderate" : "good",
                note: "Angular difference between the torso line and the shin line at depth. Large gaps often mean a more hip-dominant or knee-dominant squat pattern."
            });
        }
        if (squat.avgForwardKneeTravelIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Forward Knee Travel (Ankle Mobility Proxy)",
                value: `${squat.avgForwardKneeTravelIdx} idx`,
                level: "good",
                note: "Change in the knee's forward position over the ankle, from standing to depth. Very low values alongside heel lift can suggest limited ankle dorsiflexion."
            });
        }
        if (squat.avgLateralHipShiftIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Lateral Hip Shift (at depth)",
                value: `${squat.avgLateralHipShiftIdx} idx`,
                level: squat.avgLateralHipShiftIdx > 6 ? "flag" : squat.avgLateralHipShiftIdx > 3 ? "moderate" : "good",
                note: "Hip midpoint horizontal drift from the standing baseline, measured at the bottom of each rep."
            });
        }
        if (squat.peakLateralShiftIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Weight Distribution Symmetry (peak, full rep)",
                value: `${squat.peakLateralShiftIdx} idx`,
                level: squat.peakLateralShiftIdx > 8 ? "flag" : squat.peakLateralShiftIdx > 4 ? "moderate" : "good",
                note: "Largest hip-vs-feet-midline lateral shift seen anywhere during descent or ascent, not just at the bottom."
            });
        }
        if (squat.peakPelvicDeviationIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Pelvic Deviation (peak, full rep)",
                value: `${squat.peakPelvicDeviationIdx} idx`,
                level: squat.peakPelvicDeviationIdx > 6 ? "flag" : squat.peakPelvicDeviationIdx > 3 ? "moderate" : "good",
                note: "Largest left/right hip height difference seen at any point in the movement — asymmetric pelvic drop can occur mid-transition rather than only at depth."
            });
        }
        if (squat.shoulderDriftIdx != null) {
            push("Dynamic Squat Assessment", {
                label: "Shoulder Path Drift (\"Bar Path\")",
                value: `${squat.shoulderDriftIdx} idx`,
                level: squat.shoulderDriftIdx > 8 ? "moderate" : "good",
                note: "Horizontal drift of the shoulder midpoint across the whole rep — a bodyweight proxy for barbell bar-path consistency."
            });
        }
        if (squat.peakAngularVelocity != null) {
            push("Dynamic Squat Assessment", {
                label: "Peak Knee Angular Velocity",
                value: `${squat.peakAngularVelocity}°/s`,
                level: "good",
                note: "Fastest frame-to-frame rate of knee angle change captured — reflects movement speed/control, not inherently good or bad."
            });
        }
        if (squat.heelLiftDetected) {
            push("Dynamic Squat Assessment", {
                label: "Heel Lift",
                value: "Detected",
                level: "moderate",
                note: "Heels appeared to rise off the ground during the squat — often linked to limited ankle mobility. This is the roughest metric in the report; verify visually."
            });
        }
    }

    return rows;
}

function showReport() {
    phaseBanner.classList.add("hidden");
    metricsPanel.classList.add("hidden");
    const rows = buildReportRows();

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

/* ============================================================
   BOOT
   ============================================================ */
initializeTracker();
