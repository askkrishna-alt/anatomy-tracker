// Ensure MediaPipe namespaces are explicitly defined from the window global bundle
const FilesetResolver = qv.FilesetResolver || window.tasksVision.FilesetResolver;
const PoseLandmarker = qv.PoseLandmarker || window.tasksVision.PoseLandmarker;

// Grab HTML UI components
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const headAngleUI = document.getElementById("head-angle");
const pelvicTiltUI = document.getElementById("pelvic-tilt");
const statusUI = document.getElementById("status");

let poseLandmarker = undefined;
let lastVideoTime = -1;

// 1. Download and initialize MediaPipe WebAssembly Module
async function initializeTracker() {
    try {
        statusUI.innerText = "Loading WebAssembly Components...";
        
        // Point directly to the jsDelivr CDN hosting the compiled WASM binaries
        const vision = await FilesetResolver.forVisionTasks(
            "https://jsdelivr.net"
        );
        
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://googleapis.com`,
                delegate: "GPU" // Uses phone GPU hardware acceleration
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

// 2. Stream Android Back Camera into the Video element
function startCamera() {
    const constraints = {
        video: { facingMode: "environment", width: 640, height: 480, frameRate: { ideal: 30 } }
    };
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
            statusUI.innerText = "Scanning Active - Stand Back";
        })
        .catch((err) => {
            statusUI.innerText = "Camera Access Denied by User or System.";
            console.error("Camera Error:", err);
        });
}

// 3. Continuous Tracking Loop
async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        const results = poseLandmarker.detectForVideo(video, startTimeMs);
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0]; // Extract first detected person
            
            drawSkeleton(landmarks);
            calculateBiometrics(landmarks);
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// Helper Trigonometry Math: Calculate angle between two vectors
function calculateAngle(p1, p2, p3) {
    let radians = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) angle = 360.0 - angle;
    return Math.round(angle);
}

function calculateBiometrics(landmarks) {
    // MediaPipe Landmarks: 7 = Left Ear, 11 = Left Shoulder, 23 = Left Hip, 24 = Right Hip
    const leftEar = landmarks[7];
    const leftShoulder = landmarks[11];
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
        const pelvicTiltAngle = Math.round(hipHeightDifference * 100); 
        pelvicTiltUI.innerText = `${pelvicTiltAngle}° Delta`;
    }
}

// Basic Canvas Point rendering system
function drawSkeleton(landmarks) {
    canvasCtx.fillStyle = "#00ffcc";
    // Core tracking joint indexes: Nose, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles
    const trackJoints = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    for (const index of trackJoints) {
        const lm = landmarks[index];
        if (lm) {
            canvasCtx.beginPath();
            canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 5, 0, 2 * Math.PI);
            canvasCtx.fill();
        }
    }
}

// Fire up system on page startup
initializeTracker();
