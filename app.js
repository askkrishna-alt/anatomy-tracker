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
    statusUI.innerText = "Loading WebAssembly...";
    const vision = await FilesetResolver.forVisionTasks(
        "https://jsdelivr.net"
    );
    
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://googleapis.com`,
            delegate: "GPU" // Enforces hardware acceleration on Android Chrome
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    
    statusUI.innerText = "Ready. Accessing Camera...";
    startCamera();
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
            statusUI.innerText = "Scanning Active";
        })
        .catch((err) => {
            statusUI.innerText = "Camera Access Denied";
            console.error(err);
        });
}

// 3. Continuous Tracking Loop
async function predictWebcam() {
    // Sync canvas sizing with the live streaming video
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        // Extract 33 skeletal coordinate sets from current frame
        const results = poseLandmarker.detectForVideo(video, startTimeMs);
        
        // Clear canvas context for clean drawing frame
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            
            // Draw visual UI indicators over the landmarks
            drawSkeleton(landmarks);
            
            // Run clinical mathematical analytics on raw joint matrices
            calculateBiometrics(landmarks);
        }
    }
    // Call the next browser animation window loop frame
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
    // Extract targets (X, Y, Z space coordinates)
    const leftEar = landmarks[7];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    // Profile Assessment: Forward Head Posture (Text Neck)
    // Measures the angular alignment from ear to shoulder relative to vertical axis
    const virtualVerticalPoint = { x: leftShoulder.x, y: leftShoulder.y - 0.2 };
    const forwardHeadAngle = calculateAngle(virtualVerticalPoint, leftShoulder, leftEar);
    headAngleUI.innerText = `${forwardHeadAngle}°`;
    
    // Check if user is slouching forward significantly
    if (forwardHeadAngle > 25) {
        headAngleUI.style.color = "#ff3333"; // Flag deviation in red
    } else {
        headAngleUI.style.color = "#00ffcc";
    }

    // Frontal Assessment: Lateral Pelvic Tilt (Asymmetrical Hip Alignment)
    // Compares left and right hip height levels
    const hipHeightDifference = Math.abs(leftHip.y - rightHip.y);
    const pelvicTiltAngle = Math.round(hipHeightDifference * 100); 
    pelvicTiltUI.innerText = `${pelvicTiltAngle}° Delta`;
}

// Basic Canvas Point rendering system
function drawSkeleton(landmarks) {
    canvasCtx.fillStyle = "#00ffcc";
    // Loop and draw the primary tracked joints on the user screen
    const trackJoints =;
    for (const index of trackJoints) {
        const lm = landmarks[index];
        canvasCtx.beginPath();
        canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 5, 0, 2 * Math.PI);
        canvasCtx.fill();
    }
}

// Fire up system on page startup
initializeTracker();
