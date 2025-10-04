function get_max(arr) {
    if (!Array.isArray(arr)) return arr;
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (m < arr[i]) m = arr[i];
    }
    return m;
}
function get_min(arr) {
    if (!Array.isArray(arr)) return arr;
    let m = Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (m > arr[i]) m = arr[i];
    }
    return m;
}
function get_sum(arr) {
    if (!Array.isArray(arr)) return arr;
    let s = 0;
    for (let i = 0; i < arr.length; i++) {
        s += arr[i];
    }
    return s;
}
function computeDistance(vec1, vec2) {
    return Math.sqrt(vec1.reduce((sum, val, i) => sum + (val - vec2[i]) ** 2, 0));
}
//*******************************************************************************//
function computePairwiseDistances(descriptors) { //compute all pairwise distances
    const userIds = Object.keys(descriptors);
    const pairs_minDistances = [];
    const same_maxDistances = [];

    // Iterate over all unique pairs of users
    for (let i = 0; i < userIds.length; i++) {
        const user1 = userIds[i];
        const desc1 = descriptors[user1];
        for (let j = i + 1; j < userIds.length; j++) {            
            const user2 = userIds[j];
            const desc2 = descriptors[user2];
            // Compute minimum distance between any descriptor of user1 and user2
            let minDistance = Infinity;
            for (let d1 of desc1) {
                for (let d2 of desc2) {
                    const distance = computeDistance(d1, d2);
                    minDistance = Math.min(minDistance, distance);
                }
            }
            pairs_minDistances.push(minDistance);
        }
        // Compute maximum distance between any descriptor of same user1
        let maxDistance = -Infinity;
        for (let k = 0; k < desc1.length; k++) {
            for (let l = k + 1; l < desc1.length; l++) {
                const distance = computeDistance(desc1[k], desc1[l]);
                maxDistance = Math.max(maxDistance, distance);
            }
        }
        same_maxDistances.push(maxDistance);        
    }
    return {pairs:pairs_minDistances, same:same_maxDistances, filter:null};
}
function l2Normalize(vector) {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / norm);
}
function cropSquareToCanvas(img, box, size=160, pad=0.0) {
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const half = Math.max(box.width, box.height) * (1 + pad) / 2;
    const sx = Math.max(0, cx - half);
    const sy = Math.max(0, cy - half);
    const sw = Math.min(img.width  - sx, half * 2);
    const sh = Math.min(img.height - sy, half * 2);
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);
    return c;
}
function prewhiten(t) {
    const mean = tf.mean(t);
    const std  = tf.sqrt(tf.mean(tf.square(t.sub(mean))));
    return t.sub(mean).div(std.add(1e-6));
}
//*******************************************************************************//
let houface_featuring = null;
let houface_detection = null;
let houfer_recognizing = null;
const houfer_labels = {0:'Neg',1:'Neu',2:'Pos'};
let MODELS_READY = false;
async function loadModels() {
    if (MODELS_READY) return;
    // (optional) tie TF.js instance to face-api if needed:
    // if (faceapi.tf && faceapi.tf !== tf) faceapi.tf = tf;
    try{
        houface_featuring   = await tf.loadGraphModel('./static/models/houface512d/model.json');
        console.log('Loaded model/houface_featuring:',houface_featuring);
    } catch (e) {
        console.error('Error loading model/houface_featuring:',e);
    }
    try{
        houfer_recognizing  = await tf.loadGraphModel('./static/models/houfer/model.json');
        console.log('Loaded model/houfer_recognizing:',houfer_recognizing);
    } catch (e) {
        console.error('Error loading model/houfer_recognizing:',e);
    }
    try{
        // await faceapi.nets.ssdMobilenetv1.loadFromUri('./static/models/houdetection');
        // console.log('Loaded model/ssdMobilenetv1:',faceapi);
        function loadScript(url, callback) {
          const script = document.createElement('script');
          script.src = url;
          script.onload = callback;
          script.onerror = () => console.error(`Failed to load script: ${url}`);
          document.head.appendChild(script);
        }
        // Load the face-detection library
        loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/face-detection', async () => {
          console.log('Face detection library loaded');
          houface_detection = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
            runtime: 'tfjs',
          });
          console.log('Loaded model/tfjs/face-detection:', houface_detection);
        });        
    } catch (e) {
        console.error('Error loading model/ssdMobilenetv1:',e);
    }
    MODELS_READY = true;
}
window.modelsReady = loadModels();

async function verifyModel(path2model, name2json='model.json') {
    console.log(`verify model at ${path2model}`)
    const modelJson = await fetch(`${path2model}/${name2json}`);
    const modelData = await modelJson.json();
    
    console.log('Model topology:', modelData);
    
    // Check each weight file
    if (modelData.weightsManifest) {
        for (const manifest of modelData.weightsManifest) {
            for (const path of manifest.paths) {
                const weightUrl = `${path2model}/${path}`;
                const response = await fetch(weightUrl);
                const blob = await response.blob();
                console.log(`${path}: ${blob.size} bytes, type: ${blob.type}`);
                
                // Verify size is multiple of 4
                if (blob.size % 4 !== 0) {
                    console.error(`❌ ${path} has invalid size: ${blob.size}`);
                }
            }
        }
    }
}
verifyModel('./static/models/houface512d');
verifyModel('./static/models/houfer');
verifyModel('./static/models/houdetection','ssd_mobilenetv1_model-weights_manifest.json');

// (async () => {
//     try {
//         await Promise.all([ faceapi.nets.ssdMobilenetv1.loadFromUri('/static/models/houdetection') ]);
//         houface_featuring = await tf.loadGraphModel('/static/models/houface512d/model.json');
//         houer_recognizing = await tf.loadGraphModel('/static/models/houfer/model.json');
//     } catch (error) {
//         alert('Error loading models: ' + error.message);
//     }
// })();

async function face_detect_descriptors(img_video, withFER) {
    if (!MODELS_READY) { await loadModels(); }
    // const detection = await faceapi.detectSingleFace(img_video);
    const detections = await houface_detection.estimateFaces(img_video);
    console.log('Detected:',detections);
    if (detections.length>0) {
        const detection = detections[0];
        const faceCanvas = cropSquareToCanvas(img_video, detection.box);
        let p1 = null; 
        let p2 = null;
        const y = tf.tidy(() => {
            let x = tf.browser.fromPixels(faceCanvas).toFloat();
            let x1 = tf.image.resizeBilinear(x, [160, 160]); // adjust to expects 160
            x1 = prewhiten(x1).expandDims(0);            // [1,160,160,3]
            p1 = houface_featuring.predict(x1);
            if (withFER){
                let x2 = tf.image.resizeBilinear(x, [70, 70]); // adjust to expects 70
                x2 = x2.div(127.5).sub(1.0);
                x2 = prewhiten(x2).expandDims(0);            // [1,70,70,3]
                p2 = houfer_recognizing.predict(x2);
            }
            return {face:p1, fer:p2};
        });
        const { face, fer } = y;
        const vec = await face.data();  // Float32Array length 512: flatten
        // let val = await fer;
        let lab = null;
        if (fer){
            const arm = tf.argMax(fer, 1); // Compute argmax along axis 1
            const idx = (await arm.data())[0];
            lab = houfer_labels[idx];
        }
        // face.dispose();
        return {box:detection.box, descriptor:Array.from(l2Normalize(vec)), fer:lab};
    }else {
        return null;
    }
}
//**************************************config***********************************//
const INTERVAL_SEC = 5;           // beep every N seconds
const LOOKAHEAD_MS = 5;           // scheduler tick
const DEFAULT_VOL = 0.8;
const ALARM = { on: false };       // flip this to start/stop
let ctxAudio, masterAudio, timerAudio = null, nextBeepTime = 0;
/* init audio graph */
function initAudio() {
    if (!ctxAudio){
        ctxAudio = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    }
    masterAudio = ctxAudio.createGain();
    masterAudio.gain.value = 0.8;
    masterAudio.connect(ctxAudio.destination);
    const unlock = async () => {
        if (ctxAudio.state === 'suspended') await ctxAudio.resume();
        document.removeEventListener('pointerdown', unlock);
        document.removeEventListener('keydown', unlock);
    };
    document.addEventListener('pointerdown', unlock, { once: true });
    document.addEventListener('keydown', unlock, { once: true });    
    if (!timerAudio){
        timerAudio = setInterval(() => {
            if (!ctxAudio || !ALARM.on) return;            // just idle if off
            const now = ctxAudio.currentTime;
            while (nextBeepTime < now+1) {
                scheduleBeep(nextBeepTime);
                nextBeepTime += INTERVAL_SEC;
            }
        }, LOOKAHEAD_MS);
    }
}
/* schedule a short beep on the audio clock */
function scheduleBeep(t, dur = 0.18, freq = 880) {
    const osc = ctxAudio.createOscillator();
    const gain = ctxAudio.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.0, t);
    gain.gain.linearRampToValueAtTime(0.9, t + 0.005);
    gain.gain.linearRampToValueAtTime(0.0, t + dur);
    osc.connect(gain).connect(masterAudio);
    osc.start(t);
    osc.stop(t + dur + 0.02);
}
/* set alarm On/Off */
async function alarmSet(on) {
    if (on) {
        ALARM.on = true;
        // if (ctxAudio && ctxAudio.state === 'suspended') await ctxAudio.resume();
        const now = ctxAudio.currentTime;
        masterAudio.gain.cancelScheduledValues(now);
        masterAudio.gain.setValueAtTime(DEFAULT_VOL, now);
        nextBeepTime = Math.max(now + 0.1, nextBeepTime || 0);
    } else {
        // IMMEDIATE STOP
        ALARM.on = false;
        const now = ctxAudio.currentTime;
        masterAudio.gain.cancelScheduledValues(now);
        masterAudio.gain.setValueAtTime(0.0, now);
        nextBeepTime = 0;
    }
}
/* start for beep */
document.addEventListener('DOMContentLoaded', () => {
    initAudio();
});
//*******************************************************************************//
