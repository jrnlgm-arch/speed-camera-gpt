// Add at top:
import { Tracker } from './tracker.js';
import { SpeedEstimator } from './speed.js';

// Add near state declaration:
state.tracker = new Tracker({ iouThresh: 0.3, maxAge: 5, minHits: 3 });
state.speedEst = new SpeedEstimator();
state.activeId = null;
state.lastDetections = [];

// In initDetectorWorker(), extend onmessage handler (keep existing lines) with:
state.detectorWorker.onmessage = (ev)=>{
  const msg = ev.data;
  if (msg?.type === 'ready') { /* existing */ }
  else if (msg?.type === 'result') {
    state.inferAvgMs = msg.inferMs ?? state.inferAvgMs;
    state.lastDetections = msg.detections || [];
  } else if (msg?.type === 'error') { /* existing */ }
};

// In loop(ts), after ctx.drawImage(…), insert:
const active = state.tracker.update(state.lastDetections, ts, state.ctx);
// pick active target
let target = active[0];
if (!state.activeId && target) state.activeId = target.id;
if (state.activeId) {
  const found = active.find(t => t.id === state.activeId);
  if (found) target = found; else { state.activeId = target?.id || null; }
}
// speed compute
let speedOut = null;
if (target) {
  const cal = getCalibrationState();
  speedOut = state.speedEst.update(target, cal, ts);
  // draw overlay
  state.ctx.strokeStyle = '#4bd3ff'; state.ctx.lineWidth = 2;
  state.ctx.strokeRect(target.bbox.x, target.bbox.y, target.bbox.w, target.bbox.h);
  state.ctx.fillStyle = '#0e1327aa';
  state.ctx.fillRect(target.bbox.x, target.bbox.y-18, 140, 18);
  state.ctx.fillStyle = '#e7e9ee';
  const mph = (speedOut.speedMph||0).toFixed(1);
  const u = (speedOut.uncertaintyMph||0).toFixed(1);
  state.ctx.fillText(`${mph} mph ±${u}`, target.bbox.x+6, target.bbox.y-5);
}
// draw all detections (thin boxes)
for (const d of state.lastDetections){
  state.ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  state.ctx.strokeRect(d.x, d.y, d.w, d.h);
}
import { ui, banner, chip, debugLog, setRuntimeStatus, presetAppliedFlash } from './ui.js';
import { startLineCalibration, startHomographyCalibration, getCalibrationState, setPresetLength, resetCalibration, attachCalCleanup } from './calibration.js';
import { WorkerMsg } from './ui.js';

// thresholds (Grok-updated)
const FPS_DOWNSHIFT = 12;
const FORCE_480_FPS = 10;
const INFER_MS_HIGH = 120;
const ADAPT_WINDOW_MS = 2000;
const DETECTOR_TIMEOUT_MS = 5000;
const DETECTOR_MAX_RETRIES = 2;

const BACKENDS = [];
let state = {
  running:false, backend:'auto', resolution:480,
  cadenceK:2, videoEl:null, canvasEl:null, ctx:null,
  fps:0, frames:0, startTs:0, lastAdaptCheck:0,
  videoReady:false, videoSource:'file',
  detectorWorker:null, workerReady:false, workerRetries:0,
  inferAvgMs:0, inferTick:0
};

function detectCapabilities(){
  const hasWebGPU = 'gpu' in navigator;
  const testCan = document.createElement('canvas');
  const hasWebGL2 = !!testCan.getContext('webgl2');
  const hasWebGL = hasWebGL2 || !!testCan.getContext('webgl');

  BACKENDS.length=0;
  if (hasWebGPU) BACKENDS.push('webgpu');
  if (hasWebGL2) BACKENDS.push('webgl2');
  else if (hasWebGL) BACKENDS.push('webgl');
  BACKENDS.push('wasm');

  const sel = document.getElementById('backendSelect');
  sel.innerHTML = `<option value="auto" selected>auto (${BACKENDS[0]||'none'})</option>` + BACKENDS.map(b=>`<option value="${b}">${b}</option>`).join('');
  document.getElementById('backendStatus').textContent = `detected: ${BACKENDS.join(' → ')}`;
}
detectCapabilities();

function sizeCanvasToVideo(){
  const v = state.videoEl, c = state.canvasEl;
  const vw = v.videoWidth|0, vh = v.videoHeight|0;
  if (!vw || !vh) return;
  const targetH = state.resolution;
  const targetW = Math.round((vw/vh) * targetH);
  c.width = targetW; c.height = targetH;
  // Use CSS for video sizing (no layout thrash via width/height attrs)
  v.style.width = `${targetW}px`; v.style.height = `${targetH}px`;
  const s = `${state.backend} • ${state.resolution}p • k:${state.cadenceK} • cal:${getCalibrationState()?.qualityLabel||'N/A'}`;
  chip.setText(s);
}

async function useWebcam(){
  const v = state.videoEl;
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:'environment' }, audio:false });
    v.srcObject = stream;
    v.onloadeddata = ()=>{
      state.videoReady = true; sizeCanvasToVideo(); banner.show('Webcam ready','ok'); debugLog('Webcam started.');
    };
    await v.play();
    state.videoSource='webcam';
  }catch(e){
    banner.show('Webcam failed. Check permissions.','bad'); debugLog('Webcam error: '+e.message);
  }
}

function useFile(file){
  const v = state.videoEl;
  const url = URL.createObjectURL(file);
  v.srcObject = null; v.src = url;
  v.onloadeddata = ()=>{
    state.videoReady = true; sizeCanvasToVideo(); banner.show('Video loaded','ok'); debugLog(`Loaded file: ${file.name}`);
  };
  v.play();
  state.videoSource='file';
}

function updateChip(){ chip.flash(); }

async function initDetectorWorker(){
  if (state.detectorWorker){ state.detectorWorker.terminate(); }
  state.detectorWorker = new Worker('./detector.js', { type:'module' });
  state.workerReady = false;
  state.workerRetries = 0;

  const doInit = ()=> new Promise((resolve,reject)=>{
    let settled=false;
    const timer = setTimeout(()=>{ if(!settled){ settled=true; reject(new Error('Detector init timeout')); } }, DETECTOR_TIMEOUT_MS);

    state.detectorWorker.onmessage = (ev)=>{
      const msg = ev.data;
      if (msg?.type === 'ready') { clearTimeout(timer); settled=true; state.workerReady=true; resolve(); }
      else if (msg?.type === 'result') {
        state.inferAvgMs = msg.inferMs ?? state.inferAvgMs;
        // TODO: enqueue detections to tracker
      } else if (msg?.type === 'error') {
        clearTimeout(timer); settled=true; reject(new Error(msg.message||'Detector error'));
      }
    };

    try{
      const backend = state.backend==='auto' ? (BACKENDS[0]||'wasm') : state.backend;
      state.backend = backend;
      state.detectorWorker.postMessage(WorkerMsg.init(backend, 'yolov5n', state.resolution));
    }catch(e){ clearTimeout(timer); settled=true; reject(e); }
  });

  for (let attempt=0; attempt<=DETECTOR_MAX_RETRIES; attempt++){
    try{
      await doInit(); banner.show(`Detector ready (${state.backend})`,'ok'); debugLog('Detector worker ready.'); return;
    }catch(e){
      debugLog(`Detector init failed (attempt ${attempt+1}): ${e.message}`);
      if (attempt < DETECTOR_MAX_RETRIES){
        await switchBackend(nextFallback(state.backend));
      } else {
        banner.show('Detector unavailable; running without detections.','warn');
      }
    }
  }
}

function nextFallback(b){
  if (b==='webgpu') return BACKENDS.includes('webgl2')?'webgl2':(BACKENDS.includes('webgl')?'webgl':'wasm');
  if (b==='webgl2') return BACKENDS.includes('webgl')?'webgl':'wasm';
  if (b==='webgl') return 'wasm';
  return 'wasm';
}

async function switchBackend(b){
  state.backend = b;
  updateChip();
  debugLog(`Switching backend → ${b}`);
  if (state.detectorWorker){
    state.detectorWorker.postMessage({type:'dispose'});
    await initDetectorWorker();
    banner.show(`Backend switched to ${b}`,'warn');
  }
}

function loop(ts){
  if (!state.running) return;
  const v = state.videoEl, c = state.canvasEl, ctx = state.ctx;

  if (state.videoReady && v.videoWidth && v.videoHeight){
    ctx.drawImage(v, 0, 0, c.width, c.height);
  }

  // calibration overlays
  import('./calibration.js').then(m => m.drawCalibration(ctx));

  // FPS
  state.frames++; if (!state.startTs) state.startTs = ts;
  const elapsed = ts - state.startTs;
  if (elapsed >= 1000) {
    state.fps = Math.round((state.frames * 1000) / elapsed);
    state.frames = 0; state.startTs = ts;
    setRuntimeStatus(`FPS:${state.fps} • backend:${state.backend} • res:${state.resolution}p`);
  }

  // Adaptive every ~2s using display FPS (detector avg comes later)
  if (!state.lastAdaptCheck) state.lastAdaptCheck = ts;
  if (ts - state.lastAdaptCheck > ADAPT_WINDOW_MS){
    const avgFps = state.fps;
    if (avgFps && avgFps < FPS_DOWNSHIFT){
      if (avgFps < FORCE_480_FPS && state.resolution > 480){
        state.resolution = 480; sizeCanvasToVideo(); banner.show('Low perf → 480p','warn'); updateChip(); debugLog('Adapt: force 480p');
      } else if (state.resolution > 640){
        state.resolution = 640; sizeCanvasToVideo(); banner.show('Adaptive: 720→640','warn'); updateChip(); debugLog('Adapt: downscale 720→640');
      } else if (state.cadenceK < 3){
        state.cadenceK = 3; banner.show('Adaptive: cadence k 2→3','warn'); updateChip(); debugLog('Adapt: k=3');
      }
    }
    state.lastAdaptCheck = ts;
  }

  requestAnimationFrame(loop);
}

// ---- Event wiring
function bindUI(){
  state.videoEl = document.getElementById('video');
  state.canvasEl = document.getElementById('canvas');
  state.ctx = state.canvasEl.getContext('2d', { alpha:false });

  document.getElementById('btnWebcam').onclick = useWebcam;
  document.getElementById('fileInput').onchange = (e)=>{ const f=e.target.files?.[0]; if (f) useFile(f); };
  document.getElementById('backendSelect').onchange = (e)=>{ const val=e.target.value; state.backend = val==='auto'?(BACKENDS[0]||'wasm'):val; updateChip(); };
  document.getElementById('resSelect').onchange = (e)=>{ state.resolution = parseInt(e.target.value,10)||480; sizeCanvasToVideo(); banner.show(`Resolution → ${state.resolution}p`,'warn'); updateChip(); };
  document.getElementById('btnStart').onclick = async()=>{
    if (!state.running){
      state.running = true; requestAnimationFrame(loop);
      sizeCanvasToVideo(); banner.show('Running','ok'); debugLog('Loop started.');
      await initDetectorWorker(); // safe if it fails (graceful)
    }
  };
  document.getElementById('btnStop').onclick = ()=>{
    state.running = false; banner.show('Stopped','warn'); debugLog('Loop stopped.');
    // ensure listeners from calibration are cleaned up:
    attachCalCleanup(true);
  };
  // Calibration UI
  document.getElementById('preset').onchange = (e)=>{ const val=e.target.value; const units=document.getElementById('units').value; if (val){ setPresetLength(parseFloat(val), units); presetAppliedFlash(); banner.show('Preset applied','ok'); } };
  document.getElementById('btnCalibrate').onclick = ()=>{
    const mode = document.getElementById('calMode').value;
    const len = parseFloat(document.getElementById('realLen').value);
    const units = document.getElementById('units').value;
    if (mode==='line') startLineCalibration(state.canvasEl, len, units);
    else startHomographyCalibration(state.canvasEl, len, units);
  };
  document.getElementById('btnCalReset').onclick = ()=>{ resetCalibration(); banner.show('Calibration reset','warn'); updateChip(); };
}
bindUI();
setRuntimeStatus('idle'); debugLog('App loaded. Ready.');
