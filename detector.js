// detector.js — Web Worker (additive to scaffold)
// Loads ONNX Runtime Web from CDN, initializes a tiny detector, does inference,
// returns detections as [{x,y,w,h,score,class}] in canvas pixel space.

let backend = 'wasm';
let model = 'yolov5n';
let resolution = 480;
let ready = false;
let session = null;
let inputSize = 320; // model input square size (e.g., 320x320)
let offscreen = null, octx = null;

const MODEL_URL = './models/yolov5n.onnx'; // <<< put your model here (or change path)
const CLASSES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck']; // subset; we’ll filter

function post(type, payload={}){ postMessage({ type, ...payload }); }

async function loadORT() {
  if (self.ort) return;
  importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
}

async function initDetector(opts){
  await loadORT();
  backend    = opts.backend || backend;
  model      = opts.model || model;
  resolution = opts.resolution || resolution;

  const epOrder = backend==='webgpu' ? ['webgpu','wasm'] :
                  (backend==='webgl2' || backend==='webgl') ? ['webgl','wasm'] :
                  ['wasm'];

  const so = { executionProviders: epOrder, graphOptimizationLevel: 'all' };
  session = await ort.InferenceSession.create(MODEL_URL, so);

  offscreen = new OffscreenCanvas(inputSize, inputSize);
  octx = offscreen.getContext('2d', { willReadFrequently: true });

  ready = true;
  post('ready', { backend, model });
}

function dispose(){
  session = null; ready = false;
}

function letterboxDims(sw, sh, size){
  const r = Math.min(size/sw, size/sh);
  const nw = Math.round(sw*r), nh = Math.round(sh*r);
  const dx = Math.floor((size - nw)/2), dy = Math.floor((size - nh)/2);
  return { nw, nh, dx, dy, scale:r };
}

function preprocess(bitmap){
  const sw = bitmap.width, sh = bitmap.height;
  const { nw, nh, dx, dy, scale } = letterboxDims(sw, sh, inputSize);
  octx.clearRect(0,0,inputSize,inputSize);
  octx.drawImage(bitmap, 0,0, sw, sh, dx, dy, nw, nh);
  const imgd = octx.getImageData(0,0,inputSize,inputSize).data;
  // YOLO expects float32 CHW in [0,1]
  const chw = new Float32Array(3*inputSize*inputSize);
  let p = 0;
  for (let i=0;i<inputSize*inputSize;i++){
    const r = imgd[4*i], g = imgd[4*i+1], b = imgd[4*i+2];
    chw[p] = r/255; chw[p + inputSize*inputSize] = g/255; chw[p + 2*inputSize*inputSize] = b/255;
    p++;
  }
  return { tensor: new ort.Tensor('float32', chw, [1,3,inputSize,inputSize]), scale, dx, dy, sw, sh };
}

function sigmoid(x){ return 1/(1+Math.exp(-x)); }

// Basic NMS
function nms(boxes, scores, iouThresh=0.45, topK=100){
  const idxs = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const keep=[];
  function iou(a,b){
    const ax1=a.x, ay1=a.y, ax2=a.x+a.w, ay2=a.y+a.h;
    const bx1=b.x, by1=b.y, bx2=b.x+b.w, by2=b.y+b.h;
    const ix=Math.max(0, Math.min(ax2,bx2)-Math.max(ax1,bx1));
    const iy=Math.max(0, Math.min(ay2,by2)-Math.max(ay1,by1));
    const inter=ix*iy, uni=a.w*a.h + b.w*b.h - inter;
    return uni>0 ? inter/uni : 0;
  }
  for (const i of idxs){
    const bi=boxes[i]; let keepIt=true;
    for (const k of keep){ if (iou(bi, boxes[k])>iouThresh){ keepIt=false; break; } }
    if (keepIt){ keep.push(i); if (keep.length>=topK) break; }
  }
  return keep;
}

// Postprocess for YOLOv5-like output: [1, N, (x,y,w,h,conf,*classes)]
function postprocess(output, meta, scoreThresh=0.25){
  // Find the first output tensor
  const key = Object.keys(output)[0];
  const out = output[key]; // Tensor
  const data = out.data;   // Float32Array
  const n = out.dims[1];
  const s = out.dims[2];
  const stride = s;

  const boxes=[], scores=[], clsIdx=[];
  for (let i=0;i<n;i++){
    const x = data[i*stride + 0];
    const y = data[i*stride + 1];
    const w = data[i*stride + 2];
    const h = data[i*stride + 3];
    const obj = data[i*stride + 4];
    let best=-1, bestSc=0;
    for (let c=5;c<stride;c++){
      const sc = data[i*stride + c];
      if (sc>bestSc){ bestSc=sc; best=c-5; }
    }
    const conf = obj*bestSc;
    if (conf < scoreThresh) continue;
    // xywh on inputSize → undo letterbox to source pixels
    const cx = (x - meta.dx) / meta.scale;
    const cy = (y - meta.dy) / meta.scale;
    const ww = w / meta.scale;
    const hh = h / meta.scale;
    boxes.push({ x: cx - ww/2, y: cy - hh/2, w: ww, h: hh });
    scores.push(conf);
    clsIdx.push(best);
  }
  const keep = nms(boxes, scores, 0.45, 100);
  const dets = [];
  for (const k of keep){
    const cls = CLASSES[clsIdx[k]] || 'obj';
    if (cls!=='car' && cls!=='truck' && cls!=='bus') continue;
    dets.push({ ...boxes[k], score: scores[k], class: cls });
  }
  return dets;
}

onmessage = async (ev)=>{
  const msg = ev.data || {};
  try{
    if (msg.type === 'init'){
      await initDetector(msg);
    } else if (msg.type === 'dispose'){
      dispose();
    } else if (msg.type === 'frame'){
      if (!ready) return;
      const t0 = performance.now();
      const meta = preprocess(msg.data);
      const feeds = { images: meta.tensor }; // input name often "images" for yolov5 ports; adjust if needed
      const output = await session.run(feeds);
      const dets = postprocess(output, meta, 0.25);
      const t1 = performance.now();
      post('result', { detections: dets, inferMs: t1 - t0 });
      msg.data.close?.();
    }
  } catch (e){
    post('error', { message: e.message || String(e) });
  }
};
