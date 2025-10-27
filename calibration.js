import { banner, debugLog, setCalQualityLabel, setCalProgress } from './ui.js';

let cal = {
  mode:'none',
  line:null,
  homoPts:[],
  H:null,
  scale_m_per_px:null,
  axis_unit:{x:1,y:0},
  quality:0,
  qualityLabel:'N/A',
  units:'ft'
};

const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }
function dot(a,b){ return a.x*b.x + a.y*b.y; }
function norm(v){ const d=Math.hypot(v.x,v.y)||1e-6; return {x:v.x/d,y:v.y/d}; }
function inCanvas(canvas, e){ const r=canvas.getBoundingClientRect(); return e.clientX>=r.left && e.clientX<=r.right && e.clientY>=r.top && e.clientY<=r.bottom; }

function segIntersect(p1,p2,p3,p4){
  function orient(a,b,c){ return Math.sign((b.y-a.y)*(c.x-b.x) - (b.x-a.x)*(c.y-b.y)); }
  const o1=orient(p1,p2,p3), o2=orient(p1,p2,p4), o3=orient(p3,p4,p1), o4=orient(p3,p4,p2);
  if (o1 !== o2 && o3 !== o4) return true;
  return false;
}

// Minimal 4-point DLT homography solver (solves Ah=0 via 8x8; returns 3x3)
function computeHomographyDLT(src, dst){
  // src,dst: 4 points each [{x,y}]
  // Build A*h = b where we set h33=1 to make system 8x8
  // Ref: standard DLT with scale fix
  const A = []; const b = [];
  for (let i=0;i<4;i++){
    const X=src[i].x, Y=src[i].y, x=dst[i].x, y=dst[i].y;
    A.push([ X, Y, 1, 0, 0, 0, -x*X, -x*Y ]); b.push(x);
    A.push([ 0, 0, 0, X, Y, 1, -y*X, -y*Y ]); b.push(y);
  }
  // Solve 8x8 via Gaussian elimination
  function solve(A,b){
    const n = 8;
    for (let i=0;i<n;i++){
      // pivot
      let max=i; for (let r=i+1;r<n;r++) if (Math.abs(A[r][i])>Math.abs(A[max][i])) max=r;
      [A[i],A[max]]=[A[max],A[i]]; [b[i],b[max]]=[b[max],b[i]];
      const piv = A[i][i]||1e-12;
      for (let j=i;j<n;j++) A[i][j]/=piv; b[i]/=piv;
      for (let r=0;r<n;r++){
        if (r===i) continue;
        const f=A[r][i];
        for (let j=i;j<n;j++) A[r][j]-=f*A[i][j];
        b[r]-=f*b[i];
      }
    }
    return b;
  }
  const h = solve(A,b); // [h11,h12,h13,h21,h22,h23,h31,h32], h33=1
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1   ]
  ];
}

function qualityFrom(realLen, angleDeg, visualOK){
  const length_plaus = (() => {
    let score=1;
    if (cal.units==='ft'){
      if (realLen<10) score = clamp((realLen-5)/5,0,1);
      else if (realLen>120) score = clamp((200-realLen)/80,0,1);
    } else {
      if (realLen<3) score = clamp((realLen-1)/2,0,1);
      else if (realLen>40) score = clamp((60-realLen)/20,0,1);
    }
    return score;
  })();
  const shape_cons = clamp(1 - (angleDeg/40), 0, 1);
  const visual_QA = visualOK ? 1 : 0;
  return 0.4*length_plaus + 0.3*shape_cons + 0.3*visual_QA;
}

export function setPresetLength(val, units='ft'){
  cal.units = units;
  document.getElementById('realLen').value = String(val);
}

export function getCalibrationState(){ return cal; }

export function resetCalibration(){
  cal.mode='none'; cal.line=null; cal.homoPts=[]; cal.H=null; cal.scale_m_per_px=null; cal.quality=0; cal.qualityLabel='N/A';
  setCalQualityLabel('N/A'); setCalProgress(0);
}

let detachFns = [];
export function attachCalCleanup(clear=false){
  if (clear){ detachFns.forEach(fn=>fn&&fn()); detachFns=[]; }
}

// ---- LINE MODE ----
export function startLineCalibration(canvas, realLen, units='ft'){
  attachCalCleanup(true);
  cal.mode='line'; cal.units=units; cal.line=null; cal.scale_m_per_px=null; cal.quality=0; cal.qualityLabel='N/A';
  banner.show('Draw a line along the road segment, then release.','ok');

  const ctx = canvas.getContext('2d');
  let tmp = { drawing:false, x1:0,y1:0,x2:0,y2:0 };

  function onDown(e){ if(!inCanvas(canvas,e)) return; const r=canvas.getBoundingClientRect(); tmp.x1=e.clientX-r.left; tmp.y1=e.clientY-r.top; tmp.drawing=true; }
  function onMove(e){
    if (!tmp.drawing||!inCanvas(canvas,e)) return;
    const r=canvas.getBoundingClientRect(); tmp.x2=e.clientX-r.left; tmp.y2=e.clientY-r.top;
  }
  function onUp(e){
    if (!tmp.drawing) return; tmp.drawing=false;
    const pxLen = Math.hypot(tmp.x2-tmp.x1,tmp.y2-tmp.y1);
    if (pxLen < 20) { banner.show('Line too short (<20 px). Try again.','bad'); return; }
    const rl = parseFloat(realLen||document.getElementById('realLen').value);
    if (!rl || rl<=0) { banner.show('Enter a real length or use a preset.','warn'); return; }

    cal.line = { x1:tmp.x1,y1:tmp.y1,x2:tmp.x2,y2:tmp.y2, pxLen };
    const meters = (cal.units==='ft') ? rl*0.3048 : rl;
    cal.scale_m_per_px = meters / pxLen;
    cal.axis_unit = norm({x: tmp.x2-tmp.x1, y: tmp.y2-tmp.y1});
    cal.quality = qualityFrom(rl, 0, true);
    cal.qualityLabel = cal.quality>=0.75?'Good': cal.quality>=0.5?'Fair':'Poor';
    setCalQualityLabel(cal.qualityLabel);
    banner.show(`Line calibration set (${rl} ${cal.units}).`,'ok');
    debugLog(`Line pxLen=${pxLen.toFixed(1)} scale=${cal.scale_m_per_px.toFixed(5)} m/px`);
  }

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  detachFns.push(()=>{ canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerup', onUp); });
}

// ---- HOMOGRAPHY MODE ----
export function startHomographyCalibration(canvas, laneWidthValue, units='ft'){
  attachCalCleanup(true);
  cal.mode='homo'; cal.units=units; cal.homoPts=[]; cal.H=null; cal.scale_m_per_px=null; cal.quality=0; cal.qualityLabel='N/A';
  setCalProgress(0);
  banner.show('Click 4 lane corners: near-left → near-right → far-right → far-left.','ok');

  function onClick(e){
    if (!inCanvas(canvas,e)) return;
    const r=canvas.getBoundingClientRect();
    const p={ x:e.clientX-r.left, y:e.clientY-r.top };
    cal.homoPts.push(p); setCalProgress(cal.homoPts.length);

    if (cal.homoPts.length===4){
      const [p1,p2,p3,p4]=cal.homoPts;
      if (segIntersect(p1,p2,p3,p4)||segIntersect(p2,p3,p4,p1)){
        banner.show('Trapezoid self-intersects. Reset and try again.','bad'); return;
      }
      const near = dist(p1,p2), far=dist(p3,p4);
      const vNear=norm({x:p2.x-p1.x,y:p2.y-p1.y});
      const vFar =norm({x:p3.x-p4.x,y:p3.y-p4.y});
      const ang = Math.acos(clamp(dot(vNear,vFar),-1,1))*180/Math.PI;

      // Compute H: map to unit square
      const src=[p1,p2,p3,p4];
      const dst=[{x:0,y:1},{x:1,y:1},{x:1,y:0},{x:0,y:0}];
      cal.H = computeHomographyDLT(src,dst);
      cal.axis_unit = vNear;

      // Quality score
      const rl = parseFloat(laneWidthValue || document.getElementById('realLen').value || '12');
      cal.quality = qualityFrom(rl, ang, true);
      cal.qualityLabel = cal.quality>=0.75?'Good': cal.quality>=0.5?'Fair':'Poor';
      setCalQualityLabel(cal.qualityLabel);

      banner.show('Homography set.','ok');
      debugLog(`Homo near=${near.toFixed(1)} far=${far.toFixed(1)} angle≈${ang.toFixed(1)}°`);
    }
  }
  canvas.addEventListener('pointerdown', onClick);
  detachFns.push(()=>{ canvas.removeEventListener('pointerdown', onClick); });
}

// called each frame by main loop
export function drawCalibration(ctx){
  const c = ctx.canvas;
  if (cal.mode==='line' && cal.line){
    ctx.save();
    ctx.strokeStyle='#4bd3ff'; ctx.lineWidth=2; ctx.setLineDash([6,4]);
    ctx.beginPath(); ctx.moveTo(cal.line.x1,cal.line.y1); ctx.lineTo(cal.line.x2,cal.line.y2); ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
  }
  if (cal.mode==='homo' && cal.homoPts.length){
    const pts=cal.homoPts;
    const drawPt=(p,i)=>{ ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fillStyle='#ffd54b'; ctx.fill(); ctx.fillStyle='#cbb26b'; ctx.fillText(String(i+1), p.x+6, p.y-6); };
    ctx.save(); ctx.strokeStyle='#ffd54b'; ctx.lineWidth=2;
    for (let i=0;i<pts.length;i++) drawPt(pts[i],i);
    if (pts.length>=2){ ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y); for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x,pts[i].y); ctx.stroke(); }
    ctx.restore();
  }
}

export { cal as _cal_internal };
