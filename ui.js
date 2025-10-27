const $ = (id)=>document.getElementById(id);

const bannerEl = $('banner');
const debugEl  = $('debug');
const chipEl   = $('adaptiveChip');
const runtimeStatusEl = $('runtimeStatus');
const calQualityEl = $('calQuality');
const calProgressEl = $('calProgress');

let bannerTimer = null;
let logBuffer = [];
let logScheduled = false;

export const banner = {
  show(msg, kind='ok'){
    bannerEl.textContent = msg;
    bannerEl.className = `banner show ${kind}`;
    if (bannerTimer) clearTimeout(bannerTimer);
    bannerTimer = setTimeout(()=>{ bannerEl.classList.remove('show'); }, 2200);
  }
};

export function debugLog(msg){
  const t = new Date().toLocaleTimeString();
  logBuffer.push(`[${t}] ${msg}`);
  if (!logScheduled) {
    logScheduled = true;
    requestAnimationFrame(()=>{
      debugEl.textContent = (logBuffer.join('\n') + '\n' + debugEl.textContent).slice(0, 12000);
      logBuffer = [];
      logScheduled = false;
    });
  }
}

export const chip = {
  setText(t){ chipEl.textContent = t; },
  flash(){ chipEl.classList.remove('flash'); void chipEl.offsetWidth; chipEl.classList.add('flash'); }
};

export function setRuntimeStatus(text){ runtimeStatusEl.textContent = text; }
export function setCalQualityLabel(label){ calQualityEl.textContent = label; }
export function setCalProgress(n){ calProgressEl.textContent = `${n}/4`; }
export function presetAppliedFlash(){
  const input = document.getElementById('realLen');
  input.style.outline = '2px solid #4bd3ff';
  setTimeout(()=>{ input.style.outline = 'none'; }, 700);
}
