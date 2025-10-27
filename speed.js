// speed.js — computes speed along calibrated road axis + uncertainty band
// API:
//   const sp = new SpeedEstimator();
//   const out = sp.update(activeTrack, calState, tsMs);
//   out => { speedMps, speedMph, uncertaintyMph }

export class SpeedEstimator{
  constructor(){ this.last = null; this.ema = null; }
  update(track, cal, ts){
    if (!track || !cal) return { speedMps:0, speedMph:0, uncertaintyMph:0 };

    const cx = track.bbox.x + track.bbox.w/2;
    const cy = track.bbox.y + track.bbox.h/2;
    // project onto road axis
    const ax = cal.axis_unit?.x ?? 1, ay = cal.axis_unit?.y ?? 0;
    const proj = cx*ax + cy*ay;

    let vpx = 0;
    if (this.last){ const dt = (ts - this.last.ts)/1000; if (dt>0) vpx = (proj - this.last.proj)/dt; }
    this.last = { proj, ts };

    // scale px/s → m/s
    const scale = cal.scale_m_per_px || 1; // in homography mode you may derive from bar; start with line mode
    const v = vpx * scale; // m/s

    // EMA smoothing in m/s
    const alpha = 0.25;
    this.ema = (this.ema==null) ? v : (alpha*v + (1-alpha)*this.ema);
    const mps = this.ema;
    const mph = mps * 2.236936;

    // uncertainty
    const cal_q = cal.quality || 0.5;
    const avg_conf = Math.min(1, Math.max(0, track.conf || 0.6));
    const reid_factor = Math.min(1, (track.reidCount||0)/5);
    const base = 0.2 * mph;
    let u = base * (0.5*(1-cal_q) + 0.3*(1-avg_conf) + 0.2*reid_factor);
    // floor at 2 mph for very low speeds (per Grok alt)
    u = Math.max(u, 2);

    return { speedMps: mps, speedMph: mph, uncertaintyMph: u };
  }
}
