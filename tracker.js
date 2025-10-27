// tracker.js — SORT-ish tracker with short re-ID (HSV histogram)
// Public API:
//   const trk = new Tracker({ iouThresh:0.3, maxAge:5, minHits:3 });
//   trk.update(detections, ts, ctxForHist); // ctx used to sample HSV hist from frame
//   const { tracks } = trk; // each track: {id, bbox:{x,y,w,h}, age, hits, lastTs, conf, reidCount}

function iou(a,b){
  const ax1=a.x, ay1=a.y, ax2=a.x+a.w, ay2=a.y+a.h;
  const bx1=b.x, by1=b.y, bx2=b.x+b.w, by2=b.y+b.h;
  const ix=Math.max(0, Math.min(ax2,bx2)-Math.max(ax1,bx1));
  const iy=Math.max(0, Math.min(ay2,by2)-Math.max(ay1,by1));
  const inter=ix*iy, uni=a.w*a.h + b.w*b.h - inter;
  return uni>0 ? inter/uni : 0;
}
function hsvHist(ctx, box){
  const bins = new Array(8*8*8).fill(0);
  const {x,y,w,h} = box;
  const sx = Math.max(0, Math.floor(x)), sy = Math.max(0, Math.floor(y));
  const ex = Math.min(ctx.canvas.width-1, Math.floor(x+w));
  const ey = Math.min(ctx.canvas.height-1, Math.floor(y+h));
  if (ex<=sx || ey<=sy) return bins;
  const img = ctx.getImageData(sx,sy,ex-sx,ey-sy).data;
  for (let i=0; i<img.length; i+=4){
    const r=img[i]/255, g=img[i+1]/255, b=img[i+2]/255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b); const d=max-min;
    let h=0, s=max===0?0:d/max, v=max;
    if (d!==0){
      if (max===r) h=( (g-b)/d + (g<b?6:0) );
      else if (max===g) h=( (b-r)/d + 2 );
      else h=( (r-g)/d + 4 );
      h/=6;
    }
    const hb=Math.min(7, Math.floor(h*8));
    const sb=Math.min(7, Math.floor(s*8));
    const vb=Math.min(7, Math.floor(v*8));
    bins[hb*64 + sb*8 + vb] += 1;
  }
  const sum = bins.reduce((a,b)=>a+b,0)||1;
  for (let i=0;i<bins.length;i++) bins[i]/=sum;
  return bins;
}
function cosSim(a,b){ let s=0, na=0, nb=0; for (let i=0;i<a.length;i++){ s+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } return s/(Math.sqrt(na*nb)||1e-6); }

let NEXT_ID = 1;

export class Tracker{
  constructor(opts={}){
    this.iouThresh = opts.iouThresh ?? 0.3;
    this.maxAge = opts.maxAge ?? 5;
    this.minHits = opts.minHits ?? 3;
    this.tracks = [];
  }
  // Greedy assignment (fast); adequate for 1–3 vehicles
  _assign(dets){
    const matches=[], umD=[], umT=[];
    const usedT = new Set(), usedD = new Set();
    for (let ti=0; ti<this.tracks.length; ti++){
      let bestD=-1, bestI=0;
      for (let di=0; di<dets.length; di++){
        if (usedD.has(di)) continue;
        const i = iou(this.tracks[ti].bbox, dets[di]);
        if (i>bestI){ bestI=i; bestD=di; }
      }
      if (bestD>=0 && bestI>=this.iouThresh){ matches.push([ti,bestD]); usedT.add(ti); usedD.add(bestD); }
    }
    for (let di=0; di<dets.length; di++) if (!usedD.has(di)) umD.push(di);
    for (let ti=0; ti<this.tracks.length; ti++) if (!usedT.has(ti)) umT.push(ti);
    return { matches, umD, umT };
  }
  _reid(umD, umT, dets, ctx){
    const extraMatches=[];
    for (const ti of umT){
      const tr = this.tracks[ti];
      if (!tr.hist || (tr.lastTs && (performance.now()-tr.lastTs) > 1200)) continue;
      let best=-1, bestCost=1e9;
      for (const di of umD){
        const hist = hsvHist(ctx, dets[di]);
        const c_iou = 1 - iou(tr.bbox, dets[di]);
        const c_hist = 1 - cosSim(tr.hist, hist);
        const cost = 0.6*c_iou + 0.4*c_hist;
        if (cost < bestCost){ bestCost=cost; best=di; }
      }
      if (best>=0 && bestCost < 0.7){ // conservative accept
        extraMatches.push([ti,best]);
        // remove from arrays
        umD.splice(umD.indexOf(best),1);
        umT.splice(umT.indexOf(ti),1);
      }
    }
    return extraMatches;
  }
  update(detections, ts, ctx){
    // Age existing tracks
    for (const t of this.tracks){ t.age++; }

    // IoU assign
    const { matches, umD, umT } = this._assign(detections);
    // Short re-ID
    const extra = this._reid(umD, umT, detections, ctx);
    for (const m of extra) matches.push(m);

    // Update matched
    for (const [ti, di] of matches){
      const tr = this.tracks[ti];
      tr.bbox = detections[di];
      tr.hits++; tr.age = 0; tr.lastTs = ts; tr.conf = detections[di].score;
      tr.hist = hsvHist(ctx, tr.bbox);
    }
    // Create new tracks for unmatched detections
    for (const di of umD){
      const det = detections[di];
      const hist = hsvHist(ctx, det);
      this.tracks.push({ id: NEXT_ID++, bbox: det, age:0, hits:1, lastTs:ts, conf:det.score, hist, reidCount:0 });
    }
    // Remove stale
    this.tracks = this.tracks.filter(t => t.age <= this.maxAge);

    // Return active (min_hits)
    return this.tracks.filter(t => t.hits >= this.minHits);
  }
}
