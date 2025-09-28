#!/usr/bin/env python3

"""
Realtime focus HUD for a static camera on a calibration target.

Displays:
- Tenengrad, Variance of Laplacian, Modified Laplacian (exposure-normalized)
- High-frequency energy ratio
- Slanted-edge MTF50 if an edge is detected
- Rolling 0-100 scales, EMA smoothing, and peak-hold

Keys:
  q      Quit
  space  Pause/resume scoring
  r      Reset peaks and scales
  c      Recenter ROI
  [ ]    Resize ROI
  w/a/s/d Move ROI
  m      Toggle MTF50 computation
  h      Toggle HF ratio
  ?      Toggle help
"""

import cv2
import numpy as np
import time, math, argparse

# ---------- Focus measures ----------
def focus_measures(gray):
    """Tenengrad, Var(Laplacian), Modified Laplacian on gray ROI. Exposure-normalized."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(gx*gx + gy*gy))
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    vol = float(lap.var())
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    ml = float(np.mean(np.abs(dxx) + np.abs(dyy)))
    norm = float(np.mean(g*g) + 1e-6)
    return {
        "tenengrad": ten / norm,
        "var_laplacian": vol / norm,
        "modified_laplacian": ml / norm
    }

def hf_energy_ratio(gray, cutoff=0.25):
    """High-frequency energy ratio in 2D FFT above radial cutoff (cycles/pixel)."""
    g = gray.astype(np.float32)
    g -= float(np.mean(g))
    H, W = g.shape
    F = np.fft.fft2(g)
    P = np.abs(F)**2
    fx = np.fft.fftfreq(W, d=1.0)
    fy = np.fft.fftfreq(H, d=1.0)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    mask = R >= float(cutoff)
    high = float(P[mask].sum())
    total = float(P.sum() + 1e-12)
    return high / total

# ---------- Slanted-edge MTF50 (throttled) ----------
def detect_slanted_lines(gray):
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(g8, 50, 150, apertureSize=3, L2gradient=True)
    H, W = gray.shape
    min_len = int(0.25 * min(H, W))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_len, maxLineGap=12)
    out = []
    if lines is not None:
        for (x1,y1,x2,y2) in lines[:,0,:]:
            dx, dy = x2 - x1, y2 - y1
            length = float(math.hypot(dx, dy))
            if length < min_len: 
                continue
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            if 10.0 <= ang <= 80.0 or 100.0 <= ang <= 170.0:
                out.append((int(x1),int(y1),int(x2),int(y2),float(ang),length))
    out.sort(key=lambda t: t[-1], reverse=True)
    return out

def mtf50_from_edge(gray, line, strip_half_width=20, oversample=8):
    x1,y1,x2,y2,ang,_ = line
    H, W = gray.shape
    dx, dy = float(x2 - x1), float(y2 - y1)
    steep = abs(dy) >= abs(dx) and dy != 0.0
    os = int(oversample)
    w = int(strip_half_width)
    nbins = int(2*w*os)+1
    accum = np.zeros(nbins, np.float64)
    count = np.zeros(nbins, np.float64)
    if steep:
        y_min = max(min(y1,y2), 0); y_max = min(max(y1,y2), H-1)
        if y_max - y_min < 2*w+5:
            yc = int((y1+y2)/2); y_min = max(yc-(w+5), 0); y_max = min(yc+(w+5), H-1)
        for y in range(y_min, y_max+1):
            t = (y - y1) / dy
            x_edge = x1 + t*dx
            x_start = max(int(math.floor(x_edge - w)), 0)
            x_end   = min(int(math.ceil (x_edge + w)), W-1)
            xs = np.arange(x_start, x_end+1)
            d = xs - x_edge
            bins = np.floor((d + w) * os).astype(int)
            vals = gray[y, x_start:x_end+1].astype(np.float64)
            for b, v in zip(bins, vals):
                if 0 <= b < nbins:
                    accum[b] += v; count[b] += 1.0
    else:
        x_min = max(min(x1,x2), 0); x_max = min(max(x1,x2), W-1)
        if x_max - x_min < 2*w+5:
            xc = int((x1+x2)/2); x_min = max(xc-(w+5), 0); x_max = min(xc+(w+5), W-1)
        for x in range(x_min, x_max+1):
            t = (x - x1) / dx if dx != 0.0 else 0.0
            y_edge = y1 + t*dy
            y_start = max(int(math.floor(y_edge - w)), 0)
            y_end   = min(int(math.ceil (y_edge + w)), H-1)
            ys = np.arange(y_start, y_end+1)
            d = ys - y_edge
            bins = np.floor((d + w) * os).astype(int)
            vals = gray[y_start:y_end+1, x].astype(np.float64)
            for b, v in zip(bins, vals):
                if 0 <= b < nbins:
                    accum[b] += v; count[b] += 1.0
    with np.errstate(invalid='ignore'):
        esf = accum / np.maximum(count, 1e-12)
    valid = count >= 8
    if int(valid.sum()) < 20:
        return None
    esf = esf[valid]
    win_len = max(5, int(7*os) | 1)
    win = np.hamming(win_len); win /= win.sum()
    esf_s = np.convolve(esf, win, mode='same')
    lsf = np.gradient(esf_s)
    wwin = np.hamming(len(lsf))
    lsfw = lsf * wwin
    nfft = int(2**(int(np.ceil(np.log2(len(lsfw)))) + 1))
    L = np.fft.rfft(lsfw, n=nfft)
    mtf = np.abs(L)
    if mtf[0] <= 1e-12: 
        return None
    mtf /= mtf[0]
    freqs = np.fft.rfftfreq(nfft, d=1.0/os)  # cycles/pixel
    nyq = 0.5
    idx_nyq = int(np.searchsorted(freqs, nyq, side='right'))
    m = mtf[:idx_nyq+1]; f = freqs[:idx_nyq+1]
    above = m >= 0.5
    cross = np.where(np.diff(above.astype(np.int8)) == -1)[0]
    if len(cross) == 0:
        return None
    i = int(cross[0])
    f1, f2 = f[i], f[i+1]; m1, m2 = m[i], m[i+1]
    mtf50 = float(f1 + (0.5 - m1) * (f2 - f1) / (m2 - m1)) if m2 != m1 else float(f1)
    H, W = gray.shape
    return {
        "mtf50_cyc_per_pix": mtf50,
        "lp_per_ph": mtf50 * float(H),
        "lp_per_pw": mtf50 * float(W),
        "edge_angle_deg": float((abs(math.degrees(math.atan2((y2-y1), (x2-x1)))) % 180.0))
    }

# ---------- Helpers ----------
class EMA:
    def __init__(self, alpha=0.2): 
        self.a = float(alpha); self.y = None
    def update(self, x):
        x = float(x)
        self.y = x if self.y is None else (1.0 - self.a) * self.y + self.a * x
        return self.y

class PeakHold:
    def __init__(self, half_life_s=0.0):
        self.h = 0.0
        self.hl = float(half_life_s)
        self.last_t = None
    def update(self, v, t):
        v = float(v)
        if self.last_t is None:
            self.last_t = t
        if self.hl > 0 and self.h > 0:
            dt = t - self.last_t
            self.h *= 0.5 ** max(0.0, dt / self.hl)
        self.h = max(self.h, v)
        self.last_t = t
        return self.h
    def reset(self): 
        self.h = 0.0; self.last_t = None

class Scale:
    """Tracks session max for 0-100 mapping."""
    def __init__(self):
        self.m = 1e-6
    def update(self, v):
        self.m = max(self.m, float(v))
        return self.m
    def reset(self):
        self.m = 1e-6
    def map01(self, v):
        return max(0.0, min(1.0, float(v) / self.m))

def clamp_roi(cx, cy, hw, hh, W, H):
    x0 = int(max(0, cx - hw)); y0 = int(max(0, cy - hh))
    x1 = int(min(W, cx + hw)); y1 = int(min(H, cy + hh))
    return x0, y0, x1, y1

def draw_text(img, txt, org, scale=0.6, color=(255,255,255), thickness=1):
    x,y = org
    cv2.putText(img, txt, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_bar(img, x, y, w, h, val01, label):
    val01 = max(0.0, min(1.0, float(val01)))
    cv2.rectangle(img, (x,y), (x+w, y+h), (40,40,40), 1)
    fill = int(w * val01)
    cv2.rectangle(img, (x,y), (x+fill, y+h), (60,180,75), -1)
    draw_text(img, f"{label}: {val01*100:5.1f}", (x+5, y+h-6), 0.5)

def put_kv(img, x, y, lines, step=18):
    for i,(k,v) in enumerate(lines):
        draw_text(img, f"{k}: {v}", (x, y + i*step))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--roi_frac", type=float, default=0.33, help="ROI box as fraction of min(width,height)")
    ap.add_argument("--alpha", type=float, default=0.2, help="EMA alpha")
    ap.add_argument("--hf_cutoff", type=float, default=0.25)
    ap.add_argument("--hf_every", type=int, default=6, help="Compute HF every N frames")
    ap.add_argument("--mtf_every", type=int, default=12, help="Compute MTF every N frames")
    ap.add_argument("--mtf_strip", type=int, default=20)
    ap.add_argument("--mtf_os", type=int, default=8)
    ap.add_argument("--peak_hl", type=float, default=0.0, help="Peak half-life seconds; 0 disables decay")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device)
    if args.width>0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height>0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Try to lock AF/AE if supported
    try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception: pass
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception: pass

    # Warmup
    for _ in range(8): cap.read()

    ok, frame = cap.read()
    if not ok:
        print("Camera read failed.")
        return
    H, W = frame.shape[:2]
    r0 = int(min(W,H) * args.roi_frac * 0.5)
    cx, cy = W//2, H//2
    paused = False
    show_help = True
    enable_hf = True
    enable_mtf = True

    # States
    ema = {k: EMA(args.alpha) for k in ["tenengrad","var_laplacian","modified_laplacian","hf_ratio"]}
    scale = {k: Scale() for k in ["tenengrad","var_laplacian","modified_laplacian","hf_ratio"]}
    peak  = {k: PeakHold(args.peak_hl) for k in ["tenengrad","var_laplacian","modified_laplacian","hf_ratio"]}
    mtf_info = None; mtf_age = 0.0

    last = time.time()
    fcount = 0
    fps = 0.0

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ROI
        x0,y0,x1,y1 = clamp_roi(cx, cy, r0, r0, W, H)
        roi = gray[y0:y1, x0:x1]
        bright = float(np.mean(roi))

        if not paused:
            m = focus_measures(roi)
            svals = {k: math.log1p(v) for k,v in m.items()}  # stabilize
            # HF throttled
            if enable_hf and (fcount % max(1,args.hf_every) == 0):
                hf = hf_energy_ratio(roi, cutoff=args.hf_cutoff)
                svals["hf_ratio"] = hf
            elif enable_hf and "hf_ratio" not in svals:
                # carry previous EMA estimate forward
                if ema["hf_ratio"].y is not None:
                    svals["hf_ratio"] = ema["hf_ratio"].y
            # Update EMA, scales, peaks
            mapped = {}
            for k,v in svals.items():
                sm = ema[k].update(v)
                maxv = scale[k].update(sm)
                mapped[k] = scale[k].map01(sm)
                peak[k].update(mapped[k], t0)

            # Slanted-edge MTF throttled
            if enable_mtf and (fcount % max(1,args.mtf_every) == 0):
                lines = detect_slanted_lines(gray)
                mtf_info = None
                for line in lines[:3]:
                    mtf_info = mtf50_from_edge(gray, line, strip_half_width=args.mtf_strip, oversample=args.mtf_os)
                    if mtf_info is not None:
                        break
                mtf_age = 0.0
            else:
                mtf_age += (t0 - last)
        else:
            mapped = {k: (ema[k].y/scale[k].m if (ema[k].y is not None and scale[k].m>0) else 0.0)
                      for k in ["tenengrad","var_laplacian","modified_laplacian","hf_ratio"]}

        # HUD
        vis = frame.copy()
        # ROI box
        col = (60,180,75) if np.mean(list(mapped.values())) >= 0.7 else ((0,165,255) if np.mean(list(mapped.values())) >= 0.4 else (0,0,255))
        cv2.rectangle(vis, (x0,y0), (x1,y1), col, 2)
        cv2.drawMarker(vis, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 12, 1)

        # Bars
        bx, by, bw, bh, gap = 12, 12, 180, 16, 6
        draw_bar(vis, bx, by + 0*(bh+gap), bw, bh, mapped.get("tenengrad",0), "Tenengrad")
        draw_bar(vis, bx, by + 1*(bh+gap), bw, bh, mapped.get("var_laplacian",0), "VarLap")
        draw_bar(vis, bx, by + 2*(bh+gap), bw, bh, mapped.get("modified_laplacian",0), "ModLap")
        if enable_hf:
            draw_bar(vis, bx, by + 3*(bh+gap), bw, bh, mapped.get("hf_ratio",0), "HF ratio")

        # Numeric panel
        lines = [
            ("ROI", f"{x0},{y0} {x1-x0}x{y1-y0}"),
            ("MeanY", f"{bright:5.1f}"),
            ("FPS", f"{fps:4.1f}"),
            ("State", "PAUSED" if paused else "RUN")
        ]
        if mtf_info:
            lines += [("MTF50 cyc/px", f"{mtf_info['mtf50_cyc_per_pix']:.4f}"),
                      ("MTF50 lp/ph",  f"{mtf_info['lp_per_ph']:.0f}"),
                      ("Edge deg",    f"{mtf_info['edge_angle_deg']:.1f}")]
        put_kv(vis, bx, by + 5*(bh+gap) + 8, lines, step=18)

        # Peaks panel
        px = bx + bw + 30; py = by
        draw_text(vis, "Peaks (0-100)", (px, py))
        pvals = [
            ("Tg", 100.0*peak["tenengrad"].h),
            ("VL", 100.0*peak["var_laplacian"].h),
            ("ML", 100.0*peak["modified_laplacian"].h),
        ]
        if enable_hf:
            pvals.append(("HF", 100.0*peak["hf_ratio"].h))
        for i,(k,v) in enumerate(pvals):
            draw_text(vis, f"{k}: {v:5.1f}", (px, py + 20 + i*18), 0.6)

        if show_help:
            help_lines = [
                "q quit  space pause/resume  r reset peaks",
                "[ ] resize ROI   w/a/s/d move   c center",
                "m toggle MTF50   h toggle HF    ? help",
            ]
            for i,hl in enumerate(help_lines):
                draw_text(vis, hl, (12, H-12 - i*18), 0.55)

        # Exposure warnings
        if bright < 20 or bright > 235:
            draw_text(vis, "Exposure warning", (W-220, 24), 0.6, (0,0,255), 2)

        # FPS
        fcount += 1
        dt = t0 - last
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt) if fps>0 else (1.0/dt)
        last = t0

        cv2.imshow("Focus HUD", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            for k in scale: scale[k].reset()
            for k in peak: peak[k].reset()
        elif key == ord('c'):
            cx, cy = W//2, H//2
        elif key == ord('['):
            r0 = max(10, int(r0*0.9))
        elif key == ord(']'):
            r0 = min(int(0.95*min(W,H)//2), int(r0*1.111))
        elif key == ord('w'):
            cy = max(r0, cy - max(4, r0//10))
        elif key == ord('s'):
            cy = min(H - r0, cy + max(4, r0//10))
        elif key == ord('a'):
            cx = max(r0, cx - max(4, r0//10))
        elif key == ord('d'):
            cx = min(W - r0, cx + max(4, r0//10))
        elif key == ord('m'):
            enable_mtf = not enable_mtf
        elif key == ord('h'):
            enable_hf = not enable_hf
        elif key == ord('?'):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
