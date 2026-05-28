"""
HYCOM GOFS 3.1 — Vertical current profile analysis across the CCZ.
Covers four larval dispersal simulation windows at 3-day sampling, 00:00 UTC.
Parts 1–3: speed profile, direction profile, larval reachability cross-reference.
Outputs: 5 PNG figures, 3 summary tables.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Ensure all print() calls flush immediately so progress is visible in real time
import builtins as _builtins
_real_print = _builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _real_print(*args, **kwargs)

# ─── Configuration ────────────────────────────────────────────────────────────

URL     = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
OUTDIR  = Path(__file__).parent

LAT_MIN, LAT_MAX = 0.0,   25.0
LON_MIN, LON_MAX = 200.0, 250.0   # 160°W–110°W in 0–360 convention

FILL_VAL = -30000.0
STRIDE   = 4          # spatial stride for OPeNDAP (0.08° × 4 = 0.32° effective)

# Four simulation windows (inclusive endpoints, 00:00 UTC samples every 3 days)
WINDOWS = [
    (datetime(2019, 1,  1), datetime(2019, 3, 10), 'Jan–Mar 2019'),
    (datetime(2019, 7,  1), datetime(2019, 9,  8), 'Jul–Sep 2019'),
    (datetime(2023, 1,  1), datetime(2023, 3, 10), 'Jan–Mar 2023'),
    (datetime(2023, 7,  1), datetime(2023, 9,  8), 'Jul–Sep 2023'),
]

# Part 3 swimming speeds
SWIM_SPEEDS_MMS = [0.1, 0.2, 0.5, 0.6, 1.0]
SWIM_SOURCES = [
    'slow, below measured range',
    'Beaulieu et al. 2015 (lower bound)',
    'Larsson et al. 2014; Strömberg & Larsson 2017',
    'Beaulieu et al. 2015 (upper bound)',
    'hypothetical active swimmer',
]
SEABED_DEPTHS = [4000, 4500, 5000]
PLDS          = [19, 35, 69]

EPOCH      = datetime(2000, 1, 1)
CACHE_FILE = OUTDIR / 'hycom_data_cache.npz'

# ─── Build sample date list ───────────────────────────────────────────────────

def build_sample_dates(windows, step_days=3):
    dates = []
    labels = []
    for start, end, label in windows:
        d = start
        while d <= end:
            dates.append(d)
            labels.append(label)
            d += timedelta(days=step_days)
    return dates, labels

sample_dates, window_labels = build_sample_dates(WINDOWS)
n_t = len(sample_dates)

print(f"Total sample dates: {n_t}")
print("\n=== SAMPLED TIMESTAMPS ===")
for win_label in [w[2] for w in WINDOWS]:
    idxs = [i for i, l in enumerate(window_labels) if l == win_label]
    print(f"\n  {win_label} ({len(idxs)} timestamps):")
    for i in idxs:
        print(f"    {sample_dates[i].strftime('%Y-%m-%d %H:%M UTC')}")

# ─── Load or fetch HYCOM data ─────────────────────────────────────────────────

if CACHE_FILE.exists():
    print(f"\nCache found — loading from {CACHE_FILE}")
    _c           = np.load(CACHE_FILE)
    depths       = _c['depths']
    speed_ts     = _c['speed_ts']
    u_ts         = _c['u_ts']
    v_ts         = _c['v_ts']
    speed_bot_ts = _c['speed_bot_ts']
    u_bot_ts     = _c['u_bot_ts']
    v_bot_ts     = _c['v_bot_ts']
    n_depths     = len(depths)
    print(f"Loaded: {n_t} timesteps × {n_depths} depth levels.")

else:
    print("\nNo cache found — connecting to HYCOM OPeNDAP...")
    import time as _time

    ds = Dataset(URL, 'r')

    lats       = ds.variables['lat'][:]
    lons       = ds.variables['lon'][:]
    depths     = ds.variables['depth'][:]
    times_raw  = ds.variables['time'][:]
    time_units = ds.variables['time'].units
    n_depths   = len(depths)

    lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
    lon_idx = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
    lat_s, lat_e = lat_idx[0], lat_idx[-1] + 1
    lon_s, lon_e = lon_idx[0], lon_idx[-1] + 1

    print(f"Spatial subset: lat {lats[lat_s]:.2f}–{lats[lat_e-1]:.2f}°N, "
          f"lon {lons[lon_s]:.2f}–{lons[lon_e-1]:.2f}°E  "
          f"(stride={STRIDE})")

    def dt_to_hours(dt: datetime) -> float:
        return (dt - EPOCH).total_seconds() / 3600.0

    target_hours = [dt_to_hours(d) for d in sample_dates]
    time_indices = [int(np.argmin(np.abs(times_raw - h))) for h in target_hours]

    print("\nTime index mapping (first 10 shown):")
    actual_times = num2date(times_raw[time_indices[:10]], time_units)
    for td, idx, ct in zip(sample_dates[:10], time_indices[:10], actual_times):
        print(f"  Target {td.strftime('%Y-%m-%d')} → index {idx} = {ct}")

    speed_ts     = np.full((n_t, n_depths), np.nan)
    u_ts         = np.full((n_t, n_depths), np.nan)
    v_ts         = np.full((n_t, n_depths), np.nan)
    speed_bot_ts = np.full(n_t, np.nan)
    u_bot_ts     = np.full(n_t, np.nan)
    v_bot_ts     = np.full(n_t, np.nan)

    print(f"\nFetching {n_t} timesteps with spatial stride={STRIDE}...")
    print(f"Approximate grid per snapshot: "
          f"{(lat_e-lat_s)//STRIDE} lat × {(lon_e-lon_s)//STRIDE} lon points\n")

    _t0 = _time.time()

    for i, tidx in enumerate(time_indices):
        _ts = _time.time()
        lbl = f"{sample_dates[i].strftime('%Y-%m-%d')}  [{window_labels[i]}]"
        print(f"  [{i+1:3d}/{n_t}] {lbl}  (t-index={tidx})", end='  ')

        ub = ds.variables['water_u_bottom'][tidx, lat_s:lat_e:STRIDE, lon_s:lon_e:STRIDE]
        vb = ds.variables['water_v_bottom'][tidx, lat_s:lat_e:STRIDE, lon_s:lon_e:STRIDE]
        ub = np.ma.masked_where(ub < FILL_VAL, ub).filled(np.nan)
        vb = np.ma.masked_where(vb < FILL_VAL, vb).filled(np.nan)
        spd_b = np.sqrt(ub**2 + vb**2)
        speed_bot_ts[i] = np.nanmean(spd_b)
        u_bot_ts[i]     = np.nanmean(ub)
        v_bot_ts[i]     = np.nanmean(vb)

        u3d = ds.variables['water_u'][tidx, :, lat_s:lat_e:STRIDE, lon_s:lon_e:STRIDE]
        v3d = ds.variables['water_v'][tidx, :, lat_s:lat_e:STRIDE, lon_s:lon_e:STRIDE]
        u3d = np.ma.masked_where(u3d < FILL_VAL, u3d).filled(np.nan)
        v3d = np.ma.masked_where(v3d < FILL_VAL, v3d).filled(np.nan)
        spd3d = np.sqrt(u3d**2 + v3d**2)

        for d in range(n_depths):
            speed_ts[i, d] = np.nanmean(spd3d[d])
            u_ts[i, d]     = np.nanmean(u3d[d])
            v_ts[i, d]     = np.nanmean(v3d[d])

        elapsed      = _time.time() - _ts
        total_so_far = _time.time() - _t0
        eta          = (total_so_far / (i + 1)) * (n_t - i - 1)
        print(f"{elapsed:.1f}s  |  ETA {eta/60:.1f} min")

    ds.close()
    print(f"\nData fetch complete. Total time: {(_time.time()-_t0)/60:.1f} min")

    np.savez(
        CACHE_FILE,
        depths=depths,
        speed_ts=speed_ts, u_ts=u_ts, v_ts=v_ts,
        speed_bot_ts=speed_bot_ts, u_bot_ts=u_bot_ts, v_bot_ts=v_bot_ts,
    )
    print(f"Cache saved: {CACHE_FILE}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — SPEED STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

speed_mean   = np.nanmean(speed_ts,   axis=0)   # (n_depths,)
speed_median = np.nanmedian(speed_ts, axis=0)
speed_sd     = np.nanstd(speed_ts,    axis=0, ddof=1)

bot_mean = float(np.nanmean(speed_bot_ts))
bot_sd   = float(np.nanstd(speed_bot_ts, ddof=1))

# Speed ratio per timestep → mean and sd of ratio
ratio_ts   = speed_ts / speed_bot_ts[:, np.newaxis]   # (n_t, n_depths)
ratio_mean = np.nanmean(ratio_ts, axis=0)
ratio_sd   = np.nanstd(ratio_ts,  axis=0, ddof=1)

bot_ratio_mean = 1.0
bot_ratio_sd   = 0.0

print("\n" + "="*80)
print("PART 1 — SPEED PROFILE SUMMARY TABLE")
print(f"{'Depth (m)':>10}  {'Mean speed (m/s)':>18}  {'Median (m/s)':>13}  "
      f"{'SD (m/s)':>10}  {'Ratio to near-bed':>18}  {'Ratio SD':>10}")
print("-"*80)
for d in range(n_depths):
    print(f"{depths[d]:>10.1f}  {speed_mean[d]:>18.5f}  {speed_median[d]:>13.5f}  "
          f"{speed_sd[d]:>10.5f}  {ratio_mean[d]:>18.3f}  {ratio_sd[d]:>10.3f}")
print("-"*80)
print(f"{'BOTTOM':>10}  {bot_mean:>18.5f}  {'—':>13}  "
      f"{bot_sd:>10.5f}  {'1.000':>18}  {'0.000':>10}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DIRECTIONAL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def circular_stats(dir_deg_array):
    """
    Compute circular mean direction and circular SD from array of directions (degrees).
    Returns (mean_dir_deg, circ_sd_deg, R).
    Oceanographic convention: direction current flows TOWARDS, CW from N.
    """
    rad = np.radians(dir_deg_array)
    S = np.nanmean(np.sin(rad))
    C = np.nanmean(np.cos(rad))
    mean_dir = np.degrees(np.arctan2(S, C)) % 360.0
    R = np.sqrt(S**2 + C**2)
    circ_sd = np.degrees(np.sqrt(-2.0 * np.log(np.clip(R, 1e-10, 1.0))))
    return mean_dir, circ_sd, R

def angular_diff(a, b):
    """Signed angular difference a−b wrapped to [−180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0

# Per-timestep spatial-mean direction at each depth (oceanographic: atan2(u, v) CW from N)
dir_ts = np.degrees(np.arctan2(u_ts, v_ts)) % 360.0   # (n_t, n_depths)

# Circular stats across n_t timesteps for each depth
dir_mean   = np.full(n_depths, np.nan)
dir_circ_sd = np.full(n_depths, np.nan)
dir_R      = np.full(n_depths, np.nan)

for d in range(n_depths):
    valid = np.isfinite(dir_ts[:, d])
    if valid.sum() > 1:
        dir_mean[d], dir_circ_sd[d], dir_R[d] = circular_stats(dir_ts[valid, d])

# Bottom direction stats
dir_bot_ts_arr = np.degrees(np.arctan2(u_bot_ts, v_bot_ts)) % 360.0
dir_bot_mean, dir_bot_circ_sd, dir_bot_R = circular_stats(dir_bot_ts_arr[np.isfinite(dir_bot_ts_arr)])

# Angular difference each depth vs near-bed
ang_diff_arr = np.array([angular_diff(dir_mean[d], dir_bot_mean) for d in range(n_depths)])

print("\n" + "="*90)
print("PART 2 — DIRECTION PROFILE SUMMARY TABLE")
print(f"{'Depth (m)':>10}  {'Mean dir (°)':>13}  {'Circ SD (°)':>12}  "
      f"{'Δ from near-bed (°)':>20}  {'Coherence R':>12}")
print("-"*90)
for d in range(n_depths):
    print(f"{depths[d]:>10.1f}  {dir_mean[d]:>13.1f}  {dir_circ_sd[d]:>12.1f}  "
          f"{ang_diff_arr[d]:>+20.1f}  {dir_R[d]:>12.3f}")
print("-"*90)
print(f"{'BOTTOM':>10}  {dir_bot_mean:>13.1f}  {dir_bot_circ_sd:>12.1f}  "
      f"{'0.0':>20}  {dir_bot_R:>12.3f}")
print("="*90)
print(f"\nNear-bed: mean direction {dir_bot_mean:.1f}° ± {dir_bot_circ_sd:.1f}° circular SD  "
      f"(coherence R={dir_bot_R:.3f})")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — LARVAL REACHABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def max_depth_reached(seabed_m: float, speed_mms: float, pld_days: int) -> float:
    """
    Depth (m) reached by continuous upward swimming from seabed.

    Parameters
    ----------
    seabed_m : float
        Seabed depth in metres.
    speed_mms : float
        Upward swimming speed in mm/s.
    pld_days : int
        Planktonic larval duration in days.

    Returns
    -------
    float
        Depth (m below surface) of shallowest point reached; ≥ 0.
    """
    ascent_m = (speed_mms / 1000.0) * pld_days * 86400.0
    return max(0.0, seabed_m - ascent_m)


def required_speed_mms(seabed_m: float, target_depth_m: float, pld_days: int) -> float:
    """
    Sustained upward swimming speed (mm/s) needed to reach target_depth from seabed
    within pld_days.

    Parameters
    ----------
    seabed_m : float
        Seabed depth in metres.
    target_depth_m : float
        Target depth (m below surface); must be < seabed_m.
    pld_days : int
        PLD in days.

    Returns
    -------
    float
        Required speed in mm/s.
    """
    return (seabed_m - target_depth_m) / (pld_days * 86400.0) * 1000.0


def nearest_hycom_level(target_depth: float, depth_levels: np.ndarray) -> tuple[int, float]:
    """
    Return index and value of nearest HYCOM level to target_depth.

    Parameters
    ----------
    target_depth : float
        Target depth in metres.
    depth_levels : np.ndarray
        Array of HYCOM depth levels.

    Returns
    -------
    tuple[int, float]
        (index, depth value) of nearest level.
    """
    idx = int(np.argmin(np.abs(depth_levels - target_depth)))
    return idx, float(depth_levels[idx])


# ─── Required swimming speed per HYCOM level ─────────────────────────────────

PLD_FULL = 69   # full PLD used for required-speed table

print("\n" + "="*100)
print(f"PART 3a — REQUIRED SWIMMING SPEED TO REACH EACH HYCOM LEVEL (PLD = {PLD_FULL} days)")
print(f"Formula: speed (mm/s) = (seabed_depth − target_depth) / ({PLD_FULL} × 86400 s) × 1000")
print("-"*100)
hdr_req = (f"{'HYCOM level (m)':>16}  {'Mean spd (m/s)':>15}  {'Ratio to near-bed':>18}  "
           + "  ".join([f"Req. speed from {s}m (mm/s)" for s in SEABED_DEPTHS]))
print(hdr_req)
print("-"*100)
for d in range(n_depths):
    lev = float(depths[d])
    spd_str   = f"{speed_mean[d]:.5f}" if np.isfinite(speed_mean[d]) else "  NaN  "
    ratio_str = f"{ratio_mean[d]:.3f}"  if np.isfinite(ratio_mean[d]) else "  NaN  "
    req_parts = []
    for sb in SEABED_DEPTHS:
        if lev >= sb:
            req_parts.append(f"{'N/A (below seabed)':>28}")
        else:
            req_parts.append(f"{required_speed_mms(sb, lev, PLD_FULL):>28.4f}")
    print(f"{lev:>16.1f}  {spd_str:>15}  {ratio_str:>18}  {'  '.join(req_parts)}")
print("-"*100)
print(f"{'BOTTOM (near-bed)':>16}  {bot_mean:.5f}  {'1.000':>18}  "
      + "  ".join([f"{'0.0000':>28}"] * len(SEABED_DEPTHS)))
print("="*100)

# Build combined reachability table
print("\n" + "="*130)
print("PART 3 — COMBINED REACHABILITY TABLE")
hdr = (f"{'Speed (mm/s)':>12}  {'Source':>42}  {'Seabed (m)':>10}  "
       f"{'PLD (d)':>7}  {'Depth reached (m)':>18}  "
       f"{'HYCOM level (m)':>15}  {'Speed (m/s)':>12}  {'Speed±SD':>16}  "
       f"{'Ratio':>7}  {'Ratio±SD':>10}  {'Dir (°)':>8}  {'Dir±SD':>10}  {'ΔDir (°)':>10}")
print(hdr)
print("-"*130)

# Store rows for later use in figure
reach_rows = []

for speed_mms, source in zip(SWIM_SPEEDS_MMS, SWIM_SOURCES):
    for seabed in SEABED_DEPTHS:
        for pld in PLDS:
            dr = max_depth_reached(seabed, speed_mms, pld)
            # Only levels shallower than seabed are relevant
            valid_levels = depths[depths < seabed]
            if dr > valid_levels.max():
                # Cannot reach any HYCOM level above seabed
                row = (speed_mms, source, seabed, pld, dr, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                print(f"{speed_mms:>12.1f}  {source:>42}  {seabed:>10}  "
                      f"{pld:>7}  {dr:>18.1f}  {'below HYCOM levels':>18}")
                reach_rows.append(row)
                continue

            hidx, hlevel = nearest_hycom_level(dr, depths)
            # Ensure we use a level that's shallower than seabed
            while depths[hidx] >= seabed and hidx > 0:
                hidx -= 1
            hlevel = float(depths[hidx])

            spd_val  = float(speed_mean[hidx])
            spd_sd   = float(speed_sd[hidx])
            rat_val  = float(ratio_mean[hidx])
            rat_sd   = float(ratio_sd[hidx])
            dir_val  = float(dir_mean[hidx])
            dir_sd   = float(dir_circ_sd[hidx])
            adir_val = float(ang_diff_arr[hidx])

            print(f"{speed_mms:>12.1f}  {source[:42]:>42}  {seabed:>10}  "
                  f"{pld:>7}  {dr:>18.1f}  {hlevel:>15.1f}  "
                  f"{spd_val:>12.5f}  {spd_val:.4f}±{spd_sd:.4f}  "
                  f"{rat_val:>7.3f}  {rat_val:.3f}±{rat_sd:.3f}  "
                  f"{dir_val:>8.1f}  {dir_val:.1f}±{dir_sd:.1f}  {adir_val:>+10.1f}")

            row = (speed_mms, source, seabed, pld, dr, hlevel,
                   spd_val, spd_sd, rat_val, rat_sd, dir_val, dir_sd, adir_val)
            reach_rows.append(row)

print("="*130)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Mean speed vs depth with ±1 SD shading
# ═══════════════════════════════════════════════════════════════════════════════

fig1, ax = plt.subplots(figsize=(7, 10))

valid = np.isfinite(speed_mean)
ax.fill_betweenx(depths[valid],
                 speed_mean[valid] - speed_sd[valid],
                 speed_mean[valid] + speed_sd[valid],
                 color='steelblue', alpha=0.20, label='±1 SD')
ax.plot(speed_mean[valid], depths[valid], 'o-', color='steelblue', lw=2,
        ms=5, label='Mean speed')
ax.axvspan(bot_mean - bot_sd, bot_mean + bot_sd,
           alpha=0.12, color='firebrick', label='Near-bed ±1 SD')
ax.axvline(bot_mean, color='firebrick', lw=2.5, ls='--',
           label=f'Near-bed mean: {bot_mean:.4f} m/s')

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlim(left=0)
ax.set_xlabel('Mean current speed (m/s)', fontsize=13)
ax.set_ylabel('Depth (m)', fontsize=13)
ax.set_title(
    'HYCOM GOFS 3.1 — Mean current speed vs depth\n'
    'CCZ region (160–110°W, 0–25°N)\n'
    f'4 windows × ~23 timesteps (3-daily, 00:00 UTC), n={n_t} total',
    fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)

fig1.tight_layout()
fig1.savefig(OUTDIR / 'hycom_speed_profile_full.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: hycom_speed_profile_full.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Speed ratio vs depth with ±1 SD shading
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(7, 10))

ax.fill_betweenx(depths[valid],
                 ratio_mean[valid] - ratio_sd[valid],
                 ratio_mean[valid] + ratio_sd[valid],
                 color='darkorange', alpha=0.20, label='±1 SD')
ax.plot(ratio_mean[valid], depths[valid], 's-', color='darkorange', lw=2,
        ms=5, label='Mean ratio')
ax.axvline(1.0, color='firebrick', lw=2.5, ls='--', label='Near-bed (= 1.0)')
ax.axvline(1.5, color='grey', lw=1.2, ls=':', alpha=0.8, label='1.5× near-bed')
ax.axvline(2.0, color='grey', lw=1.2, ls='--', alpha=0.8, label='2× near-bed')

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlabel('Speed ratio (depth level / near-bed)', fontsize=13)
ax.set_ylabel('Depth (m)', fontsize=13)
ax.set_title(
    'Speed ratio relative to near-bed speed\n'
    'CCZ region — HYCOM GOFS 3.1\n'
    f'n={n_t} timesteps across 4 simulation windows',
    fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)

fig2.tight_layout()
fig2.savefig(OUTDIR / 'hycom_speed_ratio_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: hycom_speed_ratio_full.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Direction profile: mean direction + angular diff + coherence R
# ═══════════════════════════════════════════════════════════════════════════════

fig3, axes = plt.subplots(1, 3, figsize=(17, 10), sharey=True)
valid_dir = np.isfinite(dir_mean)

# Panel 1: Mean flow direction with ±1 circular SD shading
ax = axes[0]
lo = (dir_mean[valid_dir] - dir_circ_sd[valid_dir]) % 360.0
hi = (dir_mean[valid_dir] + dir_circ_sd[valid_dir]) % 360.0

# Shade ±1 circular SD (handle wrap at 0/360 by simply clipping — acceptable for manuscript)
ax.fill_betweenx(depths[valid_dir],
                 dir_mean[valid_dir] - dir_circ_sd[valid_dir],
                 dir_mean[valid_dir] + dir_circ_sd[valid_dir],
                 color='steelblue', alpha=0.20, label='±1 circular SD')
ax.plot(dir_mean[valid_dir], depths[valid_dir], 'o-', color='steelblue',
        lw=2, ms=5, label='Mean direction')
ax.axvline(dir_bot_mean, color='firebrick', lw=2.5, ls='--',
           label=f'Near-bed: {dir_bot_mean:.1f}°')
ax.axvspan(dir_bot_mean - dir_bot_circ_sd, dir_bot_mean + dir_bot_circ_sd,
           alpha=0.12, color='firebrick')

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xticklabels(['N\n0°', 'E\n90°', 'S\n180°', 'W\n270°', 'N\n360°'])
ax.set_xlabel('Mean flow direction (°, CW from N)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Mean flow direction\nvs depth', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)

# Panel 2: Angular difference from near-bed (bar chart, signed)
ax = axes[1]
bar_heights = np.diff(np.concatenate([depths[valid_dir],
                                       [depths[valid_dir][-1] + 100]]))
colors_bar = ['steelblue' if d >= 0 else 'darkorange'
              for d in ang_diff_arr[valid_dir]]
ax.barh(depths[valid_dir], ang_diff_arr[valid_dir],
        height=bar_heights, color=colors_bar, alpha=0.75, align='edge')
ax.axvline(0,   color='firebrick', lw=2,   ls='--', label='Near-bed direction')
ax.axvline(30,  color='grey',      lw=1.2, ls=':',  alpha=0.8, label='±30°')
ax.axvline(-30, color='grey',      lw=1.2, ls=':',  alpha=0.8)
ax.axvline(90,  color='grey',      lw=1.2, ls='--', alpha=0.8, label='±90°')
ax.axvline(-90, color='grey',      lw=1.2, ls='--', alpha=0.8)

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlabel('Angular difference from near-bed (°)', fontsize=12)
ax.set_title('Direction offset\nfrom near-bed', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35, axis='x')
cw_patch  = mpatches.Patch(color='steelblue',  alpha=0.75, label='CW (positive)')
ccw_patch = mpatches.Patch(color='darkorange', alpha=0.75, label='CCW (negative)')
ax.legend(handles=[cw_patch, ccw_patch], fontsize=9, loc='lower right')

# Panel 3: Directional coherence R
ax = axes[2]
ax.plot(dir_R[valid_dir], depths[valid_dir], 's-', color='purple',
        lw=2, ms=5, label='3D levels')
ax.axvline(dir_bot_R, color='firebrick', lw=2.5, ls='--',
           label=f'Near-bed: {dir_bot_R:.3f}')
ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlim(0, 1)
ax.set_xlabel('Directional coherence R (0–1)', fontsize=12)
ax.set_title('Directional coherence\n(mean resultant length)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)

fig3.suptitle(
    'HYCOM GOFS 3.1 — Flow directionality vs depth\n'
    f'CCZ region (160–110°W, 0–25°N) | n={n_t} timesteps across 4 simulation windows',
    fontsize=12, y=1.01)
plt.tight_layout()
fig3.savefig(OUTDIR / 'hycom_direction_profile_full.png', dpi=150, bbox_inches='tight')
print(f"Saved: hycom_direction_profile_full.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Reachable depth vs swimming speed (panels per seabed depth)
# ═══════════════════════════════════════════════════════════════════════════════

# Identify oceanographic threshold depths for reference lines
speed_1p5x_depth = None
speed_2x_depth   = None
speed_10x_depth  = None
ang30_depth      = None
ang90_depth      = None

for d in range(n_depths - 1, -1, -1):   # search from deep to shallow
    if np.isfinite(ratio_mean[d]) and ratio_mean[d] > 1.5 and speed_1p5x_depth is None:
        speed_1p5x_depth = float(depths[d])
    if np.isfinite(ratio_mean[d]) and ratio_mean[d] > 2.0 and speed_2x_depth is None:
        speed_2x_depth = float(depths[d])
    if np.isfinite(ratio_mean[d]) and ratio_mean[d] > 10.0 and speed_10x_depth is None:
        speed_10x_depth = float(depths[d])
    if np.isfinite(ang_diff_arr[d]) and abs(ang_diff_arr[d]) > 30 and ang30_depth is None:
        ang30_depth = float(depths[d])
    if np.isfinite(ang_diff_arr[d]) and abs(ang_diff_arr[d]) > 90 and ang90_depth is None:
        ang90_depth = float(depths[d])

# Depth of maximum directional divergence from near-bed
_valid_ang = np.where(np.isfinite(ang_diff_arr))[0]
_ang_max_idx  = _valid_ang[np.argmax(np.abs(ang_diff_arr[_valid_ang]))]
ang_max_depth = float(depths[_ang_max_idx])
ang_max_val   = float(ang_diff_arr[_ang_max_idx])   # signed degrees

print(f"\nOceanographic thresholds (deepest depth where criterion first met from surface):")
print(f"  Speed ratio >1.5×: {speed_1p5x_depth} m")
print(f"  Speed ratio >2×:   {speed_2x_depth} m")
print(f"  Speed ratio >10×:  {speed_10x_depth} m")
print(f"  |Δdir| >30°:       {ang30_depth} m")
print(f"  |Δdir| >90°:       {ang90_depth} m")
print(f"  Max |Δdir|:        {abs(ang_max_val):.1f}° at {ang_max_depth} m")

swim_range = np.logspace(np.log10(0.05), np.log10(2.0), 200)  # mm/s

pld_colors = {19: '#e6550d', 35: '#fdae6b', 69: '#2171b5'}
pld_ls     = {19: '-',      35: '--',      69: ':'}

fig4, axes4 = plt.subplots(1, 3, figsize=(16, 9), sharey=True)

for ax, seabed in zip(axes4, SEABED_DEPTHS):
    for pld in PLDS:
        reach = np.array([max_depth_reached(seabed, s, pld) for s in swim_range])
        # clip to seabed (can't go deeper than seabed, and no negative depth)
        reach = np.clip(reach, 0, seabed)
        ax.plot(swim_range, reach, color=pld_colors[pld], lw=2, ls=pld_ls[pld],
                label=f'PLD {pld} d')

    # Oceanographic reference lines
    ref_lines = [
        (speed_2x_depth,  '#555555', ':',  'Speed >2× near-bed'),
        (speed_10x_depth, '#222222', '--', 'Speed >10× near-bed'),
        (ang_max_depth,   '#8B0000', '--', f'Max |Δdir| = {abs(ang_max_val):.0f}°'),
    ]
    for ref_d, rc, rls, rlbl in ref_lines:
        if ref_d is not None and ref_d < seabed:
            ax.axhline(ref_d, color=rc, lw=1.5, ls=rls, alpha=0.75, label=rlbl)

    # Mark literature swimming speeds
    for s_mms in SWIM_SPEEDS_MMS:
        ax.axvline(s_mms, color='lightgrey', lw=0.8, alpha=0.6)

    ax.invert_yaxis()
    ax.set_ylim(seabed * 1.02, 0)
    ax.set_xscale('log')
    ax.set_xlim(0.05, 2.0)
    ax.set_xlabel('Upward swimming speed (mm/s)', fontsize=11)
    ax.set_title(f'Seabed = {seabed} m', fontsize=12)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')

axes4[0].set_ylabel('Maximum depth reached (m)', fontsize=12)
fig4.suptitle(
    'Larval reachable depth vs swimming speed — CCZ seabed depths\n'
    'Continuous upward swimming within PLD windows',
    fontsize=12)
plt.tight_layout()
fig4.savefig(OUTDIR / 'hycom_reachability_curves.png', dpi=150, bbox_inches='tight')
print(f"Saved: hycom_reachability_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Combined context: speed ratio and direction profiles
#            annotated with reachability for key swimming speeds
# ═══════════════════════════════════════════════════════════════════════════════

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 10), sharey=True)

seabed_colors = {4000: '#2196F3', 4500: '#FF9800', 5000: '#9C27B0'}
speed_markers = {0.2: 'o', 0.5: 's', 1.0: '^'}
pld_ref = 69   # use full PLD for annotation

# Left panel: speed ratio profile with reachability bands
ax = axes5[0]
ax.fill_betweenx(depths[valid],
                 ratio_mean[valid] - ratio_sd[valid],
                 ratio_mean[valid] + ratio_sd[valid],
                 color='darkorange', alpha=0.20)
ax.plot(ratio_mean[valid], depths[valid], 's-', color='darkorange',
        lw=2.5, ms=5, label='Mean ratio ±1 SD')
ax.axvline(1.0, color='firebrick', lw=2, ls='--', label='Near-bed = 1.0')
ax.axvline(1.5, color='grey', lw=1.2, ls=':', alpha=0.9, label='1.5×')
ax.axvline(2.0, color='grey', lw=1.2, ls='--', alpha=0.9, label='2×')

# Add horizontal markers: deepest reachable HYCOM level for each seabed × swim speed
for seabed in SEABED_DEPTHS:
    for s_mms, mstyle in speed_markers.items():
        dr = max_depth_reached(seabed, s_mms, pld_ref)
        hidx, hlevel = nearest_hycom_level(dr, depths)
        while depths[hidx] >= seabed and hidx > 0:
            hidx -= 1
        hlevel = float(depths[hidx])
        rat = ratio_mean[hidx]
        ax.plot(rat, hlevel, marker=mstyle, color=seabed_colors[seabed],
                ms=9, zorder=5,
                label=f'{seabed}m bed, {s_mms}mm/s → {hlevel:.0f}m (ratio={rat:.2f})')

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlabel('Speed ratio (depth level / near-bed)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Speed ratio profile\n+ reachable depths (PLD=69 d)', fontsize=11)
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.35)

# Right panel: angular difference profile with reachability markers
ax = axes5[1]
ax.fill_betweenx(depths[valid_dir],
                 ang_diff_arr[valid_dir] - dir_circ_sd[valid_dir],
                 ang_diff_arr[valid_dir] + dir_circ_sd[valid_dir],
                 color='steelblue', alpha=0.20, label='±1 circ SD')
ax.plot(ang_diff_arr[valid_dir], depths[valid_dir], 'o-', color='steelblue',
        lw=2.5, ms=5, label='Angular diff from near-bed')
ax.axvline(0,   color='firebrick', lw=2,   ls='--', label='Near-bed direction')
ax.axvline(30,  color='grey', lw=1.2, ls=':', alpha=0.9, label='±30°')
ax.axvline(-30, color='grey', lw=1.2, ls=':', alpha=0.9)
ax.axvline(90,  color='grey', lw=1.2, ls='--', alpha=0.9, label='±90°')
ax.axvline(-90, color='grey', lw=1.2, ls='--', alpha=0.9)

for seabed in SEABED_DEPTHS:
    for s_mms, mstyle in speed_markers.items():
        dr = max_depth_reached(seabed, s_mms, pld_ref)
        hidx, hlevel = nearest_hycom_level(dr, depths)
        while depths[hidx] >= seabed and hidx > 0:
            hidx -= 1
        hlevel = float(depths[hidx])
        adir = ang_diff_arr[hidx]
        ax.plot(adir, hlevel, marker=mstyle, color=seabed_colors[seabed],
                ms=9, zorder=5,
                label=f'{seabed}m, {s_mms}mm/s → {hlevel:.0f}m (Δdir={adir:+.0f}°)')

ax.invert_yaxis()
ax.set_ylim(5200, 0)
ax.set_xlabel('Angular difference from near-bed direction (°)', fontsize=12)
ax.set_title('Direction offset profile\n+ reachable depths (PLD=69 d)', fontsize=11)
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.35)

# Shared legend patches for seabed colours
legend_patches = [mpatches.Patch(color=seabed_colors[s], label=f'Seabed {s} m')
                  for s in SEABED_DEPTHS]
marker_patches = [plt.Line2D([0], [0], marker=m, color='k', ls='None',
                              ms=8, label=f'{s} mm/s')
                  for s, m in speed_markers.items()]
fig5.legend(handles=legend_patches + marker_patches, fontsize=9,
            loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.04))

fig5.suptitle(
    'Oceanographic context at larval reachable depths — HYCOM GOFS 3.1\n'
    f'CCZ region | n={n_t} timesteps | markers = max depth reached in 69-day PLD',
    fontsize=12)
plt.tight_layout()
fig5.savefig(OUTDIR / 'hycom_reachability_context.png', dpi=150, bbox_inches='tight')
print(f"Saved: hycom_reachability_context.png")

print("\nAll done. Five figures written to:", OUTDIR)

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE — 1 × 3 publication layout
# ═══════════════════════════════════════════════════════════════════════════════

def make_combined_figure() -> None:
    """
    Assemble a single 1×3 publication-quality figure from three analyses.

    Parameters
    ----------
    None — uses script-level variables.

    Panels
    ------
    (a) Mean current speed ± 1 SD vs depth (Part 1).
    (b) Mean flow direction ± 1 circular SD vs depth; near-bed direction and
        circular SD shown as a marker + shaded band below the deepest HYCOM
        level (Part 2, panel 0).
    (c) Maximum larval reachable depth vs upward swimming speed for seabed
        depths 4000, 4500, 5000, 5500 m across PLDs of 19, 35, 69 d, with
        oceanographic threshold reference lines (Part 3, Fig 4 extended).

    Output
    ------
    {OUTDIR}/hycom_combined_figure.png at 300 dpi.
    """
    SEABED_DEPTHS_EXT = [4000, 4500, 5000, 5500]
    seabed_colors_ext = {
        4000: '#2171b5',
        4500: '#e6550d',
        5000: '#31a354',
        5500: '#756bb1',
    }
    swim_range_loc = np.logspace(np.log10(0.05), np.log10(2.0), 200)

    fig, axes = plt.subplots(1, 3, figsize=(21, 10))

    # ── (a) Mean current speed vs depth ───────────────────────────────────────
    ax = axes[0]
    ax.fill_betweenx(
        depths[valid],
        speed_mean[valid] - speed_sd[valid],
        speed_mean[valid] + speed_sd[valid],
        color='steelblue', alpha=0.20, label='±1 SD',
    )
    ax.plot(speed_mean[valid], depths[valid], 'o-', color='steelblue',
            lw=2, ms=5, label='Mean speed')
    ax.axvspan(bot_mean - bot_sd, bot_mean + bot_sd,
               alpha=0.12, color='firebrick', label='Near-bed ±1 SD')
    ax.axvline(bot_mean, color='firebrick', lw=2.5, ls='--',
               label=f'Near-bed: {bot_mean:.4f} m/s')
    ax.invert_yaxis()
    ax.set_ylim(5200, 0)
    ax.set_xlim(left=0)
    ax.set_xlabel('Mean current speed (m/s)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('(a) Current speed profile', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── (b) Mean flow direction vs depth with near-bed circular SD ────────────
    ax = axes[1]
    ax.fill_betweenx(
        depths[valid_dir],
        dir_mean[valid_dir] - dir_circ_sd[valid_dir],
        dir_mean[valid_dir] + dir_circ_sd[valid_dir],
        color='steelblue', alpha=0.20, label='±1 circular SD',
    )
    ax.plot(dir_mean[valid_dir], depths[valid_dir], 'o-', color='steelblue',
            lw=2, ms=5, label='Mean direction')

    # Near-bed: marker + fill_betweenx band below deepest HYCOM level
    near_bed_y  = float(depths[valid_dir][-1]) + 120.0
    band_half_m = 60.0
    ax.fill_betweenx(
        [near_bed_y - band_half_m, near_bed_y + band_half_m],
        dir_bot_mean - dir_bot_circ_sd,
        dir_bot_mean + dir_bot_circ_sd,
        color='firebrick', alpha=0.25,
        label=f'Near-bed ±1 circ SD ({dir_bot_circ_sd:.1f}°)',
    )
    ax.plot(dir_bot_mean, near_bed_y, 'D', color='firebrick', ms=8, zorder=5,
            label=f'Near-bed: {dir_bot_mean:.1f}°')

    ax.invert_yaxis()
    ax.set_ylim(near_bed_y + band_half_m + 40, 0)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['N\n0°', 'E\n90°', 'S\n180°', 'W\n270°', 'N\n360°'])
    ax.set_xlabel('Mean flow direction (°, CW from N)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('(b) Flow direction profile', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # ── (c) Larval reachable depth vs time — seabed × swim speed combos ─────
    ax = axes[2]

    t_days = np.linspace(0, 69, 500)
    pld_marker_days  = [19, 35, 69]
    pld_markers      = {19: 'o', 35: 's', 69: '^'}
    pld_marker_label = {19: 'PLD 19 d', 35: 'PLD 35 d', 69: 'PLD 69 d'}

    swim_colors = {
        0.1: '#aec7e8',
        0.2: '#2ca02c',
        0.5: '#1f77b4',
        0.6: '#ff7f0e',
        1.0: '#d62728',
    }
    seabed_ls = {4000: '-', 4500: '--', 5000: ':'}

    for seabed in [4000, 4500, 5000]:
        for s_mms in SWIM_SPEEDS_MMS:
            reach = np.clip(seabed - (s_mms / 1000.0) * t_days * 86400.0, 0, seabed)
            ax.plot(t_days, reach,
                    color=swim_colors[s_mms], lw=1.6, ls=seabed_ls[seabed],
                    alpha=0.85)
            # PLD milestone markers
            for pld_d in pld_marker_days:
                r_pld = max(0.0, seabed - (s_mms / 1000.0) * pld_d * 86400.0)
                r_pld = min(r_pld, seabed)
                ax.plot(pld_d, r_pld,
                        marker=pld_markers[pld_d],
                        color=swim_colors[s_mms], ms=6,
                        markeredgecolor='k', markeredgewidth=0.4,
                        ls='None', zorder=5)

    # Oceanographic reference lines (speed thresholds in legend, direction annotated)
    speed_ref_lines = [
        (speed_2x_depth,  '#555555', ':',  'Speed >2× near-bed'),
        (speed_10x_depth, '#222222', '--', 'Speed >10× near-bed'),
    ]
    for ref_d, rc, rls, rlbl in speed_ref_lines:
        if ref_d is not None:
            ax.axhline(ref_d, color=rc, lw=1.5, ls=rls, alpha=0.80, label=rlbl)

    # Max directional divergence — annotated directly above the line
    ax.axhline(ang_max_depth, color='#8B0000', lw=1.5, ls='--', alpha=0.80)
    ax.text(1, ang_max_depth - 30,
            f'Max Δdir from seabed: {abs(ang_max_val):.0f}°',
            color='#8B0000', fontsize=7.5, va='bottom', ha='left')

    # Legend entries
    speed_handles = [
        plt.Line2D([0], [0], color=swim_colors[s], lw=2, label=f'{s} mm/s')
        for s in SWIM_SPEEDS_MMS
    ]
    seabed_handles = [
        plt.Line2D([0], [0], color='k', lw=1.8, ls=seabed_ls[sb], label=f'Seabed {sb} m')
        for sb in [4000, 4500, 5000]
    ]
    pld_handles = [
        plt.Line2D([0], [0], marker=pld_markers[p], color='k', ls='None',
                   ms=7, markeredgecolor='k', label=pld_marker_label[p])
        for p in pld_marker_days
    ]
    ref_handles = [
        plt.Line2D([0], [0], color=rc, lw=1.5, ls=rls, alpha=0.80, label=rlbl)
        for ref_d, rc, rls, rlbl in speed_ref_lines if ref_d is not None
    ]
    ax.legend(handles=speed_handles + seabed_handles + pld_handles + ref_handles,
              fontsize=7.5, loc='upper left',
              title='Speed / Seabed / PLD / Thresholds', title_fontsize=7.5)

    ax.invert_yaxis()
    ax.set_ylim(5100, 0)
    ax.set_xlim(0, 75)
    ax.set_xlabel('Time elapsed (days)', fontsize=12)
    ax.set_ylabel('Maximum depth reached (m)', fontsize=12)
    ax.set_title('(c) Larval reachability', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        'HYCOM GOFS 3.1 — Current profile and larval reachability, CCZ region\n'
        f'(160–110°W, 0–25°N) | n={n_t} timesteps across 4 simulation windows',
        fontsize=13,
    )
    plt.tight_layout()
    out_path = OUTDIR / 'hycom_combined_figure.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: hycom_combined_figure.png")


make_combined_figure()
