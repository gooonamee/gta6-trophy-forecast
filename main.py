#!/usr/bin/env python3
"""Deterministic trophy rarity forecasts; uses only the Python standard library."""
import datetime as dt
import glob
import json
import math
import os

RELEASE = dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc).timestamp()
Z = (-1.644853626951, -0.674489750196, 0.0, 0.674489750196, 1.644853626951)


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def logtime(t):
    return math.log1p(max(0.0, (t - RELEASE) / 3600.0))


def logit(rate):
    p = min(99.99, max(0.01, rate)) / 100.0
    return math.log(p / (1.0 - p))


def percent(value):
    value = max(-30.0, min(30.0, value))
    return min(99.99, max(0.01, 100.0 / (1.0 + math.exp(-value))))


def regression(points, weights=None):
    n = len(points)
    if n < 2:
        return 0.0, 0.0, 0.0
    weights = weights or [1.0] * n
    total = sum(weights)
    mx = sum(w * x for (x, _), w in zip(points, weights)) / total
    my = sum(w * y for (_, y), w in zip(points, weights)) / total
    sxx = sum(w * (x - mx) ** 2 for (x, _), w in zip(points, weights))
    slope = sum(w * (x - mx) * (y - my) for (x, y), w in zip(points, weights)) / sxx if sxx > 1e-12 else 0.0
    residual = math.sqrt(sum(w * (y - my - slope * (x - mx)) ** 2 for (x, y), w in zip(points, weights)) / total)
    return slope, residual, sxx


def forecast(case):
    history = sorted(case.get('history', []), key=lambda row: (timestamp(row['at']), row.get('slot', 0)))
    target = timestamp(case['target'])
    checkpoint = timestamp(case['at'])
    horizon_days = max(0.0, float(case['horizon_minutes']) / 1440.0)
    tx = logtime(target)
    # A 24-hour half-life retains early curvature while favoring current dynamics.
    def weights(times):
        return [math.exp(-math.log(2.0) * max(0.0, checkpoint - t) / 86400.0) for t in times]

    player_rows = [(timestamp(r['at']), float(r['players_tracked']))
                   for r in history if r.get('players_tracked', 0) > 0]
    player_points = [(logtime(t), math.log(p)) for t, p in player_rows]
    player_slope, _, _ = regression(player_points, weights([t for t, _ in player_rows]))
    player_slope = max(0.0, min(3.0, player_slope))
    series = []
    groups = {}
    for i, trophy in enumerate(case['trophies']):
        rows = [(timestamp(r['at']), float(r['players_tracked']), float(r['rates'][i]))
                for r in history if i < len(r.get('rates', [])) and r['rates'][i] is not None and r.get('players_tracked', 0) > 0]
        recent = [(logtime(t), logit(rate)) for t, _, rate in rows]
        slope, residual, sxx = regression(recent, weights([t for t, _, _ in rows]))
        info = (rows, recent, slope, residual, sxx)
        series.append(info)
        if len(recent) >= 3 and sxx > 1e-10:
            # Reduce the influence of noisy or very short fits on the tier prior.
            weight = min(1.0, sxx / 0.25) / (1.0 + residual * residual * 8.0)
            groups.setdefault(trophy.get('tier', ''), []).append((slope, weight))
    priors = {tier: sum(s * w for s, w in vals) / sum(w for _, w in vals)
              for tier, vals in groups.items() if sum(w for _, w in vals) > 0}
    result = []
    for trophy, (rows, recent, slope, residual, sxx) in zip(case['trophies'], series):
        if not rows:
            center, spread = logit(0.01), 0.5 * math.sqrt(horizon_days)
        else:
            last_t, last_players, last_rate = rows[-1]
            last = logit(last_rate)
            center = last
            rw = 0.25 * math.sqrt(horizon_days)
            spread = rw
            if len(rows) >= 3 and len(recent) >= 3 and sxx > 1e-10:
                dx = max(0.0, tx - logtime(last_t))
                prior = priors.get(trophy.get('tier', ''), 0.0)
                reliability = (sxx / (sxx + 0.35)) / (1.0 + 5.0 * residual * residual)
                shrunk = reliability * slope + (1.0 - reliability) * prior
                # Curb extrapolation of early, poorly supported slopes.
                shrunk = max(-3.0, min(3.0, shrunk))
                quality = min(1.0, len(recent) / 12.0) / (1.0 + 2.0 * residual)
                blend = (1.0 - math.exp(-horizon_days / 0.8)) * (0.65 + 0.35 * quality)
                center += blend * shrunk * dx
                slope_error = residual / math.sqrt(sxx + 0.15)
                spread = math.sqrt(rw * rw + (0.65 * residual) ** 2 + (0.5 * dx * slope_error) ** 2)
                # Even a completely static page can refresh and dilute rates.
                if max(y for _, y in recent) - min(y for _, y in recent) < 1e-8:
                    center -= min(0.15, 0.12 * math.sqrt(horizon_days))
                predicted_players = last_players * math.exp(min(4.0, player_slope * dx))
                growth = predicted_players / last_players
                earner_rows = [(t, max(0.01, p * rate / 100.0)) for t, p, rate in rows if rate > 0]
                if len(earner_rows) >= 3:
                    ep = [(logtime(t), math.log(e)) for t, e in earner_rows]
                    eslope, enoise, _ = regression(ep, weights([t for t, _ in earner_rows]))
                    last_earners = last_players * last_rate / 100.0
                    predicted_earners = max(last_earners, last_earners * math.exp(min(4.0, max(0.0, eslope) * dx)))
                    earner_center = logit(100.0 * predicted_earners / predicted_players)
                    confidence = min(1.0, len(earner_rows) / 24.0) / (1.0 + 8.0 * enoise)
                    view_weight = 0.9 * (1.0 - math.exp(-2.0 * max(0.0, math.log(growth)))) * confidence
                    center += view_weight * (earner_center - center)
                spread = math.sqrt(spread * spread + (0.18 * math.log(growth)) ** 2)
                floor_rate = last_rate / growth
                center = max(center, logit(floor_rate))
            # A rounded boundary is interval-censored, not literal certainty.
            if last_rate >= 100.0:
                center = min(center, logit(99.95))
            elif last_rate <= 0.0:
                center = max(center, logit(0.015))
        spread = min(2.25, spread)
        quantiles = [round(percent(center + z * spread), 6) for z in Z]
        for j in range(1, len(quantiles)):
            quantiles[j] = max(quantiles[j], quantiles[j - 1])
        result.append({'id': trophy['id'], 'q': quantiles})
    return {'trophies': result}


def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(input_dir, '*.in'))):
        with open(path, encoding='utf-8') as f:
            case = json.load(f)
        output = forecast(case)
        with open(os.path.join(output_dir, str(case['id']) + '.out'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(output, f, separators=(',', ':'), allow_nan=False)
            f.write('\n')


if __name__ == '__main__':
    main()
