#!/usr/bin/env python3
"""Compare truncated practice histories against their final observed read."""
import copy
import datetime as dt
import glob
import json
import math
import os

from main import RELEASE, Z, forecast, logit, logtime, percent, regression, timestamp


def score(prediction, truth):
    total = 0.0
    count = 0
    for row, actual in zip(prediction['trophies'], truth):
        if actual is None:
            continue
        for tau, q in zip((0.05, 0.25, 0.5, 0.75, 0.95), row['q']):
            difference = logit(actual) - logit(q)
            total += 2.0 * max(tau * difference, (tau - 1.0) * difference)
            count += 1
    return total / count if count else float('nan')


def run(path):
    with open(path, encoding='utf-8') as f:
        original = json.load(f)
    history = original['history']
    final = history[-1]
    for cut in sorted(set((2, 3, len(history) // 2, len(history) - 2))):
        if cut < 1 or cut >= len(history):
            continue
        case = copy.deepcopy(original)
        case['history'] = history[:cut]
        case['at'] = history[cut - 1]['at']
        case['target'] = final['at']
        case['horizon_minutes'] = max(0.0, (timestamp(final['at']) - timestamp(case['at'])) / 60.0)
        spread = 0.5 * math.sqrt(case['horizon_minutes'] / 1440.0)
        baseline = {'trophies': []}
        for i, trophy in enumerate(case['trophies']):
            rates = [r['rates'][i] for r in case['history'] if i < len(r['rates']) and r['rates'][i] is not None]
            center = logit(rates[-1] if rates else 0.01)
            baseline['trophies'].append({'id': trophy['id'], 'q': [percent(center + z * spread) for z in Z]})
        print(f"{os.path.basename(path)} reads={cut:2d} horizon={case['horizon_minutes']:6.1f} min  model={score(forecast(case), final['rates']):.6f}  no-change={score(baseline, final['rates']):.6f}")


def synthetic():
    # Player growth is multiplicative in log-time; earners approach distinct
    # ceilings through a saturating hazard, including rounded edge cases.
    specs = [(0.98, 1.3), (0.90, 0.75), (0.8, 0.42), (0.65, 0.25),
             (0.45, 0.13), (0.30, 0.07), (0.18, 0.035), (0.08, 0.012),
             (9.48, 0.1), (0.001, 0.015)]
    def values(hours):
        x = math.log1p(hours)
        players = 100000 * math.exp(0.30 * x)
        rates = [100 * ceiling * (1 - math.exp(-speed * x)) / math.exp(0.30 * x)
                 for ceiling, speed in specs]
        return players, rates
    def iso(hours):
        return dt.datetime.fromtimestamp(RELEASE + 3600 * hours, dt.timezone.utc).isoformat()
    history = []
    for i in range(121):
        hours = 1 + i * 0.5
        players, rates = values(hours)
        history.append({'at': iso(hours), 'players_tracked': round(players),
                        'rates': [round(r, 2) for r in rates]})
    target_hours = 169
    case = {'at': history[-1]['at'], 'target': iso(target_hours),
            'horizon_minutes': (target_hours - 61) * 60, 'history': history,
            'trophies': [{'id': str(i), 'tier': str(i // 3)} for i in range(len(specs))]}
    truth = values(target_hours)[1]
    rw = 0.5 * math.sqrt(case['horizon_minutes'] / 1440)
    no_change, recent = {'trophies': []}, {'trophies': []}
    for i, trophy in enumerate(case['trophies']):
        center = logit(history[-1]['rates'][i])
        no_change['trophies'].append({'id': trophy['id'], 'q': [percent(center + z * rw) for z in Z]})
        points = [(logtime(timestamp(r['at'])), logit(r['rates'][i])) for r in history[-49:]]
        slope, residual, _ = regression(points)
        predicted = center + slope * (logtime(timestamp(case['target'])) - points[-1][0])
        spread = math.sqrt(rw * rw + residual * residual)
        recent['trophies'].append({'id': trophy['id'], 'q': [percent(predicted + z * spread) for z in Z]})
    print(f"synthetic: model={score(forecast(case), truth):.6f}  no-change={score(no_change, truth):.6f}  last-24h={score(recent, truth):.6f}")


if __name__ == '__main__':
    for filename in sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'practice', '*.in'))):
        run(filename)

    synthetic()
