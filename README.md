# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gooonamee/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gooonamee/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

This contest forecasts the percentage of tracked PlayStation 5 players who have earned each Grand Theft Auto VI trophy at a future time, as reported by Exophase. The entry supplies five quantiles per trophy. Scoring uses pinball loss on logit-transformed percentages, so calibration and accuracy near both endpoints matter.

`main.py` reads each JSON `.in` file from `MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to `MAKEFUN_OUTPUT_DIR` (default `/output`). It uses Python 3's standard library only and makes no network requests. For example:

```sh
MAKEFUN_INPUT_DIR=../practice MAKEFUN_OUTPUT_DIR=/tmp/forecast-output python3 main.py
```

The model fits logit rates against log time since release using the full history, with a 24-hour exponential half-life, and shrinks slopes toward weighted trophy-tier priors. A second view projects both the number of earners and tracked players in log time. Its influence increases with expected player growth and decreases for short or noisy earners histories; projected earners cannot fall below their latest observed count. The final median also respects an earners floor. Uncertainty combines a random walk, fit residuals, slope uncertainty, and player-growth dilution, with a cap on the total logit spread. Sparse histories use a no-change forecast; rounded zero and 100 percent readings get small boundary adjustments. Quantiles are bounded and ordered.

Run `python3 dev_backtest.py` to compare the model with the no-change reference at several cut points in each practice history, using its last read as a stand-in for the target. It also compares both references on a synthetic case with known player and earner dynamics. These practice histories are flat, so this is a limited sanity check rather than evidence of performance on the later refresh or the contest's much longer horizon. The model and uncertainty choices are heuristic; winning or 90% coverage is not guaranteed.
