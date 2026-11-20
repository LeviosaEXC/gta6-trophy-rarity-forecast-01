# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/LeviosaEXC/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/LeviosaEXC/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on [make.fun](https://make.fun/) — it forecasts the earn rate of all 55 GTA VI trophies on PlayStation 5 (Exophase) at the case target, as five quantiles (5/25/50/75/95%), scored by mean pinball loss on the logit scale.

## Method

1. **Recent robust trend.** Per trophy, a Theil-Sen fit of logit(rate) on ln(1 + hours since release), fitted only on the last 24 hours of refresh states — trophy curves saturate, so the whole-history slope overshoots.
2. **Damping x0.75 toward persistence** — the measured sweet spot between chasing the trend and respecting saturation.
3. **Refresh mixture.** The predictive quantiles are a mixture: with probability P at least one refresh lands before the target, the rate drops by D logits beyond the trend (P from the observed refresh rate; D the trophy's median surprise drop at observed refreshes, prior 1.0 logit). Spread is trend uncertainty plus the house random walk (0.5·√days), floored when the slope is unidentified.

## Measured

- Practice (reconstructed refresh answers): mean pinball loss 0.77 (house baselines 1.55), coverage 0.982.
- 6 synthetic finals (independent generative model): mean loss 0.60, worst 0.71 (house curve 0.86, house persistence 1.36).

## Run

```sh
node main.ts
```

Reads `*.in` case files from the input directory (`MAKEFUN_INPUT_DIR`, default `/input`) and writes `/output/<id>.out` (`MAKEFUN_OUTPUT_DIR`, default `/output`) — one JSON object `{"trophies":[{"id","q":[5 numbers]}]}` per case. Node 22, stdlib only, no network, no randomness; two runs write identical bytes (~0.1 s per case).

## License

MIT — see [LICENSE](LICENSE).
