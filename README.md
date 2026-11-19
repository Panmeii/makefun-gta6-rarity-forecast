[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Panmeii/makefun-gta6-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Panmeii/makefun-gta6-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecaster

An entry for the make.fun bounty **[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)**:
a single-file Python program (`main.py`, standard library only) that reads one
case — the house's recorded reads of GTA VI's Exophase trophy page up to a
checkpoint — and writes, for each of the game's 55 PS5 trophies, five quantiles
(5/25/50/75/95%) of its earn rate at the case's target, scored by pinball loss
on the logit scale.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py` once per case with the case file alone
in `/input`; the program reads every `*.in` it finds and writes one `<id>.out`
per case. It is deterministic (fixed iteration counts, no randomness), runs in
a fraction of the 8-second cap, and needs no network and nothing installed.

## The model

A rate is a ratio — earners over players tracked — and the two sides fail
differently, so two models are blended:

- **Index model** (logit rates directly): `logit(rate)` is a shared progress
  index minus a per-trophy difficulty, fitted by a recency-weighted two-way
  decomposition over the distinct page states, with a per-trophy slope on the
  index shrunk toward its class mean (chapter / mission / grind / … classes
  are read off each trophy's own description). The index is extrapolated to
  the target by an ensemble of trend fits whose disagreement sets the spread.
- **Earner model** (the ratio's parts): players tracked follows an ensemble
  of growth curves (saturating exponential, log-linear, damped power law);
  each trophy's earners follow a saturating curve in log-time whose location
  and ceiling are fitted per trophy but regularised by pseudo-observations
  from a metadata prior, so a fit made on the curve's foot cannot collapse
  to a tiny ceiling. This is the model that sees dilution coming: earners can
  saturate while the tracked base keeps growing.

The blend widens by how far the two medians disagree, shrinks toward the last
observed rate when the history holds few distinct states, and applies
structural caps: the platinum and the 100%-completion trophy never exceed
the trophies they presuppose, and later story chapters never exceed earlier
ones. A trophy pinned at 100% in every state stays there.

A case whose page never changed (a stale launch-night page, like the
practice cases) takes a dilution branch instead: at launch the tracked base
broadens far faster than earners accumulate — the first refresh multiplied
the base sixfold while ordinary trophies' earners rose by a third — so the
median prices the fall and the spread prices the unknown size of the jump.

## Validation

On the kit's practice cases (whose histories are the flat launch-night page),
the program was scored against reconstructions of the refreshed page that
satisfy every fact the rules state (3,868 players tracked, 4,276 trophies
earned, every rate but the 100% one lower): mean pinball loss **0.143**
against the house persistence baseline's ~1.39 on the same reconstructions
(the baseline's loss on the true page is stated as ~1.51). On synthetic
final-shaped cases (two-plus days of stepped refreshes, 4.5-day horizon,
three different growth regimes) it scores 0.17–0.33 against the persistence
baseline's 0.31–0.56 and a replica of the house curve baseline's 0.22–0.56.
