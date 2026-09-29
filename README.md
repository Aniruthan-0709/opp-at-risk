# Opportunity Win-Propensity Model (v2: two-model design)

Predicts whether open Salesforce opportunities will close Won or Lost, and explains why,
at the opportunity, Vertical, and Vertical x Initiative level.

## Design
- **Suspect model**: opps that have never left Suspect (`MAX_STAGE_REACHED == 1`).
  Static, day-one features plus `DAYS_IN_SUSPECT`. Trained on every closed deal,
  expanded into snapshots (age 0, 30, 60, ... days while still in Suspect), weighted so
  each deal counts once.
- **Prospect+ model**: opps that reached Prospect or later at any point (`>= 2`), even if
  they have since moved back. All static features plus amount, stage, cycle length, activity.
- **Owner gate**: open opps owned by an inactive human user are not scored; they go to
  `reports/owner_reassignment_alerts.csv` and are scored on a later run once reassigned.
- **Metadata only, never features**: `INITIATIVE__C`, `IS_AUTO_CLOSED`, stage name, all IDs.
  Feature lists live in `configs/features.yaml`.

## 1. Extract (Snowflake)
Run and save as CSV:
| Query | Save as |
|---|---|
| `sql/closed_opps.sql` | `data/input/closed_opps.csv` |
| `sql/open_opps.sql` | `data/input/open_opps.csv` |

One closed-deals file feeds both training and evaluation; Python splits it by close date
using `configs/data.yaml` (training through `train_cutoff`, out-of-time eval in the eval window).

## 2. Run (from the repo root)
```powershell
python scripts/prepare_data.py                  # datasets for both models
python scripts/run_eda.py --model all           # diagnostics (removes nothing)
python scripts/select_features.py --model all   # writes configs/<model>_feature_list.json
python scripts/rank_features.py --model all     # mutual information (diagnostic)
python scripts/train_baseline.py                # the ONLY step that trains; saves models/
python scripts/evaluate.py                      # holdout + out-of-time + combined + calibration
python scripts/explain_model.py --model all     # SHAP ranking
python scripts/derive_insights.py --model all   # bin-level win-rate tables, all features
python scripts/score_open_opps.py               # gate, route, score, prediction log
python scripts/summarize_scores.py
python scripts/segment_risk_summary.py          # WHERE: Vertical x Initiative
python scripts/reasons.py --direction loss --population open --model all        # WHY at risk
python scripts/reasons.py --direction win  --population open --model all        # WHY promising
python scripts/reasons.py --direction loss --population historical --model all
python scripts/reasons.py --direction win  --population historical --model all
```
Re-run `train_baseline.py` whenever input data, features, or config change.
Each quarter: roll the windows in `configs/data.yaml` forward and retrain.

## Outputs (`reports/`)
- `evaluation_summary.json`, `<model>_calibration.csv`
- `<model>_shap_ranking.csv`, `<model>_shap_summary.png`, `<model>_insights.csv`
- `open_opp_scores.csv`, `owner_reassignment_alerts.csv`, `top_opps_priority_list.csv`, `prediction_log.csv`
- `segment_risk_summary.csv`
- `<direction>_reasons_<population>_<model>.csv`, `<direction>_reason_bins_...csv`, `<direction>_reasons_per_opp_<model>.csv`

## Verify before the first run
- `ACTIVE__C` is boolean in EDH (if text, change the filter to `ACTIVE__C = 'true'` in both SQL files)
- The Prospect Desk owner name matches `ILIKE '%PROSPECT DESK%'` in `open_opps.sql`
- `OPTYS_GOLD.OWNERID` exists (used to join `EDH.SFDC.USER_V` for the owner gate)
- The shared CTEs in the two SQL files stay identical (change one, change both)
