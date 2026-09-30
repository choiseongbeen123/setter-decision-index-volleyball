# Setter Decision-Making Index (Volleyball)

This repository contains the data pipeline, modeling code, and analysis scripts supporting a research study that develops and validates two decision-based indicators — **unpredictability** (Brier Score, derived from a CatBoost attack-zone classification model) and **diversity** (Shannon entropy of attack-option distribution) — as a process-oriented complement to the conventional set-success rate metric for evaluating professional volleyball setters.

Using play-by-play data from all 126 regular-season matches of the 2025-26 Korean V-League (KOVO), independent variables for the predictive model were derived and validated through a modified Delphi survey (N=20 experts). The analysis includes discriminant validity testing, bootstrapped mediation analysis (examining blocker count as a mediating pathway to attack success), and a composite ranking index weighted by Delphi-derived importance ratings.

## Contents

- `src/dvw_batch_pipeline.py` — custom .dvw scouting file parser and preprocessing pipeline
- `notebooks/analysis_full.ipynb` — full analysis notebook (modeling, entropy/Brier Score computation, mediation analysis, visualization)
- `data/df_final_anonymized_v2.csv` — anonymized, preprocessed dataset used in the analysis
- Model training & evaluation (Random Forest, XGBoost, CatBoost)
- Entropy / Brier Score computation and bootstrap confidence interval analysis
- Mediation analysis (blocker count as mediator)
- Delphi survey data and CVR/CV validation

## Data Preprocessing

Match data were provided as `.dvw` scouting files (DataVolley format) and processed through a custom pipeline (`src/dvw_batch_pipeline.py`):

1. **Encoding**: Korean `.dvw` files are read using `cp949` encoding.
2. **Corrupted file recovery**: Some files had player/team names stored as `?` due to encoding issues; these were automatically recovered from a Unicode backup field embedded elsewhere in the file.
3. **Team name normalization**: Sponsor name changes mid-season (e.g., a team's sponsor changed partway through the season) were mapped to a single canonical team identifier.
4. **Player ID normalization**: Player IDs with inconsistent formatting (e.g., hyphen vs. underscore) were normalized using team code + jersey number.
5. **Sequence extraction**: Reception/dig/freeball → set → attack sequences were extracted, requiring the same rally number and same team across the three touches.
6. **Feature engineering**: 34 derived columns were computed, including:
   - `touch_quality_score` (0–3 scale based on pass quality)
   - `score_diff`, `score_stage`, `is_clutch` (situational context)
   - `attacker_rt_eff` (within-match cumulative attacker efficiency, shifted to prevent data leakage)
   - `attack_zone`, `num_blockers` (parsed from raw play codes)
   - `settercall_type` (parsed from the `[3SETTERCALL]` section)

The final dataset includes 19,399 valid sequences from 9 primary setters (≥30% team setting share) across 126 regular-season matches.

## Data Availability

본 저장소에는 연구에 사용된 전처리 데이터셋만 포함되어 있습니다. 원본 스카우팅 데이터(.dvw)는 데이터 제공 방송사의 정책에 따라 공개하지 않습니다.

*This repository includes only the preprocessed dataset used in the analysis. The original scouting files (.dvw) are not released, in accordance with the data provider broadcaster's policy.*

Team names and setter identities have been anonymized (Team_A–Team_G, P1–P9) to protect player and club privacy.

## Requirements

See `requirements.txt`.

## Citation

If you use this code or methodology, please cite the associated research paper (details to be added upon publication).