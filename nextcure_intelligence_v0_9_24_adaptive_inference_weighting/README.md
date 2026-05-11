# NextCure Intelligence System — v0.9.24 Adaptive Inference Weighting

This build refines the first real external intelligence lane: ClinicalTrials.gov.

## What changed

- Removed manually seeded/pseudo patent, grant, and funding placeholders from the executive flow.
- Kept ClinicalTrials.gov as the first real live external source.
- Rewrote ClinicalTrials.gov synthesis so the four executive buckets receive interpreted reads instead of raw source-count language.
- Improved usefulness of the trial lane by emphasizing direct-lane activity, active sponsor/phase density, repeated trial-design language, ovarian ADC activity, side-channel reads, and positioning implications.
- Preserved the future database hook via `persistence_payload`.
- No Streamlit cache was added; each run performs a fresh lightweight pull.

## Audit

- Python compile check
- Pytest suite
- Direct analysis smoke test
- ZIP integrity check
