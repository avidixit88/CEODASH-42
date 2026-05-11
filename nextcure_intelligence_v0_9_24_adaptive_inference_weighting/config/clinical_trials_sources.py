"""ClinicalTrials.gov query universe for Phase 1 live intelligence.

Keep these terms narrow and company-relevant. The goal is not to mirror the
entire ClinicalTrials.gov database; the goal is to pull a compact live signal
set that can be scored and compressed into the four executive buckets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalTrialSearchSpec:
    label: str
    query: str
    strategic_lane: str
    priority: int


CLINICAL_TRIAL_SEARCH_SPECS: tuple[ClinicalTrialSearchSpec, ...] = (
    ClinicalTrialSearchSpec(
        label="CDH6 / Ovarian ADC",
        query='("CDH6" OR "cadherin 6") AND (ADC OR "antibody drug conjugate" OR ovarian)',
        strategic_lane="Direct pipeline / ovarian ADC relevance",
        priority=1,
    ),
    ClinicalTrialSearchSpec(
        label="B7-H4 ADC",
        query='("B7-H4" OR "B7H4" OR "VTCN1") AND (ADC OR "antibody drug conjugate" OR solid tumor)',
        strategic_lane="Direct target-adjacent competitive relevance",
        priority=1,
    ),
    ClinicalTrialSearchSpec(
        label="Ovarian ADC",
        query='("ovarian cancer" OR "ovarian carcinoma") AND (ADC OR "antibody drug conjugate")',
        strategic_lane="Ovarian ADC category momentum",
        priority=2,
    ),
    ClinicalTrialSearchSpec(
        label="ADC Oncology",
        query='oncology AND (ADC OR "antibody drug conjugate")',
        strategic_lane="Broad ADC category pressure / validation",
        priority=3,
    ),
    ClinicalTrialSearchSpec(
        label="Alzheimer's Side Channel",
        query='Alzheimer AND (antibody OR immunotherapy OR biomarker)',
        strategic_lane="Side-channel scientific drift",
        priority=4,
    ),
    ClinicalTrialSearchSpec(
        label="Bone Disease Side Channel",
        query='("bone disease" OR osteoporosis OR osteoarthritis) AND (antibody OR biomarker OR biologic)',
        strategic_lane="Side-channel scientific drift",
        priority=4,
    ),
)

# Intentionally low for Streamlit prototype efficiency. Each search returns the
# latest updated records; deduplication happens after ingestion.
CLINICAL_TRIALS_PAGE_SIZE = 8
CLINICAL_TRIALS_TIMEOUT_SECONDS = 7
