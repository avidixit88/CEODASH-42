from __future__ import annotations

import json
from unittest.mock import patch

from engines.clinical_trials_engine import build_clinical_trials_intelligence


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "B7-H4 ADC in Solid Tumors"},
                        "statusModule": {
                            "overallStatus": "Recruiting",
                            "startDateStruct": {"date": "2025-01"},
                            "lastUpdatePostDateStruct": {"date": "2026-05-01"},
                        },
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Example Bio"}},
                        "designModule": {"phases": ["PHASE1"]},
                        "conditionsModule": {"conditions": ["Ovarian Cancer"]},
                        "armsInterventionsModule": {"interventions": [{"name": "Example ADC"}]},
                    }
                }
            ]
        }).encode("utf-8")


@patch("engines.clinical_trials_engine.urlopen", return_value=_FakeResponse())
def test_clinical_trials_live_pull_contract(_mock_urlopen):
    summary = build_clinical_trials_intelligence()
    assert summary.total_trials == 1
    assert summary.active_trials == 1
    assert summary.source_status == "live"
    assert not summary.trial_table.empty
    assert summary.persistence_payload[0]["source"] == "clinicaltrials.gov"
    assert any(signal.bucket == "new_information" for signal in summary.signals)


@patch("engines.clinical_trials_engine.urlopen", return_value=_FakeResponse())
def test_clinical_trials_executive_language_avoids_dump_terms(_mock_urlopen):
    summary = build_clinical_trials_intelligence()
    executive_text = " ".join(signal.finding for signal in summary.signals)
    banned = ["returned records", "normalized records", "configured lanes", "+ more", "the new layer compares"]
    assert not any(term in executive_text for term in banned)
    assert "ClinicalTrials.gov" not in executive_text or "ClinicalTrials.gov did not provide" in executive_text
