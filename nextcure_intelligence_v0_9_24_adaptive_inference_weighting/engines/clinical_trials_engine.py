"""Live ClinicalTrials.gov intelligence lane.

Phase 1 goal:
- Pull compact live trial signals from ClinicalTrials.gov on each analysis run.
- Score and synthesize them into the four executive buckets.
- Preserve backend hooks so the same structured study records can later be persisted
  into a database without changing the executive UI contract.

This module intentionally avoids a Streamlit cache. While the prototype is on
Streamlit Community Cloud, each run fetches fresh data with small page sizes and
short timeouts, then fails gracefully if the upstream service is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from config.clinical_trials_sources import (
    CLINICAL_TRIALS_PAGE_SIZE,
    CLINICAL_TRIALS_TIMEOUT_SECONDS,
    CLINICAL_TRIAL_SEARCH_SPECS,
    ClinicalTrialSearchSpec,
)

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


DIRECT_LANE_ORDER = ["CDH6 / Ovarian ADC", "B7-H4 ADC", "Ovarian ADC"]
SIDE_LANE_ORDER = ["Alzheimer's Side Channel", "Bone Disease Side Channel"]
LANE_DISPLAY = {
    "CDH6 / Ovarian ADC": "CDH6 / ovarian ADC",
    "B7-H4 ADC": "B7-H4 ADC",
    "Ovarian ADC": "ovarian ADC",
    "ADC Oncology": "broader oncology ADC",
    "Alzheimer's Side Channel": "Alzheimer's exploratory area",
    "Bone Disease Side Channel": "bone-disease exploratory area",
}


def _lane_label(lane: str) -> str:
    return LANE_DISPLAY.get(lane, lane.replace(" Side Channel", " exploratory area"))


def _join_labels(lanes: list[str]) -> str:
    labels = [_lane_label(lane) for lane in lanes]
    if not labels:
        return "the monitored clinical landscape"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


@dataclass(frozen=True)
class TrialRecord:
    nct_id: str
    title: str
    sponsor: str
    phase: str
    status: str
    conditions: str
    interventions: str
    start_date: str
    last_update: str
    source_query: str
    lane: str
    url: str
    enrollment: str
    primary_outcomes: str
    secondary_outcomes: str
    eligibility_criteria: str
    countries: str
    collaborators: str
    sponsor_type: str


@dataclass(frozen=True)
class ClinicalTrialSignal:
    bucket: str
    title: str
    finding: str
    value: str
    evidence: str
    priority: int


@dataclass(frozen=True)
class ClinicalTrialsSummary:
    source_status: str
    fetched_at_utc: str
    total_trials: int
    active_trials: int
    lanes_covered: list[str]
    signals: list[ClinicalTrialSignal]
    trial_table: pd.DataFrame
    persistence_payload: list[dict[str, Any]]
    source_errors: list[str]

    @property
    def new_information(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "new_information"]

    @property
    def value_interpretation(self) -> list[str]:
        return [s.value for s in self.signals if s.bucket == "value"]

    @property
    def trend_inference(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "trend"]

    @property
    def positioning_implications(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "positioning"]


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_extract_text(v) for v in value if _extract_text(v))
    if isinstance(value, dict):
        return ", ".join(_extract_text(v) for v in value.values() if _extract_text(v))
    return str(value).strip()


def _first_date(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("date") or value.get("startDate") or value.get("completionDate") or "")
    return _extract_text(value)


def _phase(protocol: dict[str, Any]) -> str:
    phases = protocol.get("designModule", {}).get("phases")
    text = _extract_text(phases)
    return text or "Not specified"


def _sponsor(protocol: dict[str, Any]) -> str:
    lead = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    return _extract_text(lead.get("name")) or "Unknown sponsor"


def _interventions(protocol: dict[str, Any]) -> str:
    arms = protocol.get("armsInterventionsModule", {}).get("interventions", []) or []
    names = []
    for item in arms:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            names.append(str(name))
    return ", ".join(dict.fromkeys(names)) or "Not specified"


def _conditions(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("conditionsModule", {}).get("conditions")) or "Not specified"


def _status(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("statusModule", {}).get("overallStatus")) or "Unknown"


def _title(protocol: dict[str, Any]) -> str:
    id_module = protocol.get("identificationModule", {})
    return _extract_text(id_module.get("briefTitle") or id_module.get("officialTitle")) or "Untitled trial"


def _nct_id(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("identificationModule", {}).get("nctId"))


def _enrollment(protocol: dict[str, Any]) -> str:
    enrollment = protocol.get("designModule", {}).get("enrollmentInfo", {})
    count = enrollment.get("count") if isinstance(enrollment, dict) else None
    if count in (None, ""):
        return "Not specified"
    return str(count)


def _outcomes(protocol: dict[str, Any], key: str) -> str:
    outcomes = protocol.get("outcomesModule", {}).get(key, []) or []
    parts: list[str] = []
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        measure = _extract_text(item.get("measure"))
        description = _extract_text(item.get("description"))
        if measure and description:
            parts.append(f"{measure}: {description}")
        elif measure:
            parts.append(measure)
        elif description:
            parts.append(description)
    return "; ".join(dict.fromkeys(parts)) or "Not specified"


def _eligibility_criteria(protocol: dict[str, Any]) -> str:
    text = _extract_text(protocol.get("eligibilityModule", {}).get("eligibilityCriteria"))
    return text or "Not specified"


def _countries(protocol: dict[str, Any]) -> str:
    locations = protocol.get("contactsLocationsModule", {}).get("locations", []) or []
    countries: list[str] = []
    for item in locations:
        if isinstance(item, dict):
            country = _extract_text(item.get("country"))
            if country and country not in countries:
                countries.append(country)
    return ", ".join(countries) or "Not specified"


def _collaborators(protocol: dict[str, Any]) -> str:
    module = protocol.get("sponsorCollaboratorsModule", {})
    collaborators = module.get("collaborators", []) or []
    names: list[str] = []
    for item in collaborators:
        if isinstance(item, dict):
            name = _extract_text(item.get("name"))
            if name and name not in names:
                names.append(name)
    return ", ".join(names) or "None listed"


def _sponsor_type_from_name(name: str) -> str:
    text = (name or "").lower()
    if any(token in text for token in ["university", "hospital", "institute", "center", "centre", "m.d. anderson", "massachusetts general", "national cancer institute", "nih"]):
        return "Academic / government"
    if any(token in text for token in ["bristol", "merck", "astrazeneca", "genmab", "gilead", "pfizer", "roche", "novartis", "eli lilly", "abbvie", "bayer", "sanofi", "johnson"]):
        return "Large pharma / established oncology"
    if any(token in text for token in ["biotech", "pharma", "therapeutics", "bioscience", "medicines", "biopharma", "bio", "limited", "ltd", "inc", "llc", "gmbh"]):
        return "Biotech / emerging sponsor"
    return "Other sponsor"


def _record_from_study(study: dict[str, Any], spec: ClinicalTrialSearchSpec) -> TrialRecord | None:
    protocol = study.get("protocolSection", {}) if isinstance(study, dict) else {}
    nct_id = _nct_id(protocol)
    if not nct_id:
        return None
    status_module = protocol.get("statusModule", {})
    return TrialRecord(
        nct_id=nct_id,
        title=_title(protocol),
        sponsor=_sponsor(protocol),
        phase=_phase(protocol),
        status=_status(protocol),
        conditions=_conditions(protocol),
        interventions=_interventions(protocol),
        start_date=_first_date(status_module.get("startDateStruct")),
        last_update=_first_date(status_module.get("lastUpdatePostDateStruct")),
        source_query=spec.query,
        lane=spec.label,
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        enrollment=_enrollment(protocol),
        primary_outcomes=_outcomes(protocol, "primaryOutcomes"),
        secondary_outcomes=_outcomes(protocol, "secondaryOutcomes"),
        eligibility_criteria=_eligibility_criteria(protocol),
        countries=_countries(protocol),
        collaborators=_collaborators(protocol),
        sponsor_type=_sponsor_type_from_name(_sponsor(protocol)),
    )


def _request_payload(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_BASE}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "NextCure-Intelligence-Prototype/0.9.13"})
    with urlopen(request, timeout=CLINICAL_TRIALS_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public API endpoint
        return json.loads(response.read().decode("utf-8"))


def _fetch_spec(spec: ClinicalTrialSearchSpec) -> tuple[list[TrialRecord], str | None]:
    base_params = {
        "query.term": spec.query,
        "pageSize": str(CLINICAL_TRIALS_PAGE_SIZE),
        "format": "json",
    }
    attempts = [
        # Preferred if accepted by the upstream API: newest/most recently updated first.
        base_params | {"sort": "LastUpdatePostDate:desc"},
        # Safe fallback if the API rejects or changes sort syntax.
        base_params,
    ]
    last_error: str | None = None
    payload: dict[str, Any] | None = None
    for params in attempts:
        try:
            payload = _request_payload(params)
            break
        except Exception as exc:  # network/API failure should never break the dashboard
            last_error = f"{type(exc).__name__}: {exc}"

    if payload is None:
        return [], f"{spec.label}: {last_error or 'unknown upstream error'}"

    records: list[TrialRecord] = []
    for study in payload.get("studies", []) or []:
        record = _record_from_study(study, spec)
        if record is not None:
            records.append(record)
    return records, None


def _is_active(status: str) -> bool:
    text = status.lower()
    return any(token in text for token in ["recruiting", "active", "enrolling", "not yet recruiting"])


def _trial_table(records: list[TrialRecord]) -> pd.DataFrame:
    columns = [
        "Lane", "NCT ID", "Sponsor", "Sponsor Type", "Phase", "Status", "Title",
        "Conditions", "Interventions", "Primary Outcomes", "Secondary Outcomes",
        "Enrollment", "Countries", "Collaborators", "Start Date", "Last Update", "URL",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "Lane": r.lane,
            "NCT ID": r.nct_id,
            "Sponsor": r.sponsor,
            "Sponsor Type": r.sponsor_type,
            "Phase": r.phase,
            "Status": r.status,
            "Title": r.title,
            "Conditions": r.conditions,
            "Interventions": r.interventions,
            "Primary Outcomes": r.primary_outcomes,
            "Secondary Outcomes": r.secondary_outcomes,
            "Enrollment": r.enrollment,
            "Countries": r.countries,
            "Collaborators": r.collaborators,
            "Start Date": r.start_date,
            "Last Update": r.last_update,
            "URL": r.url,
        }
        for r in records
    ])


def _summarize_lanes(records: list[TrialRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for r in records:
        lane = summary.setdefault(r.lane, {"count": 0, "active": 0, "sponsors": set(), "phases": set()})
        lane["count"] += 1
        lane["active"] += 1 if _is_active(r.status) else 0
        lane["sponsors"].add(r.sponsor)
        lane["phases"].add(r.phase)
    for lane in summary.values():
        lane["sponsors"] = sorted(lane["sponsors"])
        lane["phases"] = sorted(lane["phases"])
    return summary



def _lane_records(records: list[TrialRecord], lane_name: str) -> list[TrialRecord]:
    return [r for r in records if r.lane == lane_name]


def _active_records(records: list[TrialRecord]) -> list[TrialRecord]:
    return [r for r in records if _is_active(r.status)]


def _unique_values(records: list[TrialRecord], attr: str, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    values: list[str] = []
    for r in records:
        raw = getattr(r, attr, "") or ""
        for part in [x.strip() for x in str(raw).split(",") if x.strip()]:
            if part not in excluded and part not in values:
                values.append(part)
    return values


def _sponsor_phrase(records: list[TrialRecord]) -> str:
    sponsors = _unique_values(records, "sponsor", {"Unknown sponsor"})
    if not sponsors:
        return "sponsor detail not clearly listed"
    return ", ".join(sponsors)


def _sponsor_type_mix(records: list[TrialRecord]) -> str:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.sponsor_type] = counts.get(r.sponsor_type, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return "; ".join(f"{label}: {count}" for label, count in ordered) or "Sponsor type detail unavailable"


def _country_phrase(records: list[TrialRecord]) -> str:
    countries = _unique_values(records, "countries", {"Not specified"})
    if not countries:
        return "country/site geography not consistently listed"
    return ", ".join(countries)


def _enrollment_read(records: list[TrialRecord]) -> str:
    values: list[int] = []
    for r in records:
        try:
            values.append(int(float(str(r.enrollment).replace(",", ""))))
        except Exception:
            pass
    if not values:
        return "enrollment size was not consistently available across the surfaced records"
    return f"listed enrollment sizes range from {min(values):,} to {max(values):,}, with median-style midpoint around {sorted(values)[len(values)//2]:,}"


def _trial_text(r: TrialRecord) -> str:
    return " ".join([
        r.title, r.conditions, r.interventions, r.primary_outcomes, r.secondary_outcomes,
        r.eligibility_criteria, r.countries, r.collaborators, r.sponsor, r.phase, r.status,
    ]).lower()


def _keyword_presence(records: list[TrialRecord], terms: list[str]) -> list[TrialRecord]:
    return [r for r in records if any(term.lower() in _trial_text(r) for term in terms)]


def _differentiation_reads(records: list[TrialRecord]) -> list[str]:
    if not records:
        return []
    reads: list[str] = []
    biomarker = _keyword_presence(records, ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress"])
    prior_therapy = _keyword_presence(records, ["platinum", "recurrent", "refractory", "resistant", "prior therapy", "previous therapy", "relapsed"])
    combo = _keyword_presence(records, ["combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", "plus"])
    safety = _keyword_presence(records, ["safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "adverse event"])
    endpoints = _keyword_presence(records, ["overall response", "objective response", "progression-free", "duration of response", "dose limiting", "recommended phase 2"])

    if biomarker:
        reads.append(f"Patient-selection signal: {len(biomarker)} surfaced oncology record(s) contain biomarker, expression, positivity, or selection language. This is the part to watch because precision of patient selection is where a CDH6 story can become more than generic ADC exposure.")
    else:
        reads.append("Patient-selection signal: the surfaced oncology records did not consistently expose biomarker-selection language. That makes explicit CDH6 rationale and patient-selection clarity a potential messaging edge if supported by company data.")
    if prior_therapy:
        reads.append(f"Treatment-context signal: {len(prior_therapy)} record(s) reference recurrent, resistant, refractory, platinum, or prior-therapy language. That helps identify whether competitors are fighting in late-line salvage settings versus trying to move into cleaner earlier-line narratives.")
    if combo:
        reads.append(f"Combination signal: {len(combo)} record(s) include combination or partner-therapy language. If peers are leaning on combinations, a cleaner single-agent or better-tolerated positioning can become strategically important if the data support it.")
    if safety or endpoints:
        reads.append(f"Endpoint/safety signal: {len(set([r.nct_id for r in safety + endpoints]))} record(s) expose safety, tolerability, response, PFS, DOR, dose-limiting, or RP2D-style endpoint language. That is where the battlefield shifts from 'who has an ADC' to 'who can prove usable clinical benefit.'")
    return reads


def _phase_phrase(phases: list[str] | set[str]) -> str:
    clean = [p for p in sorted(phases) if p and p != "Not specified"]
    return ", ".join(clean[:4]) if clean else "phase detail not consistently specified"


def _clinical_activity_phrase(data: dict[str, Any], lane_name: str) -> str:
    active = int(data.get("active", 0) or 0)
    total = int(data.get("count", 0) or 0)
    label = _lane_label(lane_name)
    if total <= 0:
        return f"{label} did not contribute enough usable clinical signal to elevate this run"
    ratio = active / total
    if active >= 6 and ratio >= 0.75:
        return f"{label} is showing broad active clinical presence in this run"
    if active >= 3:
        return f"{label} remains meaningfully active in the current clinical sample"
    if active > 0:
        return f"{label} is present, but the signal is narrower than the larger monitored lanes"
    return f"{label} appeared in the clinical landscape, but active development was limited in this run"


def _phase_stage_phrase(phases: list[str] | set[str]) -> str:
    clean = {str(p).upper().replace(" ", "") for p in phases if p and p != "Not specified"}
    if any("PHASE3" in p for p in clean):
        return "the landscape includes late-stage programs, so the field is no longer purely exploratory"
    if any("PHASE2" in p for p in clean):
        return "mid-stage studies are present, which suggests the space is moving beyond first-in-human exploration"
    if any("PHASE1" in p for p in clean):
        return "the activity is still mostly early-stage, leaving room for differentiated clinical positioning"
    return "phase detail is inconsistent, so maturity should be interpreted cautiously"


def _maturity_label(phases: list[str] | set[str]) -> str:
    clean = {str(p).upper().replace(" ", "") for p in phases if p and p != "Not specified"}
    if any("PHASE3" in p for p in clean):
        return "late-stage anchor present"
    if any("PHASE2" in p for p in clean):
        return "mid-stage validation emerging"
    if any("PHASE1" in p for p in clean):
        return "early clinical field"
    return "maturity unclear"


def _theme_phrase(theme: str) -> str:
    mapping = {
        "biomarker / patient-selection language": "patient-selection / biomarker language",
        "combination strategy": "combination strategy",
        "ovarian / gynecologic focus": "ovarian / gynecologic focus",
        "antibody / ADC modality language": "antibody / ADC modality language",
    }
    return mapping.get(theme, theme)


def _theme_hits(records: list[TrialRecord]) -> dict[str, int]:
    theme_terms = {
        "biomarker / patient-selection language": ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular"],
        "combination strategy": ["combination", "combined", "plus", "with pembrolizumab", "with chemotherapy", "with paclitaxel"],
        "ovarian / gynecologic focus": ["ovarian", "fallopian", "peritoneal", "gynecologic", "gynaecologic"],
        "antibody / ADC modality language": ["adc", "antibody drug", "antibody-drug", "antibody", "conjugate"],
    }
    counts = {theme: 0 for theme in theme_terms}
    for r in records:
        haystack = " ".join([r.title, r.conditions, r.interventions]).lower()
        for theme, terms in theme_terms.items():
            if any(term in haystack for term in terms):
                counts[theme] += 1
    return {theme: count for theme, count in counts.items() if count > 0}


def _top_theme_sentence(records: list[TrialRecord], scope: str) -> tuple[str, str] | None:
    hits = _theme_hits(records)
    if not hits:
        return None
    ranked = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    top_theme, _ = ranked[0]
    other = [_theme_phrase(name) for name, _count in ranked[1:3]]
    detail = f"; secondary themes include {', '.join(other)}" if other else ""
    return (
        f"Across {scope}, the strongest repeated trial-design language is {_theme_phrase(top_theme)}{detail}.",
        "This matters because repeated protocol language reveals what sponsors are choosing to emphasize clinically, which is more useful than simply knowing that studies exist.",
    )


def _fragmentation_read(records: list[TrialRecord], lane_names: list[str]) -> str:
    lane_count = len(lane_names)
    sponsor_count = len({r.sponsor for r in records if r.sponsor and r.sponsor != "Unknown sponsor"})
    phases = {r.phase for r in records if r.phase and r.phase != "Not specified"}
    maturity = _phase_stage_phrase(phases)
    if sponsor_count >= 5 and lane_count >= 2:
        return (
            f"The direct oncology battlefield is active but fragmented across multiple sponsors; {maturity}. "
            "That is not automatically good or bad. The edge is to make the CDH6 / ovarian ADC story sharper than the category itself: why this target, why this patient population, and why the approach can stand out inside a crowded ADC conversation."
        )
    if sponsor_count >= 2:
        return (
            f"The direct oncology battlefield has multiple active sponsors but is not overwhelmingly broad in this sample; {maturity}. "
            "The edge is focus: use the clinical landscape to show that the category is alive while keeping the differentiation narrative specific to NextCure's own program rather than generic ADC momentum."
        )
    return (
        f"The direct oncology signal is present but narrow in this run; {maturity}. "
        "The edge is selectivity: avoid overstating category heat and instead emphasize the most defensible clinical angle supported by NextCure's own data and upcoming catalysts."
    )


def _latest_update_sentence(records: list[TrialRecord]) -> str | None:
    latest = sorted(_active_records(records), key=lambda r: r.last_update or "", reverse=True)[:4]
    if not latest:
        return None
    pieces = []
    for r in latest:
        phase = f", {r.phase}" if r.phase and r.phase != "Not specified" else ""
        pieces.append(f"{r.sponsor} — {_lane_label(r.lane)}{phase} [{r.nct_id}]")
    return "Recent clinical-record movement worth knowing: " + "; ".join(pieces) + "."


def _phase_mix(records: list[TrialRecord]) -> str:
    order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
    counts: dict[str, int] = {}
    for r in records:
        phase = (r.phase or "Not specified").upper().replace(" ", "")
        counts[phase] = counts.get(phase, 0) + 1
    parts = []
    for key in order:
        if key in counts:
            parts.append(f"{key.replace('_', ' ')}: {counts[key]}")
    for key, val in sorted(counts.items()):
        if key not in order and key != "NOTSPECIFIED":
            parts.append(f"{key}: {val}")
    if counts.get("NOTSPECIFIED"):
        parts.append(f"phase not specified: {counts['NOTSPECIFIED']}")
    return "; ".join(parts) or "phase mix not available"


def _phase_anchor_sponsors(records: list[TrialRecord], phase_token: str = "PHASE3") -> list[str]:
    names: list[str] = []
    for r in records:
        if phase_token in (r.phase or "").upper().replace(" ", "") and r.sponsor not in names:
            names.append(r.sponsor)
    return names


def _lane_profile_sentence(records: list[TrialRecord], lane: str) -> str:
    lane_recs = _lane_records(records, lane)
    if not lane_recs:
        return f"{_lane_label(lane)}: no usable live clinical profile in this run."
    anchors = _phase_anchor_sponsors(lane_recs, "PHASE3")
    anchor_phrase = f" Late-stage anchor sponsor(s): {', '.join(anchors)}." if anchors else " No Phase 3 anchor was surfaced in this lane in this run."
    return (
        f"{_lane_label(lane)} profile — sponsors: {_sponsor_phrase(lane_recs)}. "
        f"Phase mix: {_phase_mix(lane_recs)}. "
        f"Sponsor mix: {_sponsor_type_mix(lane_recs)}. "
        f"Geography: {_country_phrase(lane_recs)}. "
        f"Enrollment signal: {_enrollment_read(lane_recs)}."
        f"{anchor_phrase}"
    )


def _battlefield_edge_sentence(ovarian_records: list[TrialRecord], b7h4_records: list[TrialRecord]) -> str:
    ovarian_anchors = _phase_anchor_sponsors(ovarian_records, "PHASE3")
    sponsor_mix = _sponsor_type_mix(ovarian_records) if ovarian_records else "Sponsor type detail unavailable"
    if ovarian_anchors:
        return (
            f"Ovarian ADC is not an empty or purely early-stage field; Phase 3 anchor sponsor(s) surfaced: {', '.join(ovarian_anchors)}. "
            f"The useful edge is not claiming first-mover category novelty. It is sharper CDH6-specific positioning inside a field that still shows sponsor fragmentation ({sponsor_mix}). "
            "That gives leadership a better board/investor framing: the category is validated enough to matter, but not so consolidated that a clear CDH6 rationale, patient-selection story, and catalyst path cannot stand out."
        )
    return (
        f"Ovarian ADC activity is visible but the current live pull did not surface a Phase 3 anchor inside the ovarian-linked set. Sponsor mix: {sponsor_mix}. "
        "That creates a different edge: the field is active enough to validate attention, while the clinical narrative may still be shaped by whoever can communicate the cleanest target rationale and patient-selection logic."
    )



def _sponsor_segments(records: list[TrialRecord]) -> str:
    buckets: dict[str, list[str]] = {
        "large pharma / established oncology": [],
        "biotech / emerging sponsor": [],
        "academic / government": [],
        "other sponsor": [],
    }
    for r in records:
        name = r.sponsor.strip() or "Unknown sponsor"
        if name == "Unknown sponsor":
            continue
        key = r.sponsor_type.lower()
        if "large pharma" in key:
            bucket = "large pharma / established oncology"
        elif "biotech" in key:
            bucket = "biotech / emerging sponsor"
        elif "academic" in key:
            bucket = "academic / government"
        else:
            bucket = "other sponsor"
        if name not in buckets[bucket]:
            buckets[bucket].append(name)
    parts = []
    for label, names in buckets.items():
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) or "sponsor segmentation was not available"


def _endpoint_strategy_read(records: list[TrialRecord]) -> str:
    if not records:
        return "Endpoint strategy could not be assessed from the surfaced records."
    categories = {
        "response and tumor-control endpoints": ["objective response", "overall response", "orr", "response rate", "duration of response", "dor", "disease control"],
        "time-to-event endpoints": ["progression-free", "pfs", "overall survival", "os", "time to"],
        "dose/safety endpoints": ["safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "rp2d", "adverse event"],
    }
    hits: dict[str, list[str]] = {k: [] for k in categories}
    for r in records:
        haystack = " ".join([r.primary_outcomes, r.secondary_outcomes, r.title]).lower()
        for label, terms in categories.items():
            if any(term in haystack for term in terms) and r.nct_id not in hits[label]:
                hits[label].append(r.nct_id)
    ordered = [(label, ids) for label, ids in hits.items() if ids]
    if not ordered:
        return "Endpoint strategy is not consistently exposed in the surfaced records, so trial maturity should be judged more from phase, sponsor type, and enrollment design."
    phrases = [f"{label} in {len(ids)} study/studies" for label, ids in ordered]
    leader = max(ordered, key=lambda item: len(item[1]))[0]
    return f"Endpoint emphasis: {', '.join(phrases)}. The most visible endpoint posture is {leader}, which helps show whether competitors are optimizing for early activity signals, durability, or dose usability."


def _patient_selection_read(records: list[TrialRecord]) -> str:
    biomarker = _keyword_presence(records, ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress", "cdh6", "b7-h4", "b7h4"])
    prior = _keyword_presence(records, ["platinum", "recurrent", "refractory", "resistant", "relapsed", "prior therapy", "previous therapy", "after", "progressed"])
    if biomarker and prior:
        return f"Patient-selection read: {len({r.nct_id for r in biomarker})} study/studies expose biomarker, target-expression, or selection language and {len({r.nct_id for r in prior})} study/studies expose recurrent, refractory, resistant, platinum, relapsed, or prior-therapy language. The useful edge is seeing whether competitors are defining who should receive the ADC, not just whether they have an ADC."
    if biomarker:
        return f"Patient-selection read: {len({r.nct_id for r in biomarker})} study/studies expose biomarker, target-expression, or selection language. This is where a CDH6 story can become sharper than generic ovarian ADC exposure if the target rationale is communicated clearly."
    if prior:
        return f"Treatment-context read: {len({r.nct_id for r in prior})} study/studies expose recurrent, refractory, resistant, platinum, relapsed, or prior-therapy language. This helps separate late-line salvage positioning from broader ovarian oncology ambition."
    return "Patient-selection read: biomarker and treatment-line language were not strongly visible in the surfaced records. That absence itself matters because a clearer CDH6 patient-selection rationale can become more distinctive if supported by NextCure's own evidence."


def _combination_read(records: list[TrialRecord]) -> str:
    combo = _keyword_presence(records, ["combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", "plus", "with"])
    if not combo:
        return "Combination read: the surfaced records do not strongly point to combination-heavy positioning. That keeps attention on target rationale, monotherapy activity, tolerability, and patient selection rather than assuming combinations are the main battlefield."
    return f"Combination read: {len({r.nct_id for r in combo})} study/studies contain combination or partner-therapy language. If competitors lean on combinations, the strategic question becomes whether a program can show cleaner single-agent contribution, better tolerability, or a clearer role in the treatment sequence."


def _geography_depth_read(records: list[TrialRecord]) -> str:
    countries = _unique_values(records, "countries", {"Not specified"})
    if not countries:
        return "Geography read: trial-site country detail was not consistently visible."
    regions = []
    lower = {c.lower() for c in countries}
    if "united states" in lower:
        regions.append("U.S.")
    if any(c in lower for c in {"china", "hong kong", "taiwan", "korea, republic of", "japan", "singapore"}):
        regions.append("Asia-Pacific")
    if any(c in lower for c in {"france", "germany", "spain", "italy", "united kingdom", "netherlands", "belgium", "poland"}):
        regions.append("Europe")
    region_phrase = f" Region signal: {', '.join(regions)}." if regions else ""
    return f"Geography read: surfaced countries include {', '.join(countries)}.{region_phrase} Broad geography can indicate operational seriousness; narrow geography can indicate earlier or more localized development."


def _enrollment_depth_read(records: list[TrialRecord]) -> str:
    values: list[tuple[int, TrialRecord]] = []
    for r in records:
        try:
            values.append((int(float(str(r.enrollment).replace(',', ''))), r))
        except Exception:
            pass
    if not values:
        return "Enrollment read: enrollment size was not consistently available, so confidence should lean more on phase, sponsor type, and protocol design."
    values.sort(key=lambda x: x[0], reverse=True)
    top_n = values[:3]
    top_text = "; ".join(f"{r.sponsor} {r.phase} {n:,} planned/actual participants" for n, r in top_n)
    return f"Enrollment read: the largest surfaced enrollment signals are {top_text}. Larger enrollment can indicate seriousness or later-stage breadth; smaller enrollment often points to exploratory signal-finding."


def _board_ammunition_read(records: list[TrialRecord]) -> str:
    if not records:
        return "No board-level clinical ammunition was available from this source run."
    sponsors = _sponsor_segments(records)
    endpoint = _endpoint_strategy_read(records)
    selection = _patient_selection_read(records)
    combo = _combination_read(records)
    return (
        "Board/investor ammunition from ClinicalTrials.gov: "
        f"1) sponsor map — {sponsors}. "
        f"2) {endpoint} "
        f"3) {selection} "
        f"4) {combo}"
    )


def _edge_read(records: list[TrialRecord], lane_name: str) -> str:
    lane_recs = _lane_records(records, lane_name)
    if not lane_recs:
        return f"{_lane_label(lane_name)}: no edge read available from this run."
    phase3 = _phase_anchor_sponsors(lane_recs, "PHASE3")
    phase2 = _phase_anchor_sponsors(lane_recs, "PHASE2")
    sponsors = _unique_values(lane_recs, "sponsor", {"Unknown sponsor"})
    sponsor_count = len(sponsors)
    if phase3 and sponsor_count >= 4:
        setup = f"{_lane_label(lane_name)} has late-stage anchor sponsor(s) ({', '.join(phase3)}) plus a broader sponsor set ({', '.join(sponsors)})."
        edge = "That points to a validated but contested field: the edge is not novelty; the edge is whether NextCure can make CDH6 feel more precise, more biologically justified, and better timed than broad ADC category exposure."
    elif phase2 or phase3:
        anchors = phase3 or phase2
        setup = f"{_lane_label(lane_name)} has visible mid/late clinical anchors ({', '.join(anchors)}) but does not look fully consolidated in this pull."
        edge = "That creates room for a differentiated clinical narrative if NextCure can clearly explain target selection, patient fit, and evidence path."
    else:
        setup = f"{_lane_label(lane_name)} appears active but mainly earlier-stage in this pull, with sponsors including {', '.join(sponsors)}."
        edge = "That is a shapeable battlefield: the edge is establishing clinical credibility and narrative specificity before the space becomes more crowded or later-stage."
    return f"{setup} {edge}"


def _edge_read_for_records(label: str, lane_recs: list[TrialRecord]) -> str:
    if not lane_recs:
        return f"{label}: no edge read available from this run."
    phase3 = _phase_anchor_sponsors(lane_recs, "PHASE3")
    phase2 = _phase_anchor_sponsors(lane_recs, "PHASE2")
    sponsors = _unique_values(lane_recs, "sponsor", {"Unknown sponsor"})
    sponsor_count = len(sponsors)
    if phase3 and sponsor_count >= 4:
        setup = f"{label} has late-stage anchor sponsor(s) ({', '.join(phase3)}) plus a broader sponsor set ({', '.join(sponsors)})."
        edge = "That points to a validated but contested field: the edge is not novelty; the edge is whether NextCure can make CDH6 feel more precise, more biologically justified, and better timed than broad ADC category exposure."
    elif phase2 or phase3:
        anchors = phase3 or phase2
        setup = f"{label} has visible mid/late clinical anchors ({', '.join(anchors)}) but does not look fully consolidated in this pull."
        edge = "That creates room for a differentiated clinical narrative if NextCure can clearly explain target selection, patient fit, and evidence path."
    else:
        setup = f"{label} appears active but mainly earlier-stage in this pull, with sponsors including {', '.join(sponsors)}."
        edge = "That is a shapeable battlefield: the edge is establishing clinical credibility and narrative specificity before the space becomes more crowded or later-stage."
    return f"{setup} {edge}"



# --- v0.9.24: adaptive inference-weighted clinical edge intelligence helpers ---


@dataclass(frozen=True)
class ClinicalLaneSignature:
    """Adaptive strategic state derived from combinations of ClinicalTrials.gov fields.

    The signature exists to prevent the Executive Summary from becoming a trial
    database narration. It compresses raw fields into a confidence-weighted
    battlefield read that can change as the live pull changes.
    """

    label: str
    sponsors: list[str]
    phase3_sponsors: list[str]
    phase2_sponsors: list[str]
    phase1_sponsors: list[str]
    sponsor_types: dict[str, list[str]]
    patient_selection_strength: str
    combination_strength: str
    safety_strength: str
    response_strength: str
    geography_strength: str
    enrollment_strength: str
    strategic_state: str
    narrative_owner: str
    edge_thesis: str
    proof_burden: str
    investor_question: str
    signature_codes: list[str]
    priority_score: int
    confidence: str
    confidence_reason: str


def _sponsors_for_phase(records: list[TrialRecord], token: str) -> list[str]:
    token = token.upper().replace(" ", "")
    names: list[str] = []
    for r in records:
        phase = (r.phase or "").upper().replace(" ", "")
        if token in phase and r.sponsor not in names and r.sponsor != "Unknown sponsor":
            names.append(r.sponsor)
    return names


def _unique_sponsors(records: list[TrialRecord]) -> list[str]:
    return _unique_values(records, "sponsor", {"Unknown sponsor"})


def _sponsor_type_buckets(records: list[TrialRecord]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "established oncology/pharma": [],
        "emerging/specialist developers": [],
        "academic/government": [],
        "other named sponsors": [],
    }
    for r in records:
        name = (r.sponsor or "").strip()
        if not name or name == "Unknown sponsor":
            continue
        sponsor_type = r.sponsor_type.lower()
        if "large pharma" in sponsor_type:
            key = "established oncology/pharma"
        elif "biotech" in sponsor_type:
            key = "emerging/specialist developers"
        elif "academic" in sponsor_type:
            key = "academic/government"
        else:
            key = "other named sponsors"
        if name not in buckets[key]:
            buckets[key].append(name)
    return {k: v for k, v in buckets.items() if v}


def _signal_records(records: list[TrialRecord], terms: list[str]) -> list[TrialRecord]:
    return _keyword_presence(records, terms)


def _signal_strength(records: list[TrialRecord], terms: list[str]) -> tuple[str, list[TrialRecord]]:
    hits = _signal_records(records, terms)
    unique = {r.nct_id for r in hits}
    if not records:
        return "not assessable", []
    ratio = len(unique) / max(1, len({r.nct_id for r in records}))
    if len(unique) >= 3 and ratio >= 0.45:
        return "prominent", hits
    if unique:
        return "present", hits
    return "not prominent", []


def _geography_strength(records: list[TrialRecord]) -> str:
    countries = _unique_values(records, "countries", {"Not specified"})
    if len(countries) >= 12:
        return "global / operationally broad"
    if len(countries) >= 4:
        return "multi-region"
    if countries:
        return "localized / narrower"
    return "not clearly exposed"


def _enrollment_strength(records: list[TrialRecord]) -> str:
    values: list[int] = []
    for r in records:
        try:
            values.append(int(float(str(r.enrollment).replace(",", ""))))
        except Exception:
            pass
    if not values:
        return "not clearly exposed"
    if max(values) >= 500:
        return "large-scale enrollment visible"
    if max(values) >= 150:
        return "mid-sized enrollment visible"
    return "small / signal-finding enrollment"


def _phase_architecture_short_from_sig(sig: ClinicalLaneSignature) -> str:
    parts: list[str] = []
    if sig.phase3_sponsors:
        parts.append(f"Phase 3 anchor: {', '.join(sig.phase3_sponsors)}")
    if sig.phase2_sponsors:
        parts.append(f"Phase 2/mid-stage: {', '.join(sig.phase2_sponsors)}")
    if sig.phase1_sponsors:
        parts.append(f"early-stage: {', '.join(sig.phase1_sponsors)}")
    return "; ".join(parts) if parts else "phase architecture not clearly exposed"


def _sponsor_segment_text(sig: ClinicalLaneSignature) -> str:
    if not sig.sponsor_types:
        return "sponsor segmentation unavailable"
    return "; ".join(f"{label}: {', '.join(names)}" for label, names in sig.sponsor_types.items())


def _narrative_owner_label(records: list[TrialRecord]) -> str:
    sponsors = _unique_sponsors(records)
    phase3 = _sponsors_for_phase(records, "PHASE3")
    established = _sponsor_type_buckets(records).get("established oncology/pharma", [])
    if len(phase3) >= 2:
        return "late-stage ownership is contested rather than controlled by one sponsor"
    if len(phase3) == 1 and len(sponsors) >= 4:
        return f"{phase3[0]} is the late-stage reference point, but the lane is not fully owned"
    if len(phase3) == 1:
        return f"{phase3[0]} is the clearest late-stage reference point"
    if len(established) >= 2 and len(sponsors) >= 5:
        return "large oncology sponsors are present, but no single late-stage owner surfaced"
    if len(sponsors) >= 4:
        return "multiple sponsors are active, but no clear narrative owner surfaced"
    return "narrative ownership remains early or unclear"


def _confidence_from_codes(codes: list[str], records: list[TrialRecord], label: str) -> tuple[str, str]:
    evidence_count = len({r.nct_id for r in records})
    if not records:
        return "low", "no usable lane records were surfaced"
    high_codes = {"late_stage_anchor", "sponsor_fragmentation", "patient_selection_visible"}
    if evidence_count >= 4 and len(high_codes.intersection(codes)) >= 2:
        return "high", "phase, sponsor, and protocol signals point in the same direction"
    if evidence_count >= 2 and codes:
        return "moderate", "the signal is supported by multiple records but should still be monitored for confirmation"
    return "low", "the read is preliminary and should not carry the executive conclusion alone"


def _derive_lane_signature(label: str, records: list[TrialRecord]) -> ClinicalLaneSignature | None:
    if not records:
        return None

    sponsors = _unique_sponsors(records)
    phase3 = _sponsors_for_phase(records, "PHASE3")
    phase2 = _sponsors_for_phase(records, "PHASE2")
    phase1 = _sponsors_for_phase(records, "PHASE1")
    sponsor_types = _sponsor_type_buckets(records)
    patient_strength, _ = _signal_strength(records, [
        "biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular", "ihc", "overexpress", "cdh6", "b7-h4", "b7h4"
    ])
    combo_strength, _ = _signal_strength(records, [
        "combination", "combined", "pembrolizumab", "nivolumab", "paclitaxel", "carboplatin", "bevacizumab", "chemotherapy", " plus ", " with "
    ])
    safety_strength, _ = _signal_strength(records, [
        "safety", "tolerability", "dose limiting", "maximum tolerated", "recommended phase 2", "rp2d", "adverse event"
    ])
    response_strength, _ = _signal_strength(records, [
        "objective response", "overall response", "orr", "duration of response", "dor", "progression-free", "pfs", "overall survival"
    ])
    geography = _geography_strength(records)
    enrollment = _enrollment_strength(records)
    owner = _narrative_owner_label(records)

    sponsor_count = len(sponsors)
    phase3_count = len(phase3)
    codes: list[str] = []
    score = 0

    if phase3_count:
        codes.append("late_stage_anchor")
        score += 3
    if sponsor_count >= 4:
        codes.append("sponsor_fragmentation")
        score += 2
    if patient_strength in {"prominent", "present"}:
        codes.append("patient_selection_visible")
        score += 2
    if combo_strength in {"prominent", "present"}:
        codes.append("combination_context_visible")
        score += 1
    if safety_strength in {"prominent", "present"}:
        codes.append("usability_burden_visible")
        score += 1
    if response_strength in {"prominent", "present"}:
        codes.append("efficacy_burden_visible")
        score += 1
    if geography in {"multi-region", "global / operationally broad"}:
        codes.append("operational_breadth")
        score += 1

    label_lower = label.lower()
    if label.startswith("CDH6"):
        if phase3_count == 1 and sponsor_count >= 4:
            state = "validated but still shapeable"
            edge = "A single late-stage anchor validates CDH6, while the broader sponsor map still leaves room for target-specific narrative ownership."
        elif phase3_count >= 2:
            state = "validated and increasingly contested"
            edge = "CDH6 would need to compete on patient definition, clinical usability, and evidence quality rather than category participation."
        elif sponsor_count >= 4:
            state = "active but not late-stage-owned"
            edge = "CDH6 remains narrative-open if the target rationale can be made clearer than the broader ovarian ADC field."
        else:
            state = "early and narrative-open"
            edge = "CDH6 is still early enough that credible biology and patient definition can help define what the lane means."
    elif "b7-h4" in label_lower:
        state = "adjacent attention comparator"
        edge = "B7-H4 is useful gynecologic-oncology read-through, but it should not be blended into the CDH6 thesis."
    elif "ovarian adc" in label_lower:
        state = "active category context"
        edge = "Broad ovarian ADC activity validates investor attention, but it also creates noise that CDH6 positioning must cut through."
    elif "adc oncology" in label_lower:
        state = "category weather"
        edge = "Broad ADC oncology activity can explain modality appetite, but it should not replace the target-specific answer."
    else:
        state = "exploratory context"
        edge = "This lane is useful as optionality context, not as the core oncology positioning thesis."

    proof_parts: list[str] = []
    if patient_strength == "prominent":
        proof_parts.append("patient-selection logic is prominent, so target-expression credibility becomes a differentiator")
    elif patient_strength == "present":
        proof_parts.append("patient-selection logic is present, so the target rationale needs to be explicit")
    if combo_strength == "prominent":
        proof_parts.append("combination language is prominent, so single-agent contribution, tolerability, or sequencing can become a counter-position")
    elif combo_strength == "present":
        proof_parts.append("combination language is present, so treatment-sequence clarity matters")
    if safety_strength in {"prominent", "present"}:
        proof_parts.append("dose, safety, and tolerability remain visible proof burdens")
    if response_strength in {"prominent", "present"}:
        proof_parts.append("response, durability, and time-to-event endpoints keep the burden on usable benefit")
    proof = "; ".join(proof_parts) if proof_parts else "the proof burden remains target rationale, patient fit, response durability, and clinical timing"

    if label.startswith("CDH6") and phase3_count == 1:
        question = "How can CDH6 use Daiichi's late-stage presence as validation without conceding the whole narrative?"
    elif label.startswith("CDH6") and phase3_count >= 2:
        question = "How does the company avoid sounding late to an increasingly validated CDH6 field?"
    elif label.startswith("CDH6"):
        question = "Can the company define CDH6 before the field becomes more crowded or later-stage?"
    elif "B7-H4" in label:
        question = "How should B7-H4 be used as attention read-through without confusing it with CDH6?"
    elif "ovarian ADC" in label_lower:
        question = "How much of ovarian ADC activity is useful context versus category noise?"
    else:
        question = "Does this activity change the core thesis, or is it only context?"

    confidence, confidence_reason = _confidence_from_codes(codes, records, label)

    return ClinicalLaneSignature(
        label=label,
        sponsors=sponsors,
        phase3_sponsors=phase3,
        phase2_sponsors=phase2,
        phase1_sponsors=phase1,
        sponsor_types=sponsor_types,
        patient_selection_strength=patient_strength,
        combination_strength=combo_strength,
        safety_strength=safety_strength,
        response_strength=response_strength,
        geography_strength=geography,
        enrollment_strength=enrollment,
        strategic_state=state,
        narrative_owner=owner,
        edge_thesis=edge,
        proof_burden=proof,
        investor_question=question,
        signature_codes=codes,
        priority_score=score,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


def _signature_evidence(sig: ClinicalLaneSignature) -> str:
    return (
        f"Sponsors: {', '.join(sig.sponsors) if sig.sponsors else 'none surfaced'}. "
        f"{_phase_architecture_short_from_sig(sig)}. "
        f"Sponsor segmentation: {_sponsor_segment_text(sig)}. "
        f"Signals: {', '.join(sig.signature_codes) if sig.signature_codes else 'no high-conviction signature'}; "
        f"patient selection: {sig.patient_selection_strength}; combinations: {sig.combination_strength}; "
        f"safety/tolerability: {sig.safety_strength}; response/durability: {sig.response_strength}; "
        f"geography: {sig.geography_strength}; enrollment: {sig.enrollment_strength}; confidence: {sig.confidence}."
    )


def _evidence_tag(sig: ClinicalLaneSignature) -> str:
    return f"Confidence: {sig.confidence}. Evidence basis: {sig.confidence_reason}."


def _clinical_leverage_thesis(cdh6_sig: ClinicalLaneSignature | None, ovarian_sig: ClinicalLaneSignature | None, b7h4_sig: ClinicalLaneSignature | None) -> str:
    if cdh6_sig:
        context_parts: list[str] = []
        if ovarian_sig:
            context_parts.append(f"broad ovarian ADC is {ovarian_sig.strategic_state}")
        if b7h4_sig:
            context_parts.append("B7-H4 is an adjacent comparator, not the same target story")
        context = f" Context: {'; '.join(context_parts)}." if context_parts else ""
        return (
            f"Highest-priority clinical signal: CDH6 is {cdh6_sig.strategic_state}. "
            f"{cdh6_sig.narrative_owner}. {cdh6_sig.edge_thesis} {context} "
            f"{_evidence_tag(cdh6_sig)}"
        )
    if ovarian_sig:
        return (
            f"Highest-priority clinical signal: broad ovarian ADC is {ovarian_sig.strategic_state}. "
            f"{ovarian_sig.narrative_owner}. {ovarian_sig.edge_thesis} {_evidence_tag(ovarian_sig)}"
        )
    if b7h4_sig:
        return f"Highest-priority clinical signal: B7-H4 is {b7h4_sig.strategic_state}. {b7h4_sig.edge_thesis} {_evidence_tag(b7h4_sig)}"
    return "The live pull did not create a high-conviction direct ovarian/CDH6 clinical edge read this run."


def _recent_movement_read(records: list[TrialRecord]) -> str | None:
    direct_lanes = {"CDH6 / Ovarian ADC", "B7-H4 ADC", "Ovarian ADC"}
    direct_latest = sorted(_active_records([r for r in records if r.lane in direct_lanes]), key=lambda r: r.last_update or "", reverse=True)[:3]
    if not direct_latest:
        return None
    pieces = [f"{r.sponsor} ({_lane_label(r.lane)} {r.phase}; {r.nct_id})" for r in direct_latest]
    return "Recent direct oncology movement to have ready: " + "; ".join(pieces) + "."


def _investor_ammunition_read(cdh6_sig: ClinicalLaneSignature | None, ovarian_sig: ClinicalLaneSignature | None, b7h4_sig: ClinicalLaneSignature | None) -> str:
    parts: list[str] = []
    if cdh6_sig:
        parts.append(
            f"CDH6 answer: {cdh6_sig.strategic_state}; {cdh6_sig.narrative_owner}. "
            f"Named CDH6 sponsors: {', '.join(cdh6_sig.sponsors)}. {_phase_architecture_short_from_sig(cdh6_sig)}. "
            f"Board question: {cdh6_sig.investor_question}"
        )
    if ovarian_sig:
        parts.append(
            f"Ovarian ADC context: {ovarian_sig.strategic_state}; {ovarian_sig.narrative_owner}. "
            "This validates category attention but should not be treated as the whole NXTC thesis."
        )
    if b7h4_sig:
        parts.append(
            f"B7-H4 comparator: {b7h4_sig.strategic_state}. "
            f"Named sponsors: {', '.join(b7h4_sig.sponsors)}. It is read-through, not replacement."
        )
    return " ".join(parts) if parts else "No high-priority direct oncology ammunition surfaced in this clinical run."


def _proof_burden_read(sig: ClinicalLaneSignature | None) -> str | None:
    if sig is None:
        return None
    return (
        f"Clinical proof burden for {sig.label}: {sig.proof_burden}. "
        "The practical edge is knowing what the lane must prove for its narrative to become harder to dismiss."
    )


def _trend_line(sig: ClinicalLaneSignature, role: str) -> str:
    sponsor_text = ", ".join(sig.sponsors) if sig.sponsors else "no named sponsors surfaced"
    return (
        f"{sig.label}: {sig.strategic_state}. {role}: {sig.edge_thesis} "
        f"{_phase_architecture_short_from_sig(sig)}. Sponsors: {sponsor_text}. Confidence: {sig.confidence}."
    )


def _trend_lines_from_signatures(signatures: list[ClinicalLaneSignature]) -> list[str]:
    # Progressive hierarchy: CDH6 first when present, then category context, then comparator.
    by_label = {s.label: s for s in signatures}
    ordered: list[ClinicalLaneSignature] = []
    for label in ["CDH6 / ovarian ADC", "Broad ovarian ADC", "B7-H4 ADC"]:
        sig = by_label.get(label)
        if sig:
            ordered.append(sig)
    ordered.extend([s for s in sorted(signatures, key=lambda s: s.priority_score, reverse=True) if s not in ordered])
    lines: list[str] = []
    for sig in ordered:
        if sig.label.startswith("CDH6"):
            lines.append(_trend_line(sig, "Core battlefield read"))
        elif "Broad ovarian ADC" in sig.label:
            lines.append(_trend_line(sig, "Category context"))
        elif "B7-H4" in sig.label:
            lines.append(_trend_line(sig, "Comparator read"))
    return lines


def _positioning_line(cdh6_sig: ClinicalLaneSignature | None, ovarian_sig: ClinicalLaneSignature | None, b7h4_sig: ClinicalLaneSignature | None) -> str:
    if cdh6_sig:
        comparator = ""
        if ovarian_sig and b7h4_sig:
            comparator = " Broad ovarian ADC supports category relevance; B7-H4 supports adjacent gynecologic-oncology attention, but neither replaces the CDH6-specific answer."
        return (
            "NXTC should not be framed as generic ADC exposure. "
            f"The stronger positioning line is CDH6 narrative ownership, not category participation: {cdh6_sig.edge_thesis} "
            f"{cdh6_sig.proof_burden}.{comparator}"
        )
    if ovarian_sig:
        return f"NXTC should be judged against active ovarian ADC category context, but the positioning answer still needs target specificity. {ovarian_sig.edge_thesis}"
    if b7h4_sig:
        return "B7-H4 provides gynecologic-oncology attention read-through, but it should not replace a company-specific CDH6 positioning answer."
    return "The live clinical pull did not create a strong direct positioning read this run."


def _db_hook_evidence(records: list[TrialRecord]) -> str:
    return f"Structured ClinicalTrials.gov records preserved for future longitudinal database comparison: {len(records)}."


def _side_channel_read(records: list[TrialRecord]) -> str | None:
    parts: list[str] = []
    for lane in SIDE_LANE_ORDER:
        lane_recs = _lane_records(records, lane)
        if not lane_recs:
            continue
        sig = _derive_lane_signature(_lane_label(lane), lane_recs)
        if sig:
            parts.append(f"{sig.label}: {sig.strategic_state}; {sig.narrative_owner}")
    if not parts:
        return None
    return "Exploratory watch only: " + " | ".join(parts) + "."


def _build_signals(records: list[TrialRecord], errors: list[str]) -> list[ClinicalTrialSignal]:
    signals: list[ClinicalTrialSignal] = []
    if not records:
        detail = "ClinicalTrials.gov did not provide enough usable signal to support a clinical-landscape conclusion in this run."
        if errors:
            detail += " Source diagnostics were captured without interrupting the dashboard."
        return [ClinicalTrialSignal(
            bucket="new_information",
            title="Clinical source check",
            finding=detail,
            value="This prevents the dashboard from overstating external clinical intelligence when the source pull is degraded or empty.",
            evidence="; ".join(errors[:3]) if errors else "No matching records returned.",
            priority=99,
        )]

    cdh6_records = _lane_records(records, "CDH6 / Ovarian ADC")
    b7h4_records = _lane_records(records, "B7-H4 ADC")
    ovarian_records = _lane_records(records, "Ovarian ADC")
    adc_records = _lane_records(records, "ADC Oncology")
    side_records = [r for r in records if r.lane in SIDE_LANE_ORDER]

    cdh6_sig = _derive_lane_signature("CDH6 / ovarian ADC", cdh6_records)
    ovarian_sig = _derive_lane_signature("Broad ovarian ADC", ovarian_records)
    b7h4_sig = _derive_lane_signature("B7-H4 ADC", b7h4_records)
    signatures = [s for s in [cdh6_sig, ovarian_sig, b7h4_sig] if s is not None]
    signature_records = cdh6_records + ovarian_records + b7h4_records

    # Q1: What was found. One clean prioritized read plus direct oncology movement.
    if signatures:
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Priority clinical edge signature",
            finding=_clinical_leverage_thesis(cdh6_sig, ovarian_sig, b7h4_sig),
            value="The clinical read is derived from combinations of phase anchors, sponsor structure, and protocol-language signals rather than any single raw field.",
            evidence=" | ".join(_signature_evidence(s) for s in signatures),
            priority=1,
        ))

    recent = _recent_movement_read(records)
    if recent:
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Direct oncology movement",
            finding=recent,
            value="Only direct oncology movement is elevated here; exploratory updates remain supporting context unless they change the core thesis.",
            evidence="; ".join(f"{r.nct_id}: {r.sponsor} — {r.title}" for r in sorted(_active_records(records), key=lambda r: r.last_update or "", reverse=True)[:6]),
            priority=2,
        ))

    # Q2: Why it matters. Advance the meaning, do not restate Q1.
    if signatures:
        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Investor ammunition",
            finding=_investor_ammunition_read(cdh6_sig, ovarian_sig, b7h4_sig),
            value="This advances the investor or board conversation from “the category is active” to “what does CDH6 still have room to own, and what proof burden matters next.”",
            evidence=" | ".join(_signature_evidence(s) for s in signatures),
            priority=3,
        ))

    proof_sig = cdh6_sig or ovarian_sig or b7h4_sig
    proof_line = _proof_burden_read(proof_sig)
    if proof_line:
        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Clinical proof burden",
            finding=proof_line,
            value="This converts protocol structure into investor preparation rather than simply showing trial activity.",
            evidence=_signature_evidence(proof_sig),
            priority=4,
        ))

    # Q3: Trend. Keep lane states separated and confidence-weighted.
    for idx, line in enumerate(_trend_lines_from_signatures(signatures), start=5):
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Target-specific battlefield state",
            finding=line,
            value="Trend reads stay separated by lane so CDH6, broad ovarian ADC, and B7-H4 do not collapse into one generic ADC headline.",
            evidence=_db_hook_evidence(signature_records),
            priority=idx,
        ))

    if adc_records:
        adc_sig = _derive_lane_signature("Broader ADC oncology", adc_records)
        if adc_sig and adc_sig.priority_score >= 4:
            signals.append(ClinicalTrialSignal(
                bucket="trend",
                title="ADC category weather",
                finding=f"Broader ADC oncology: {adc_sig.strategic_state}. This remains modality/category weather, not the core CDH6 positioning thesis. Sponsors: {', '.join(adc_sig.sponsors)}. Confidence: {adc_sig.confidence}.",
                value="Broad ADC activity can support category attention, but it should not replace the target-specific CDH6 answer.",
                evidence=_signature_evidence(adc_sig),
                priority=8,
            ))

    side_line = _side_channel_read(side_records)
    if side_line:
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Exploratory watch discipline",
            finding=side_line,
            value="Optionality is visible, but the executive thesis remains anchored to CDH6 / ovarian ADC unless side-channel movement becomes strategically material.",
            evidence=_db_hook_evidence(side_records),
            priority=9,
        ))

    # Q4: Positioning. One line that uses the inferred state, not another data recap.
    signals.append(ClinicalTrialSignal(
        bucket="positioning",
        title="NXTC positioning implication",
        finding=_positioning_line(cdh6_sig, ovarian_sig, b7h4_sig),
        value="This converts clinical-trial structure into a positioning answer rather than a sponsor list.",
        evidence=" | ".join(_signature_evidence(s) for s in signatures) if signatures else "ClinicalTrials.gov records are kept below the Executive Summary as supporting evidence.",
        priority=10,
    ))

    return sorted(signals, key=lambda s: s.priority)

def build_clinical_trials_intelligence() -> ClinicalTrialsSummary:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    by_nct: dict[str, TrialRecord] = {}
    errors: list[str] = []

    for spec in CLINICAL_TRIAL_SEARCH_SPECS:
        records, error = _fetch_spec(spec)
        if error:
            errors.append(error)
        for record in records:
            existing = by_nct.get(record.nct_id)
            # Keep the highest-priority/source-specific lane for duplicates.
            if existing is None:
                by_nct[record.nct_id] = record
            else:
                existing_priority = next((s.priority for s in CLINICAL_TRIAL_SEARCH_SPECS if s.label == existing.lane), 99)
                if spec.priority < existing_priority:
                    by_nct[record.nct_id] = record

    records = list(by_nct.values())
    records.sort(key=lambda r: (r.last_update or "", r.nct_id), reverse=True)
    signals = _build_signals(records, errors)
    table = _trial_table(records)
    payload = [asdict(record) | {"fetched_at_utc": fetched_at, "source": "clinicaltrials.gov"} for record in records]
    active_count = sum(1 for r in records if _is_active(r.status))
    source_status = "live" if records else ("degraded" if errors else "empty")

    return ClinicalTrialsSummary(
        source_status=source_status,
        fetched_at_utc=fetched_at,
        total_trials=len(records),
        active_trials=active_count,
        lanes_covered=sorted({r.lane for r in records}),
        signals=signals,
        trial_table=table,
        persistence_payload=payload,
        source_errors=errors,
    )
