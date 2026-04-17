from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd


_PATIENT_LINE_RE = re.compile(r"^\s*patient_id:\s*(\d+)\s*$", re.IGNORECASE)
_CXR_LINE_RE = re.compile(r"^\s*CXR_REPORT\s+study_id=(\d+)\s+impression=(.*)$", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r"^\s*[A-Z][A-Z /_-]{2,}:\s*$")


def _normalize_numeric(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\D", "", text)


def _clean_inline_text(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    return cleaned


def resolve_cxr_report_path(cxr_files_root: Path, subject_id: object, study_id: object) -> Path:
    sid = _normalize_numeric(subject_id)
    stid = _normalize_numeric(study_id)
    if not sid or not stid:
        return cxr_files_root / "invalid.txt"
    return cxr_files_root / f"p{sid[:2]}" / f"p{sid}" / f"s{stid}.txt"


def extract_impression(report_text: str) -> str:
    lines = [line.rstrip() for line in report_text.splitlines()]
    if not lines:
        return ""

    for idx, raw in enumerate(lines):
        line = raw.strip()
        upper = line.upper()
        if not upper.startswith("IMPRESSION:"):
            continue

        first = line.split(":", 1)[1].strip()
        collected = [first] if first else []

        j = idx + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                if collected:
                    j += 1
                    continue
                j += 1
                continue
            if _SECTION_HEADER_RE.match(nxt):
                break
            collected.append(nxt)
            j += 1

        text = _clean_inline_text(" ".join(collected))
        if text:
            return text

    non_empty = [_clean_inline_text(line) for line in lines if line.strip()]
    if not non_empty:
        return ""
    return _clean_inline_text(" ".join(non_empty[:3]))[:500]


def load_subject_to_cxr_reports(
    cxr_record_list_path: Path,
    cxr_files_root: Path,
    subject_ids: Iterable[object],
    max_reports_per_patient: int = 3,
    chunksize: int = 200000,
) -> Tuple[Dict[str, List[dict]], dict]:
    wanted = {_normalize_numeric(v) for v in subject_ids if _normalize_numeric(v)}
    stats = {
        "requested_patients": len(wanted),
        "patients_with_record_entry": 0,
        "patients_with_loaded_reports": 0,
        "reports_loaded": 0,
        "missing_report_files": 0,
        "reports_without_impression": 0,
    }

    if not wanted or max_reports_per_patient <= 0:
        return {}, stats

    if not cxr_record_list_path.exists() or not cxr_files_root.exists():
        return {}, stats

    subject_to_studies: Dict[str, List[str]] = defaultdict(list)
    subject_counts = {sid: 0 for sid in wanted}
    seen_pairs = set()
    pending = set(wanted)

    reader = pd.read_csv(
        cxr_record_list_path,
        usecols=["subject_id", "study_id"],
        dtype=str,
        chunksize=chunksize,
    )

    for chunk in reader:
        if not pending:
            break

        chunk["subject_id"] = chunk["subject_id"].astype(str).map(_normalize_numeric)
        chunk["study_id"] = chunk["study_id"].astype(str).map(_normalize_numeric)
        filtered = chunk[chunk["subject_id"].isin(pending)]
        if filtered.empty:
            continue

        filtered = filtered.drop_duplicates(subset=["subject_id", "study_id"])
        for row in filtered.itertuples(index=False):
            sid = row.subject_id
            stid = row.study_id
            if not sid or not stid:
                continue
            if subject_counts[sid] >= max_reports_per_patient:
                continue

            key = (sid, stid)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            subject_to_studies[sid].append(stid)
            subject_counts[sid] += 1
            if subject_counts[sid] >= max_reports_per_patient:
                pending.discard(sid)

    subject_to_reports: Dict[str, List[dict]] = {}

    for sid in wanted:
        studies = subject_to_studies.get(sid, [])
        if studies:
            stats["patients_with_record_entry"] += 1

        loaded = []
        for stid in studies:
            report_path = resolve_cxr_report_path(cxr_files_root, sid, stid)
            if not report_path.exists():
                stats["missing_report_files"] += 1
                continue

            text = report_path.read_text(encoding="utf-8", errors="ignore")
            impression = extract_impression(text)
            if not impression:
                stats["reports_without_impression"] += 1

            loaded.append(
                {
                    "subject_id": sid,
                    "study_id": stid,
                    "report_path": str(report_path),
                    "impression": impression,
                }
            )
            stats["reports_loaded"] += 1

        if loaded:
            subject_to_reports[sid] = loaded
            stats["patients_with_loaded_reports"] += 1

    return subject_to_reports, stats


def serialize_cxr_reports_for_note(reports: Iterable[dict], max_impression_chars: int = 280) -> List[str]:
    lines = []
    for report in reports:
        stid = _normalize_numeric(report.get("study_id", ""))
        if not stid:
            continue
        impression = _clean_inline_text(report.get("impression", ""))
        if not impression:
            impression = "No clear IMPRESSION section extracted."
        if len(impression) > max_impression_chars:
            impression = impression[: max_impression_chars - 3].rstrip() + "..."
        lines.append(f"CXR_REPORT study_id={stid} impression={impression}")
    return lines


def extract_cxr_mentions_from_text(raw_text: str) -> Tuple[str | None, List[dict]]:
    patient_id = None
    mentions = []
    seen = set()

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not patient_id:
            match_patient = _PATIENT_LINE_RE.match(line)
            if match_patient:
                patient_id = match_patient.group(1)

        match_cxr = _CXR_LINE_RE.match(line)
        if not match_cxr:
            continue

        study_id = _normalize_numeric(match_cxr.group(1))
        impression = _clean_inline_text(match_cxr.group(2))
        if not study_id:
            continue

        key = (study_id, impression)
        if key in seen:
            continue
        seen.add(key)
        mentions.append({"study_id": study_id, "impression": impression})

    return patient_id, mentions
