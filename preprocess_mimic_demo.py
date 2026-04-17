import argparse
from pathlib import Path

import pandas as pd

from cxr_integration import load_subject_to_cxr_reports, serialize_cxr_reports_for_note


def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _load_table(path: Path, required_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _select_subjects(admissions: pd.DataFrame, mimic_root: Path, n_patients: int):
    admissions_subjects = admissions["subject_id"].dropna().astype(int).drop_duplicates().tolist()
    admissions_subject_set = set(admissions_subjects)

    demo_subject_path = mimic_root / "demo_subject_id.csv"
    if demo_subject_path.exists():
        demo_subjects = pd.read_csv(demo_subject_path)
        if "subject_id" in demo_subjects.columns:
            selected = []
            for raw in demo_subjects["subject_id"].dropna().tolist():
                try:
                    sid = int(raw)
                except Exception:
                    continue
                if sid in admissions_subject_set:
                    selected.append(sid)
            if selected:
                return selected[:n_patients], "demo_subject_id.csv"

    return admissions_subjects[:n_patients], "admissions.csv"


def build_reports(
    mimic_root: Path,
    output_dir: Path,
    n_patients: int,
    include_cxr: bool = True,
    max_cxr_reports_per_patient: int = 3,
) -> None:
    hosp_dir = mimic_root / "hosp"

    patients = _load_table(hosp_dir / "patients.csv", "patients.csv")
    admissions = _load_table(hosp_dir / "admissions.csv", "admissions.csv")
    diagnoses = _load_table(hosp_dir / "diagnoses_icd.csv", "diagnoses_icd.csv")
    d_icd_diag = _load_table(hosp_dir / "d_icd_diagnoses.csv", "d_icd_diagnoses.csv")
    procedures = _load_table(hosp_dir / "procedures_icd.csv", "procedures_icd.csv")
    d_icd_proc = _load_table(hosp_dir / "d_icd_procedures.csv", "d_icd_procedures.csv")
    labs = _load_table(hosp_dir / "labevents.csv", "labevents.csv")
    d_lab = _load_table(hosp_dir / "d_labitems.csv", "d_labitems.csv")
    prescriptions = _load_table(hosp_dir / "prescriptions.csv", "prescriptions.csv")
    micro = _load_table(hosp_dir / "microbiologyevents.csv", "microbiologyevents.csv")

    for df in [diagnoses, d_icd_diag, procedures, d_icd_proc]:
        if "icd_code" in df.columns:
            df["icd_code"] = df["icd_code"].astype(str)
        if "icd_version" in df.columns:
            df["icd_version"] = df["icd_version"].astype(str)

    diag_map = d_icd_diag[["icd_code", "icd_version", "long_title"]].rename(
        columns={"long_title": "diagnosis_title"}
    )
    proc_map = d_icd_proc[["icd_code", "icd_version", "long_title"]].rename(
        columns={"long_title": "procedure_title"}
    )

    diagnoses = diagnoses.merge(diag_map, on=["icd_code", "icd_version"], how="left")
    procedures = procedures.merge(proc_map, on=["icd_code", "icd_version"], how="left")
    labs = labs.merge(d_lab[["itemid", "label", "category"]], on="itemid", how="left")

    admissions = admissions.sort_values(["subject_id", "admittime"], ascending=[True, True])
    selected_subjects, subject_source = _select_subjects(admissions, mimic_root, n_patients)
    print(f"Selected {len(selected_subjects)} subjects from {subject_source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_to_cxr_reports = {}
    cxr_stats = {
        "requested_patients": 0,
        "patients_with_record_entry": 0,
        "patients_with_loaded_reports": 0,
        "reports_loaded": 0,
        "missing_report_files": 0,
        "reports_without_impression": 0,
    }

    if include_cxr:
        subject_to_cxr_reports, cxr_stats = load_subject_to_cxr_reports(
            cxr_record_list_path=mimic_root / "cxr-record-list.csv",
            cxr_files_root=mimic_root / "cxr" / "files",
            subject_ids=selected_subjects,
            max_reports_per_patient=max(0, max_cxr_reports_per_patient),
        )

    for i, subject_id in enumerate(selected_subjects):
        p_row = patients[patients["subject_id"] == subject_id].head(1)
        p = p_row.iloc[0] if not p_row.empty else None

        sub_adm = admissions[admissions["subject_id"] == subject_id].copy()

        lines = []
        lines.append(f"patient_id: {subject_id}")
        lines.append("history of present illness:")

        if p is not None:
            gender = _safe_str(p.get("gender", ""))
            age = _safe_str(p.get("anchor_age", ""))
            dod = _safe_str(p.get("dod", ""))
            lines.append(f"Demographics: gender {gender}; anchor_age {age}; date_of_death {dod or 'none listed'}.")

        for _, adm in sub_adm.head(3).iterrows():
            hadm = int(adm["hadm_id"])
            admittime = _safe_str(adm.get("admittime", ""))
            dischtime = _safe_str(adm.get("dischtime", ""))
            adm_type = _safe_str(adm.get("admission_type", ""))
            adm_loc = _safe_str(adm.get("admission_location", ""))
            dis_loc = _safe_str(adm.get("discharge_location", ""))
            race = _safe_str(adm.get("race", ""))
            insurance = _safe_str(adm.get("insurance", ""))
            hosp_exp = _safe_str(adm.get("hospital_expire_flag", ""))

            lines.append(
                "Admission summary: "
                f"hadm_id {hadm}; admit {admittime}; discharge {dischtime}; "
                f"type {adm_type}; from {adm_loc}; to {dis_loc}; race {race}; insurance {insurance}; "
                f"hospital_expire_flag {hosp_exp}."
            )

            hadm_diag = diagnoses[diagnoses["hadm_id"] == hadm].sort_values("seq_num").head(12)
            if not hadm_diag.empty:
                diag_parts = []
                for _, d in hadm_diag.iterrows():
                    title = _safe_str(d.get("diagnosis_title", "")) or f"ICD {d.get('icd_code', '')}"
                    seq = _safe_str(d.get("seq_num", ""))
                    diag_parts.append(f"[{seq}] {title}")
                lines.append("Diagnoses: " + "; ".join(diag_parts) + ".")

            hadm_proc = procedures[procedures["hadm_id"] == hadm].sort_values("seq_num").head(8)
            if not hadm_proc.empty:
                proc_parts = []
                for _, pr in hadm_proc.iterrows():
                    title = _safe_str(pr.get("procedure_title", "")) or f"ICD procedure {pr.get('icd_code', '')}"
                    chartdate = _safe_str(pr.get("chartdate", ""))
                    proc_parts.append(f"{title} on {chartdate}".strip())
                lines.append("Procedures: " + "; ".join(proc_parts) + ".")

            hadm_labs = labs[labs["hadm_id"] == hadm].copy()
            if not hadm_labs.empty:
                hadm_labs = hadm_labs.sort_values("charttime").tail(20)
                lab_parts = []
                for _, lb in hadm_labs.iterrows():
                    label = _safe_str(lb.get("label", "")) or f"itemid {lb.get('itemid', '')}"
                    value = _safe_str(lb.get("value", ""))
                    unit = _safe_str(lb.get("valueuom", ""))
                    flag = _safe_str(lb.get("flag", ""))
                    frag = f"{label}: {value} {unit}".strip()
                    if flag:
                        frag += f" (flag {flag})"
                    lab_parts.append(frag)
                lines.append("Recent labs: " + "; ".join(lab_parts) + ".")

            hadm_rx = prescriptions[prescriptions["hadm_id"] == hadm].copy()
            if not hadm_rx.empty:
                hadm_rx = hadm_rx.sort_values("starttime").tail(15)
                med_parts = []
                for _, rx in hadm_rx.iterrows():
                    drug = _safe_str(rx.get("drug", ""))
                    dose = _safe_str(rx.get("dose_val_rx", ""))
                    dose_unit = _safe_str(rx.get("dose_unit_rx", ""))
                    route = _safe_str(rx.get("route", ""))
                    med_parts.append(f"{drug} {dose} {dose_unit} via {route}".replace("  ", " ").strip())
                lines.append("Medications: " + "; ".join([m for m in med_parts if m]) + ".")

            hadm_micro = micro[micro["hadm_id"] == hadm].copy()
            if not hadm_micro.empty:
                hadm_micro = hadm_micro.sort_values("charttime").tail(10)
                micro_parts = []
                for _, mc in hadm_micro.iterrows():
                    test_name = _safe_str(mc.get("test_name", ""))
                    org_name = _safe_str(mc.get("org_name", ""))
                    interp = _safe_str(mc.get("interpretation", ""))
                    comments = _safe_str(mc.get("comments", ""))
                    bit = ", ".join([x for x in [test_name, org_name, interp, comments] if x])
                    if bit:
                        micro_parts.append(bit)
                if micro_parts:
                    lines.append("Microbiology: " + "; ".join(micro_parts) + ".")

        if include_cxr:
            lines.append("radiology findings:")
            cxr_lines = serialize_cxr_reports_for_note(subject_to_cxr_reports.get(str(subject_id), []))
            if cxr_lines:
                lines.extend(cxr_lines)
            else:
                lines.append("No CXR reports available for this patient.")

        lines.append("clinical impression:")
        lines.append(
            "This synthetic note was programmatically assembled from MIMIC-IV demo structured fields "
            "to support first-layer entity-relation graph construction."
        )

        report_path = output_dir / f"report_{i:02d}_subject_{subject_id}.txt"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(selected_subjects)} reports to: {output_dir}")

    if include_cxr:
        print("CXR integration stats:")
        print(f"  patients_requested={cxr_stats['requested_patients']}")
        print(f"  patients_with_record_entry={cxr_stats['patients_with_record_entry']}")
        print(f"  patients_with_loaded_reports={cxr_stats['patients_with_loaded_reports']}")
        print(f"  reports_loaded={cxr_stats['reports_loaded']}")
        print(f"  missing_report_files={cxr_stats['missing_report_files']}")
        print(f"  reports_without_impression={cxr_stats['reports_without_impression']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-IV demo CSV tables into report text files for Medical-Graph-RAG."
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("./mimic-iv-clinical-database-demo-2.2"),
        help="Path to MIMIC-IV demo root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dataset/mimic_demo_10"),
        help="Output directory for generated report text files.",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=10,
        help="Number of unique patients to include.",
    )
    parser.add_argument(
        "--include-cxr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to inject MIMIC-CXR report impressions into each patient note.",
    )
    parser.add_argument(
        "--max-cxr-reports-per-patient",
        type=int,
        default=3,
        help="Maximum number of CXR studies to attach to each patient note.",
    )
    args = parser.parse_args()

    if args.n_patients <= 0:
        raise ValueError("--n-patients must be > 0")
    if args.max_cxr_reports_per_patient < 0:
        raise ValueError("--max-cxr-reports-per-patient must be >= 0")

    build_reports(
        args.mimic_root,
        args.output_dir,
        args.n_patients,
        include_cxr=args.include_cxr,
        max_cxr_reports_per_patient=args.max_cxr_reports_per_patient,
    )


if __name__ == "__main__":
    main()
