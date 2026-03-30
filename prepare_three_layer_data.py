import argparse
from pathlib import Path

import pandas as pd


def _safe(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def build_bottom_layer(mimic_root: Path, out_dir: Path, max_files: int = 12) -> int:
    hosp = mimic_root / "hosp"
    d_icd_diag = pd.read_csv(hosp / "d_icd_diagnoses.csv")
    d_icd_proc = pd.read_csv(hosp / "d_icd_procedures.csv")
    d_lab = pd.read_csv(hosp / "d_labitems.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    # Build compact ontology-like dictionary files.
    chunks = []

    diag_lines = []
    for _, row in d_icd_diag.head(2500).iterrows():
        code = _safe(row.get("icd_code"))
        ver = _safe(row.get("icd_version"))
        title = _safe(row.get("long_title"))
        if title:
            diag_lines.append(f"DIAGNOSIS code={code} icd_version={ver} name={title}")
    chunks.append(("diagnosis_dictionary", diag_lines))

    proc_lines = []
    for _, row in d_icd_proc.head(1800).iterrows():
        code = _safe(row.get("icd_code"))
        ver = _safe(row.get("icd_version"))
        title = _safe(row.get("long_title"))
        if title:
            proc_lines.append(f"PROCEDURE code={code} icd_version={ver} name={title}")
    chunks.append(("procedure_dictionary", proc_lines))

    lab_lines = []
    for _, row in d_lab.head(1800).iterrows():
        itemid = _safe(row.get("itemid"))
        label = _safe(row.get("label"))
        fluid = _safe(row.get("fluid"))
        category = _safe(row.get("category"))
        if label:
            lab_lines.append(
                f"LAB_TEST itemid={itemid} label={label} fluid={fluid} category={category}"
            )
    chunks.append(("lab_dictionary", lab_lines))

    for name, lines in chunks:
        if files_written >= max_files:
            break
        if not lines:
            continue
        path = out_dir / f"bottom_{files_written:02d}_{name}.txt"
        text = "medical dictionary entries:\n" + "\n".join(lines) + "\n"
        path.write_text(text, encoding="utf-8")
        files_written += 1

    return files_written


def build_middle_layer(repo_root: Path, out_dir: Path, max_files: int = 12) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    # Source 1: MedicalBook.xlsx sheets (if available and readable).
    book_path = repo_root / "MedicalBook.xlsx"
    if book_path.exists() and files_written < max_files:
        try:
            xl = pd.ExcelFile(book_path)
            for sheet in xl.sheet_names:
                if files_written >= max_files:
                    break
                df = xl.parse(sheet)
                lines = []
                for _, row in df.head(400).iterrows():
                    vals = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v).strip() != "nan"]
                    if vals:
                        lines.append(" | ".join(vals))
                if lines:
                    p = out_dir / f"middle_{files_written:02d}_{sheet.replace(' ', '_')}.txt"
                    p.write_text(
                        f"medical knowledge source sheet={sheet}\n" + "\n".join(lines) + "\n",
                        encoding="utf-8",
                    )
                    files_written += 1
        except Exception:
            pass

    # Source 2 fallback: derive guideline-like text from MIMIC dictionaries.
    if files_written < max_files:
        hosp = repo_root / "mimic-iv-clinical-database-demo-2.2" / "hosp"
        d_icd_diag = pd.read_csv(hosp / "d_icd_diagnoses.csv")
        d_icd_proc = pd.read_csv(hosp / "d_icd_procedures.csv")

        d = d_icd_diag["long_title"].dropna().astype(str).head(1500).tolist()
        p = d_icd_proc["long_title"].dropna().astype(str).head(1000).tolist()

        # Chunk into guideline-like documents.
        step = 250
        for i in range(0, min(len(d), 1000), step):
            if files_written >= max_files:
                break
            diag_chunk = d[i : i + step]
            proc_chunk = p[i : i + step]
            lines = ["clinical guideline style notes:"]
            for j, t in enumerate(diag_chunk[:120], 1):
                lines.append(f"Condition {j}: Consider diagnosis '{t}'.")
            for j, t in enumerate(proc_chunk[:120], 1):
                lines.append(f"Intervention {j}: Procedure option '{t}'.")
            out = out_dir / f"middle_{files_written:02d}_guideline_chunk.txt"
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            files_written += 1

    return files_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local bottom/middle/top layer text datasets")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--top-path", type=Path, default=Path("dataset/mimic_demo_10"))
    parser.add_argument("--out-root", type=Path, default=Path("dataset/three_layer"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_root = args.out_root.resolve()
    bottom_dir = out_root / "bottom"
    middle_dir = out_root / "middle"
    top_dir = out_root / "top"

    bottom_n = build_bottom_layer(repo_root / "mimic-iv-clinical-database-demo-2.2", bottom_dir)
    middle_n = build_middle_layer(repo_root, middle_dir)

    top_dir.mkdir(parents=True, exist_ok=True)
    top_n = 0
    for src in sorted(args.top_path.resolve().glob("*.txt")):
        dst = top_dir / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        top_n += 1

    print(f"bottom_files={bottom_n}")
    print(f"middle_files={middle_n}")
    print(f"top_files={top_n}")
    print(f"output_root={out_root}")


if __name__ == "__main__":
    main()
