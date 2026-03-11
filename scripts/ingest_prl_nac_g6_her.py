#!/usr/bin/env python3
"""Ingest the PrL-NAc-G6-HER dataset into DuckDB.

Walks the 3-level directory tree (phase → animal → FOV) under DATA_ROOT,
constructs a pynapse Sample for each FOV, and ingests it into the database.

Usage:
    python scripts/ingest_prl_nac_g6_her.py              # full ingestion
    python scripts/ingest_prl_nac_g6_her.py --dry-run    # discover & validate only
    python scripts/ingest_prl_nac_g6_her.py --db-path /tmp/test.duckdb
"""

import argparse
import re
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/thejoshbq/Otis-Lab/Analysis/PrL-NAc-G6-HER/data/")
DB_PATH = None   # default: ~/.pynapse/pynapse.duckdb
RAW_DIR = None   # default: ~/.pynapse/raw/
PROJECT_NAME = "PrL-NAc-G6-HER"
PARADIGM_NAME = "legacy_her"
GENOTYPE = "G6"
FPS = 30.0
FRAME_AVERAGING = 4

PHASE_MAP = {
    "0 EarlyAcq": "EarlyAcq",
    "1 MidAcq": "MidAcq",
    "2 LateAcq": "LateAcq",
    "3 EarlyExt": "EarlyExt",
    "4 LateExt": "LateExt",
    "5 CueRein": "CueRein",
    "6 DrugRein": "DrugRein",
    "7 TMTRein": "TMTRein",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FOV_RE = re.compile(r"(FOV\d+)", re.IGNORECASE)


def normalize_animal_name(dirname: str) -> str:
    """Canonicalize e.g. ``PrL-NAC-G6-2F`` → ``PrL-NAc-G6-2F``."""
    return re.sub(r"PrL-NAC-G6", "PrL-NAc-G6", dirname)


def normalize_fov_name(dirname: str) -> str:
    """Extract ``FOV{N}`` from variants like ``FOV1_tracked``, ``FOV2 _tracked``."""
    m = _FOV_RE.search(dirname)
    if not m:
        raise ValueError(f"Cannot parse FOV name from: {dirname!r}")
    return m.group(1).upper().replace("FOV", "FOV")  # ensure "FOV" casing


def extract_sex(animal_name: str) -> str:
    """Return ``'M'`` or ``'F'`` from the last character of the animal name."""
    last = animal_name.rstrip()[-1].upper()
    if last not in ("M", "F"):
        raise ValueError(f"Cannot determine sex from animal name: {animal_name!r}")
    return last


def _find_single(directory: Path, glob_pattern: str, label: str) -> Path:
    """Find exactly one file matching *glob_pattern* in *directory*."""
    matches = list(directory.glob(glob_pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No {label} found in {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {label} files in {directory}: {[m.name for m in matches]}")
    return matches[0]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_fovs(data_root: Path) -> list[dict]:
    """Walk the 3-level tree and return a list of FOV descriptors."""
    fovs = []
    for phase_dir in sorted(data_root.iterdir()):
        if not phase_dir.is_dir() or phase_dir.name not in PHASE_MAP:
            continue
        phase_label = PHASE_MAP[phase_dir.name]

        for animal_dir in sorted(phase_dir.iterdir()):
            if not animal_dir.is_dir():
                continue
            # Skip population-level analysis artifacts
            if not animal_dir.name.startswith("PrL-"):
                continue
            subject_name = normalize_animal_name(animal_dir.name)
            sex = extract_sex(subject_name)

            for fov_dir in sorted(animal_dir.iterdir()):
                if not fov_dir.is_dir():
                    continue
                fov_base = normalize_fov_name(fov_dir.name)
                fov_name = f"{phase_label}_{fov_base}"

                # Locate the .mat and *extractedsignals_raw*.npy files
                mat_file = _find_single(fov_dir, "*.mat", ".mat event log")
                npy_file = _find_single(fov_dir, "*extractedsignals_raw*.npy", "signals .npy")

                fovs.append({
                    "phase_dir": phase_dir.name,
                    "phase": phase_label,
                    "subject_name": subject_name,
                    "sex": sex,
                    "fov_name": fov_name,
                    "mat_file": mat_file,
                    "npy_file": npy_file,
                })
    return fovs


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_dataset(fovs: list[dict], db_path: str):
    """Create the DB and ingest all discovered FOVs."""
    from pynapse.config.events import LEGACY_HER
    from pynapse.core.sample import Sample
    from pynapse.db import engine, ingest

    conn = engine.connect(db_path=db_path)

    project_id = ingest._get_or_create_project(PROJECT_NAME, conn)

    # One population per phase directory; subjects keyed by (phase_dir, name)
    pop_cache: dict[str, int] = {}
    subject_cache: dict[tuple[str, str], int] = {}

    ingested = 0
    failed = 0
    total = len(fovs)

    for i, fov in enumerate(fovs, 1):
        tag = f"[{i}/{total}] {fov['phase']} / {fov['subject_name']} / {fov['fov_name']}"
        try:
            # Get or create population for this phase
            phase_dir = fov["phase_dir"]
            if phase_dir not in pop_cache:
                pop_cache[phase_dir] = ingest._get_or_create_population(
                    phase_dir, project_id, conn,
                )
            pop_id = pop_cache[phase_dir]

            # Get or create subject within this population
            sname = fov["subject_name"]
            subj_key = (phase_dir, sname)
            if subj_key not in subject_cache:
                subject_cache[subj_key] = ingest._get_or_create_subject(
                    sname, pop_id, conn, sex=fov["sex"], genotype=GENOTYPE,
                )
            subject_id = subject_cache[subj_key]

            sample = Sample(
                event_data=str(fov["mat_file"]),
                signal_data=str(fov["npy_file"]),
                name=fov["fov_name"],
                event_dict=LEGACY_HER,
                fps=FPS,
                frame_averaging=FRAME_AVERAGING,
            )

            fov_id = ingest.from_sample(
                sample, subject_id, PARADIGM_NAME,
                fov_name=fov["fov_name"], conn=conn,
                copy_raw=True, raw_dir=RAW_DIR,
            )
            print(f"  OK  {tag}  (fov_id={fov_id})")
            ingested += 1

        except Exception:
            print(f"  FAIL {tag}")
            traceback.print_exc()
            failed += 1

    conn.close()
    print(f"\nDone: {ingested} ingested, {failed} failed out of {total} total.")
    return ingested, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest PrL-NAc-G6-HER into DuckDB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and validate only; do not ingest")
    parser.add_argument("--db-path", default=DB_PATH,
                        help="Database path (default: ~/.pynapse/pynapse.duckdb)")
    args = parser.parse_args()

    print(f"Data root: {DATA_ROOT}")
    print(f"DB path:   {args.db_path or '~/.pynapse/pynapse.duckdb (default)'}")
    print()

    # Discovery
    print("Discovering FOVs...")
    try:
        fovs = discover_fovs(DATA_ROOT)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    n_pops = len(set(f["phase_dir"] for f in fovs))
    n_subjects = len(set(f["subject_name"] for f in fovs))
    print(f"Found {len(fovs)} FOVs across {n_pops} populations and {n_subjects} subjects\n")

    # Summary table
    phases = sorted(set(f["phase"] for f in fovs),
                    key=lambda p: list(PHASE_MAP.values()).index(p))
    for phase in phases:
        phase_fovs = [f for f in fovs if f["phase"] == phase]
        subjects = sorted(set(f["subject_name"] for f in phase_fovs))
        print(f"  {phase:12s}  {len(phase_fovs):3d} FOVs  ({len(subjects)} animals)")
    print()

    if args.dry_run:
        print("Dry run — no ingestion performed.")
        # Print first few for verification
        for f in fovs[:5]:
            print(f"  {f['fov_name']:25s}  mat={f['mat_file'].name}")
            print(f"  {'':25s}  npy={f['npy_file'].name}")
        if len(fovs) > 5:
            print(f"  ... and {len(fovs) - 5} more")
        return

    # Ingest
    ingested, failed = ingest_dataset(fovs, args.db_path)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
