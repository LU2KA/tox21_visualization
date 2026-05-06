"""SDF conversion utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


def _row_from_mol(mol: Chem.Mol, meta: list[str], targets: list[str], props: dict) -> dict:
    row: dict = {"smiles": Chem.MolToSmiles(mol)}
    for key in meta:
        row[key] = props.get(key, np.nan)
    for key in targets:
        row[key] = props.get(key, np.nan)
    for key, value in props.items():
        if key not in meta + targets:
            row[key] = value
    return row


def convert_sdf_to_csv(
    sdf_path: str | Path,
    output_path: str | Path | None = None,
    sanitize: bool = True,
) -> pd.DataFrame:
    """Convert SDF file to a CSV-like DataFrame (and optionally save as CSV)."""
    sdf_path = Path(sdf_path)

    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if output_path is None:
        output_path = sdf_path.with_suffix(".csv")
    else:
        output_path = Path(output_path)

    print(f"Reading SDF file: {sdf_path}")
    print(f"Output CSV file: {output_path}")

    targets = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
        "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
        "SR-HSE", "SR-MMP", "SR-p53",
    ]
    meta = ["DSSTox_CID", "Formula", "FW"]

    rows = []
    invalid = 0

    RDLogger.DisableLog("rdApp.*")

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize)

    for idx, mol in enumerate(suppl):
        if mol is None:
            invalid += 1
            continue

        props = mol.GetPropsAsDict()
        rows.append(_row_from_mol(mol, meta, targets, props))

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} molecules...", end="\r")

    print(f"Valid molecules: {len(rows)}")
    print(f"Invalid molecules: {invalid}")

    df = pd.DataFrame(rows)

    preferred = ["smiles"] + meta + targets
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    for col in targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].isin([0.0, 1.0]), np.nan)

    print(f"Saving to CSV: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} rows to {output_path}")

    return df
