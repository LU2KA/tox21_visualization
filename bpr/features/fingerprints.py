"""Molecular fingerprint computation."""

from typing import Any, Callable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState.Fingerprinter import FingerprintMol as _estate_fp
import skfp.fingerprints as _skfp


def _fp_to_numpy(fp) -> np.ndarray:
    """Convert an RDKit bit vector to a numpy array."""
    arr = np.zeros(fp.GetNumBits(), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def _fp_maccs(mols: list, **_: Any) -> np.ndarray:
    return np.array([_fp_to_numpy(AllChem.GetMACCSKeysFingerprint(m)) for m in mols])


def _fp_morgan(mols: list, **kwargs: Any) -> np.ndarray:
    r = int(kwargs.get("radius", 2))
    nb = int(kwargs.get("n_bits", 2048))
    return np.array([
        _fp_to_numpy(AllChem.GetMorganFingerprintAsBitVect(m, radius=r, nBits=nb))
        for m in mols
    ])


def _fp_fcfp(mols: list, **kwargs: Any) -> np.ndarray:
    r = int(kwargs.get("radius", 2))
    nb = int(kwargs.get("n_bits", 2048))
    return np.array([
        _fp_to_numpy(AllChem.GetMorganFingerprintAsBitVect(
            m, radius=r, nBits=nb, useFeatures=True,
        ))
        for m in mols
    ])


def _fp_atom_pair(mols: list, **kwargs: Any) -> np.ndarray:
    nb = int(kwargs.get("n_bits", 2048))
    return np.array([
        _fp_to_numpy(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nb))
        for m in mols
    ])


def _fp_topological_torsion(mols: list, **kwargs: Any) -> np.ndarray:
    nb = int(kwargs.get("n_bits", 2048))
    return np.array([
        _fp_to_numpy(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nb))
        for m in mols
    ])


def _fp_rdkit_topo(mols: list, **kwargs: Any) -> np.ndarray:
    nb = int(kwargs.get("n_bits", 2048))
    return np.array([_fp_to_numpy(AllChem.RDKFingerprint(m, fpSize=nb)) for m in mols])


def _fp_avalon(mols: list, **kwargs: Any) -> np.ndarray:
    nb = int(kwargs.get("n_bits", 512))
    return np.array([_fp_to_numpy(pyAvalonTools.GetAvalonFP(m, nBits=nb)) for m in mols])


def _fp_estate(mols: list, **_: Any) -> np.ndarray:
    results = []
    for m in mols:
        counts, _ = _estate_fp(m)
        results.append(np.array(counts, dtype=np.float32))
    return np.array(results)


def _fp_klekota_roth(mols: list, **_: Any) -> np.ndarray:
    return _skfp.KlekotaRothFingerprint().transform(mols).astype(np.int8)


def _fp_map4(mols: list, **kwargs: Any) -> np.ndarray:
    r = int(kwargs.get("radius", 2))
    fp_size = int(kwargs.get("n_bits", 1024))
    return _skfp.MAPFingerprint(fp_size=fp_size, radius=r).transform(mols).astype(np.int8)


def _fp_mhfp(mols: list, **kwargs: Any) -> np.ndarray:
    r = int(kwargs.get("radius", 3))
    fp_size = int(kwargs.get("n_bits", 2048))
    return _skfp.MHFPFingerprint(fp_size=fp_size, radius=r).transform(mols).astype(np.int8)


def _fp_erg(mols: list, **kwargs: Any) -> np.ndarray:
    fuzz = float(kwargs.get("fuzz_increment", 0.3))
    min_p = int(kwargs.get("min_path", 1))
    max_p = int(kwargs.get("max_path", 15))
    return _skfp.ERGFingerprint(
        fuzz_increment=fuzz, min_path=min_p, max_path=max_p,
    ).transform(mols).astype(np.float32)


def _fp_pubchem(mols: list, **_: Any) -> np.ndarray:
    return _skfp.PubChemFingerprint().transform(mols).astype(np.int8)




_REGISTRY: dict[str, Callable] = {
    "maccs":                _fp_maccs,
    "morgan":               _fp_morgan,
    "morgan_2048":          _fp_morgan,           # alias: radius 2, n_bits 2048
    "morgan_4096":          _fp_morgan,           # alias: radius 4, n_bits 4096
    "ecfp":                 _fp_morgan,
    "fcfp":                 _fp_fcfp,
    "atom_pair":            _fp_atom_pair,
    "topological_torsion":  _fp_topological_torsion,
    "rdkit":                _fp_rdkit_topo,
    "avalon":               _fp_avalon,
    "estate":               _fp_estate,
    "klekota_roth":         _fp_klekota_roth,
    "map4":                 _fp_map4,
    "mhfp":                 _fp_mhfp,
    "erg":                  _fp_erg,
    "pubchem":              _fp_pubchem,
}

SUPPORTED_METHODS = sorted(_REGISTRY.keys())


def _smiles_series_to_mols(smiles: pd.Series) -> tuple[dict[Any, Chem.Mol], int]:
    mols: dict[Any, Chem.Mol] = {}
    failed_parse = 0
    for idx, smi in smiles.items():
        if pd.isna(smi) or not isinstance(smi, str) or not smi.strip():
            failed_parse += 1
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            failed_parse += 1
            continue
        mols[idx] = mol
    return mols, failed_parse


def compute_fingerprints(
    smiles: pd.Series,
    method: str = "morgan",
    verbose: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute fingerprints for a Series of SMILES strings."""
    method_lower = method.lower()
    if method_lower not in _REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Supported: {', '.join(SUPPORTED_METHODS)}"
        )

    mols, failed_parse = _smiles_series_to_mols(smiles)

    if not mols:
        raise ValueError(
            f"No valid molecules parsed from {len(smiles)} SMILES. "
        )

    if verbose and failed_parse:
        print(f"  {failed_parse}/{len(smiles)} SMILES failed parsing")

    indices = list(mols.keys())
    mol_list = list(mols.values())
    params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "defaults"
    if verbose:
        print(f"  Computing {method} for {len(mol_list)} molecules ({params_str})...")

    fp_func = _REGISTRY[method_lower]
    fp_array = fp_func(mol_list, **kwargs)

    if verbose:
        print(f"  Result: {fp_array.shape[0]} fingerprints × {fp_array.shape[1]} features")

    fp_df = pd.DataFrame(fp_array, index=indices)
    fp_df.columns = [f"{method}_{i}" for i in range(fp_df.shape[1])]
    return fp_df
