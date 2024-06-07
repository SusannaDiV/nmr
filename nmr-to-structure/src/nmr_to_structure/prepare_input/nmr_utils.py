import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import regex as re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rxn.chemutils.tokenization import tokenize_smiles
from sklearn.model_selection import train_test_split
import json
import logging

RANDOM_SEED = 3246

def tokenize_formula(formula: str) -> list:
    return re.findall("[A-Z][a-z]?|\d+|.", formula)

def jitter(value, jitter_range: float = 2):
    jitter_value = np.random.uniform(-jitter_range, +jitter_range)
    return value + jitter_value

def split_data(input_data: Any) -> Tuple[Any, Any, Any]:
    train_data, test_data = train_test_split(
        input_data, test_size=0.1, random_state=RANDOM_SEED
    )
    train_data, val_data = train_test_split(
        train_data, test_size=0.05, random_state=RANDOM_SEED
    )

    return (train_data, test_data, val_data)

def build_1H_peak(
    HNMR_sim_peaks: dict,
    peak: str,
    jitter_peaks: bool = False,
    mode: str = "adaptive",
    token_space: str = "separate",
) -> Tuple[float, str]:
    try:
        range_max = HNMR_sim_peaks[peak]["rangeMax"]
        range_min = HNMR_sim_peaks[peak]["rangeMin"]
        centroid = HNMR_sim_peaks[peak]["centroid"]
        category = HNMR_sim_peaks[peak]["category"]
        nH = HNMR_sim_peaks[peak]["nH"]
    except KeyError as e:
        logging.error(f"Missing key in peak data for {peak}: {e}")
        raise

    if (range_max - range_min > 0.15 and mode == "adaptive") or mode == "range":
        max_val = jitter(round(range_max, 2), jitter_range=0.2) if jitter_peaks else round(range_max, 2)
        min_val = jitter(round(range_min, 2), jitter_range=0.2) if jitter_peaks else round(range_min, 2)

        if token_space == "separate":
            max_val = "1H{:.2f}".format(max_val)
            min_val = "1H{:.2f}".format(min_val)
        elif token_space == "shared":
            max_val = "{:.1f}".format(max_val * 10)
            min_val = "{:.1f}".format(min_val * 10)

        peak_string = "| {} {} {} {}H ".format(min_val, max_val, category, nH)
        return range_max, peak_string
    else:
        centroid_val = jitter(round(centroid, 2), jitter_range=0.2) if jitter_peaks else round(centroid, 2)

        if token_space == "separate":
            centroid_val = "1H{:.2f}".format(centroid_val)
        elif token_space == "shared":
            centroid_val = "{:.1f}".format(centroid_val * 10)

        peak_string = "| {} {} {}H ".format(centroid_val, category, nH)
        return centroid, peak_string

def build_hnmr_string(
    smiles: str,
    peak_dict: Any,
    mode: str = "adaptive",
    header: bool = True,
    token_space: str = "same",
    n_aug: int = 0,
) -> List[str]:
    # Check if peak_dict is a string and attempt to convert it
    if isinstance(peak_dict, str):
        logging.error("Expected peak_dict to be a dictionary, but got a string.")
        try:
            # Attempt to convert the string to a dictionary if it's in JSON format
            peak_dict = json.loads(peak_dict)
            logging.info("Converted peak_dict from JSON string to dictionary.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to convert peak_dict to dictionary: {e}")
            raise ValueError("peak_dict must be a dictionary")

    # Check if peak_dict is a dictionary
    if not isinstance(peak_dict, dict):
        logging.error("peak_dict is not a dictionary.")
        raise ValueError("peak_dict must be a dictionary")

    # Log the contents of peak_dict for debugging
    logging.info(f"Contents of peak_dict: {json.dumps(peak_dict, indent=2)}")

    # Construct NMR string
    mol = Chem.MolFromSmiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    if header:
        formula_split = tokenize_formula(formula)
        formula_tokenized = " ".join(list(filter(None, formula_split)))
        nmr_header = "{} 1HNMR ".format(formula_tokenized)
    else:
        nmr_header = "1HNMR "

    peak_strings = list()

    for i in range(n_aug + 1):
        # No augmentation for the first set
        processed_peak = dict()
        for peak in peak_dict.keys():
            try:
                peak_pos, peak_string = build_1H_peak(
                    peak_dict,
                    peak,
                    jitter_peaks=True if i > 0 else False,
                    mode=mode,
                    token_space=token_space,
                )
                processed_peak[peak_pos] = peak_string
            except KeyError as e:
                logging.error(f"Missing key in peak_dict for peak {peak}: {e}")
                continue

        # Order such that peaks are in ascending order
        peak_string = nmr_header
        for _, peak in sorted(processed_peak.items()):
            peak_string = peak_string + peak

        peak_strings.append(peak_string)

    return peak_strings

def build_cnmr_string(
    C_NMR_entry: dict,
    header: bool = False,
    smiles: Optional[str] = None,
    token_space="shared",
    n_aug: int = 0,
) -> List[str]:
    if header:
        mol = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        formula_split = tokenize_formula(formula)
        formula_tokenized = " ".join(list(filter(None, formula_split)))

        nmr_header = "{} 13C_NMR".format(formula_tokenized)

    else:
        nmr_header = "13C_NMR"

    nmr_strings = list()
    for i in range(n_aug + 1):
        peaks = list()

        for peak in C_NMR_entry["peaks"].values():
            if peak["delta (ppm)"] > 230 or peak["delta (ppm)"] < -20:
                continue

            value = float(round(peak["delta (ppm)"], 1))
            value_str = str(jitter(value, jitter_range=0.5) if i > 0 else value)

            if token_space == "separate":
                value_str = "13C" + str(value)

            peaks.append(value_str)

        peaks = sorted(peaks)

        nmr_string = nmr_header
        for peak in peaks:
            nmr_string += f" {peak}"
        nmr_strings.append(nmr_string)

    return nmr_strings

def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str) -> None:
    smiles = list(data_set.smiles)
    smiles = [tokenize_smiles(smile.strip()) for smile in smiles]

    os.makedirs(out_path, exist_ok=True)

    with open(out_path / f"tgt-{set_type}.txt", "w") as f:
        for item in smiles:
            f.write(f"{item}\n")

    nmr_input = data_set.nmr_input
    with open(out_path / f"src-{set_type}.txt", "w") as f:
        for item in nmr_input:
            f.write(f"{item}\n")
