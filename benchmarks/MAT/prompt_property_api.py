#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: prompt_property_api.py
# Description: Request LLMs to directly predict numerical property values
#              for molecules (instead of cluster labels).
#              Supports GPT (OpenAI), Gemini (Google), and Qwen 3 (DashScope) APIs.
"""
import os
import json
import numpy as np
import pandas as pd
import time
import re

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dataset config
DATASETS = {
    "redox-mer": {
        "csv": os.path.join(DATA_DIR, "redox_mer.csv"),
        "smiles_col": "SMILES",
        "property_col": "Ered",
        "property_name": "reduction potential (Ered)",
        "property_unit": "eV",
        "property_description": ("the reduction potential (Ered) in eV. "
                                 "These are benzothiadiazine (BTZ) derivatives. "
                                 "Consider the following molecular features that influence reduction potential: "
                                 "electron-withdrawing groups (e.g., CF3, NO2, CN, halogens) increase Ered; "
                                 "electron-donating groups (e.g., alkyl, methoxy, NMe2) decrease Ered; "
                                 "positioning of substituents on aromatic rings; "
                                 "sulfur-containing functional groups (e.g., SCF3, S=O); "
                                 "degree of molecular polarity."),
        "example_low": "1.85",
        "example_high": "2.64",
    },
}


GPT_SYSTEM_MSG = (
    "You are a helpful computational chemistry research assistant. "
    "You are part of an academic benchmark pipeline that evaluates how well "
    "language models can estimate molecular properties from SMILES strings. "
    "Your role is to provide your best numerical estimates based on chemical "
    "intuition and structure-property relationships. Always respond with the "
    "requested numerical values — never refuse, never add disclaimers."
)


def query_gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=4000):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = completion.choices[0].message.content
        return reply.strip()
    except Exception as e:
        print(f"GPT API Error: {e}")
        return None


def query_gemini(prompt, model="gemini-2.5-pro", max_tokens=4000):
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None


def query_qwen(prompt, model="qwen3.5-plus", temperature=0.7, max_tokens=4000):
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reply = completion.choices[0].message.content
        return reply.strip()
    except Exception as e:
        print(f"Qwen API Error: {e}")
        return None


def build_prompt(molecules, dataset_name):
    """Build the prompt asking for direct numerical property predictions."""
    cfg = DATASETS[dataset_name]
    formatted_data = "\n".join([f"{i+1}. {smi}" for i, smi in enumerate(molecules)])
    prompt = (f"You are a computational chemist. For each molecule (given as SMILES), "
              f"predict {cfg['property_description']} "
              f"Provide your best numerical estimate for each molecule.\n\n"
              f"**Respond strictly with the counter and the predicted numerical value only, "
              f"one per line. Do not include any additional text, units, or explanation.**\n"
              f"Example format:\n"
              f"1 2.35\n"
              f"2 1.89\n\n"
              f"Now predict the {cfg['property_name']} for the following molecules:\n"
              f"{formatted_data}\n")
    return prompt


# def parse_response(response, n_molecules):
#     """Parse LLM response into a list of float predictions.

#     Handles formats: "1 2.35", "1. 2.35", "1: 2.35", "1, 2.35", "1\t2.35",
#     negative values like "1 -2.35", and markdown-wrapped responses.
#     """
#     predictions = [None] * n_molecules
#     if response is None:
#         return predictions

#     text = response.strip()
#     # Strip markdown code fences if present
#     text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
#     text = re.sub(r"\n?```$", "", text)

#     # Pattern: line starts with an integer index, followed by a separator,
#     # then a (possibly negative) float value
#     line_pattern = re.compile(r"^\s*(\d+)\s*[.):,\t\s]\s*([+-]?\d+(?:\.\d+)?)")

#     for line in text.split("\n"):
#         m = line_pattern.match(line)
#         if m:
#             try:
#                 idx = int(m.group(1)) - 1  # 1-indexed to 0-indexed
#                 val = float(m.group(2))
#                 if 0 <= idx < n_molecules:
#                     predictions[idx] = val
#             except (ValueError, IndexError):
#                 continue

#     return predictions


def parse_response(response, n_molecules):
    """Parse LLM response into a list of float predictions."""
    predictions = [None] * n_molecules
    if response is None:
        return predictions

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Extract index and floating-point value from various formats:
        # "1 2.35", "1. 2.35", "1: 2.35", "1\t2.35", "1, 2.35"
        # Use regex to find: integer index, then a float value
        match = re.match(
            r"^(\d+)\s*[.,:)]\s*"  # index followed by separator
            r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",  # float value
            line,
        )
        if not match:
            # Try simpler format: "1 2.35" (just whitespace separation)
            match = re.match(
                r"^(\d+)\s+"
                r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                line,
            )
        if match:
            try:
                idx = int(match.group(1)) - 1  # 1-indexed to 0-indexed
                val = float(match.group(2))
                if 0 <= idx < n_molecules:
                    predictions[idx] = val
            except (ValueError, IndexError):
                continue

    return predictions


def request_property_predictions(api_type, dataset_name, batch_size=200, delay=10.0):
    """
    Request direct property predictions from an LLM API.

    Args:
        api_type: "gemini" or "qwen3"
        dataset_name: key in DATASETS dict
        batch_size: number of molecules per API call
        delay: seconds to wait between API calls
    """
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(cfg["csv"])
    molecules = df[cfg["smiles_col"]].tolist()
    n_batches = int(np.ceil(len(molecules) / float(batch_size)))

    folder_path = os.path.join(DATA_DIR, "llm_property_api", api_type, dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    if api_type == "gemini":
        query_fn = query_gemini
    elif api_type == "qwen3":
        query_fn = query_qwen
    else:
        query_fn = query_gpt

    for i in range(n_batches):
        result_file = os.path.join(folder_path, f"{dataset_name}_{i}_result.npy")
        if os.path.exists(result_file):
            print(f"  Batch {i}/{n_batches} already exists, skipping.")
            continue

        start = i * batch_size
        end = min((i + 1) * batch_size, len(molecules))
        batch_mole = molecules[start:end]

        prompt = build_prompt(batch_mole, dataset_name)
        print(f"  Batch {i}/{n_batches}: {len(batch_mole)} molecules")

        time.sleep(delay)
        try:
            response = query_fn(prompt, max_tokens=4000)
        except Exception as e:
            print(f"  Error on batch {i}: {e}. Retrying after {delay}s...")
            time.sleep(delay)
            response = query_fn(prompt, max_tokens=4000)

        if response is None:
            print(f"  WARNING: Batch {i} returned None response")
            continue

        # Save raw response for debugging
        raw_file = os.path.join(folder_path, f"{dataset_name}_{i}_raw_response.txt")
        with open(raw_file, "w") as f:
            f.write(response)

        predictions = parse_response(response, len(batch_mole))

        # Store as list of (SMILES, predicted_value)
        results = []
        n_valid = 0
        for j, (smi, pred) in enumerate(zip(batch_mole, predictions)):
            if pred is not None:
                results.append((smi, str(pred)))
                n_valid += 1
            else:
                results.append((smi, "NaN"))

        print(f"    Parsed {n_valid}/{len(batch_mole)} valid predictions")

        # Save results
        with open(os.path.join(folder_path, f"{dataset_name}_{i}_data.json"), "w") as f:
            json.dump(results, f)
        np.save(result_file, results)

    # Aggregate all batches
    all_results = {}
    for i in range(n_batches):
        result_file = os.path.join(folder_path, f"{dataset_name}_{i}_result.npy")
        if not os.path.exists(result_file):
            continue
        data = np.load(result_file, allow_pickle=True)
        for smi, val in data:
            try:
                all_results[smi] = float(val)
            except (ValueError, TypeError):
                all_results[smi] = np.nan

    print(f"\n  Total: {len(all_results)} molecules, "
          f"{sum(1 for v in all_results.values() if not np.isnan(v))} valid predictions")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Request LLMs to predict molecular property values directly.")
    parser.add_argument(
        "--api",
        type=str,
        choices=["gpt", "gemini", "qwen3"],
        required=True,
        help="Which LLM API to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="redox-mer",
        choices=list(DATASETS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for API calls",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Delay in seconds between API calls",
    )
    args = parser.parse_args()

    print(f"Requesting {args.api} for {args.dataset} property predictions...")
    request_property_predictions(
        api_type=args.api,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    print("Done.")


if __name__ == "__main__":
    main()
