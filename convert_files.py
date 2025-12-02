#!/usr/bin/env python3
"""
Converter for MATLAB (.mat) and NumPy (.npy/.npz) files to JSON.
Supports Grok-compatible uploads: Exports as structured dicts with arrays as lists.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Union
import numpy as np
from scipy.io import loadmat


def to_json_serializable(obj: Any) -> Union[Dict, list, str, int, float]:
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return [to_json_serializable(item) for item in obj.flat]
        else:
            return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, complex):
        return f"{obj.real}+{obj.imag}j"
    else:
        return obj

def convert_mat_to_json(mat_path: str, json_path: str) -> bool:
    try:
        data = loadmat(mat_path, simplify_cells=True)
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        serializable = to_json_serializable(data)
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=4, default=str)
        print(f"Successfully converted {mat_path} to {json_path}")
        return True
    except Exception as e:
        print(f"Error converting {mat_path}: {e}", file=sys.stderr)
        return False

def convert_npy_to_json(npy_path: str, json_path: str) -> bool:
    try:
        if npy_path.endswith('.npz'):
            data = np.load(npy_path)
            serializable = {k: to_json_serializable(v) for k, v in data.items()}
        else:
            data = np.load(npy_path)
            serializable = {"data": to_json_serializable(data)}
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=4, default=str)
        print(f"Successfully converted {npy_path} to {json_path}")
        return True
    except Exception as e:
        print(f"Error converting {npy_path}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert .mat, .npy, or .npz to JSON for Grok uploads."
    )
    parser.add_argument(
        'inputs', nargs='+', help="Input file(s) (.mat, .npy, .npz)"
    )
    parser.add_argument(
        '-o', '--output-dir', default='.', help="Output directory (default: current)"
    )
    args = parser.parse_args()
    success_count = 0
    for input_path in args.inputs:
        if not os.path.isfile(input_path):
            print(f"File not found: {input_path}", file=sys.stderr)
            continue
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        if input_path.endswith('.mat'):
            success = convert_mat_to_json(input_path, json_path)
        elif input_path.endswith(('.npy', '.npz')):
            success = convert_npy_to_json(input_path, json_path)
        else:
            print(f"Unsupported file type: {input_path}", file=sys.stderr)
            continue
        if success:
            success_count += 1
    if success_count == 0:
        sys.exit(1)
    print(f"\nConverted {success_count}/{len(args.inputs)} files successfully.")

if __name__ == "__main__":
    main()