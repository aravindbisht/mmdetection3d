#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

def convert_json_to_pkl(json_path, pkl_path=None):
    """Convert MMDetection3D JSON predictions to PKL format.
    
    Args:
        json_path (str): Path to input JSON file
        pkl_path (str, optional): Path to output PKL file. If None, will use 
                                 same name as JSON but with .pkl extension.
    """
    # Set default output path if not provided
    if pkl_path is None:
        pkl_path = str(Path(json_path).with_suffix('.pkl'))
    
    # Load JSON predictions
    with open(json_path, 'r') as f:
        pred = json.load(f)
    
    # Convert to the format expected by visualize_results.py
    result = [{
        'boxes_3d': np.array(pred['bboxes_3d']),  # Changed from 'boxes_3d' to 'bboxes_3d'
        'scores_3d': np.array(pred['scores_3d']),
        'labels_3d': np.array(pred['labels_3d']),
        'box_type_3d': pred.get('box_type_3d', 'LiDAR')  # Add box type if present
    }]
    
    # Create output directory if it doesn't exist
    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PKL
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f'Successfully converted {json_path} to {pkl_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection3D JSON predictions to PKL format')
    parser.add_argument('json_path', help='Path to input JSON file')
    parser.add_argument('--output', '-o', help='Path to output PKL file (optional)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_json_to_pkl(args.json_path, args.output)
