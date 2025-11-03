import mmcv
import numpy as np
import torch
import json
import os
import argparse
import sys

# MMDetection3D/MMEngine imports
try:
    from mmdet3d.visualization import Det3DLocalVisualizer
    from mmdet3d.structures import LiDARInstance3DBoxes, Box3DMode
    from mmengine import load
except ImportError:
    print("ERROR: MMDetection3D or MMengine is not installed. Please ensure your environment is set up correctly.")
    sys.exit(1)


def visualize_projected_boxes(info_file_path: str, image_file_path: str, prediction_json_path: str, output_dir: str, score_threshold: float = 0.3):
    """
    Loads predicted LiDAR bounding boxes, projects them onto a 2D image 
    using KITTI calibration data, and saves the resulting image.

    Args:
        info_file_path (str): Path to the kitti_infos_test.pkl file.
        image_file_path (str): Path to the specific 2D image file (e.g., '000000.png').
        prediction_json_path (str): Path to the prediction results JSON file.
        output_dir (str): Directory to save the visualized image.
        score_threshold (float): Minimum score to display a predicted bounding box.
    """
    
    # --- 1. Identify Sample ID and Setup Output ---
    
    # Extract the unique sample ID (e.g., '000000') from the image filename
    SAMPLE_ID = os.path.splitext(os.path.basename(image_file_path))[0]
    output_filename = f"{SAMPLE_ID}_projected_2d_boxes.png"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Visualizing Sample: {SAMPLE_ID} ---")
    
    # --- 2. Load Data and Calibration ---

    # A. Load KITTI Info (Calibration)
    try:
        info_file = load(info_file_path)
    except Exception as e:
        print(f"ERROR: Could not load KITTI info PKL file from {info_file_path}. {e}", file=sys.stderr)
        return

    # Find the sample info dictionary using the SAMPLE_ID
    # The data might be a list of dicts or a dict with a 'data_list' key
    if 'data_list' in info_file:
        # Newer format with 'data_list' key
        sample_info = next((data for data in info_file['data_list'] 
                          if str(data.get('sample_idx', '')).endswith(SAMPLE_ID)), None)
    elif isinstance(info_file, list):
        # Older format where the list is the top-level object
        sample_info = next((data for data in info_file 
                          if str(data.get('sample_idx', '')).endswith(SAMPLE_ID)), None)
    else:
        print(f"ERROR: Unexpected format in {info_file_path}", file=sys.stderr)
        return

    if sample_info is None:
        # Try with just the numeric part in case the ID is stored as an integer
        try:
            sample_id_num = int(SAMPLE_ID)
            if 'data_list' in info_file:
                sample_info = next((data for data in info_file['data_list'] 
                                  if int(data.get('sample_idx', -1)) == sample_id_num), None)
            else:
                sample_info = next((data for data in info_file 
                                  if int(data.get('sample_idx', -1)) == sample_id_num), None)
        except (ValueError, TypeError):
            pass

    if sample_info is None:
        print(f"ERROR: Could not find calibration data for ID: {SAMPLE_ID} in {info_file_path}.")
        print("Available sample IDs (first 5):")
        try:
            if 'data_list' in info_file:
                sample_ids = [str(d.get('sample_idx', 'N/A')) for d in info_file['data_list'][:5]]
            else:
                sample_ids = [str(d.get('sample_idx', 'N/A')) for d in info_file[:5]]
            print(", ".join(sample_ids) + "...")
        except Exception as e:
            print(f"Could not list sample IDs: {e}")
        return

    # Extract calibration matrices
    try:
        # Try different possible key structures
        if 'images' in sample_info and 'CAM2' in sample_info['images']:
            cam2img = np.array(sample_info['images']['CAM2']['cam2img'], dtype=np.float32)
            lidar2cam_trans = np.array(sample_info['images']['CAM2']['lidar2cam'], dtype=np.float32)
        elif 'cam2img' in sample_info:
            cam2img = np.array(sample_info['cam2img'], dtype=np.float32)
            lidar2cam_trans = np.array(sample_info['lidar2cam'], dtype=np.float32)
        else:
            # Try direct access to calibration data
            cam2img = np.array(sample_info['calib']['P2']).reshape(3, 4).astype(np.float32)
            lidar2cam_trans = np.eye(4, dtype=np.float32)
            lidar2cam_trans[:3, :4] = np.array(sample_info['calib']['Tr_velo_to_cam']).reshape(3, 4)
            
        print(f"Successfully loaded calibration data for sample {SAMPLE_ID}")
        print(f"cam2img shape: {cam2img.shape}")
        print(f"lidar2cam_trans shape: {lidar2cam_trans.shape}")
        
    except KeyError as e:
        print(f"ERROR: Missing calibration key in sample data: {e}")
        print("Available keys in sample_info:", sample_info.keys())
        if 'calib' in sample_info:
            print("Available keys in calib:", sample_info['calib'].keys())
        return

    # B. Load Image
    try:
        img = mmcv.imread(image_file_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb') 
    except Exception as e:
        print(f"ERROR: Could not load image from {image_file_path}. {e}", file=sys.stderr)
        return

    # C. Load Predicted Boxes
    try:
        with open(prediction_json_path, 'r') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load prediction JSON file from {prediction_json_path}. {e}", file=sys.stderr)
        return

    if SAMPLE_ID not in predictions:
        print(f"WARNING: No predictions found for sample ID: {SAMPLE_ID} in the JSON file. Saving image only.")
        mmcv.imwrite(mmcv.imconvert(img, 'rgb', 'bgr'), output_path)
        return

    sample_preds = predictions[SAMPLE_ID]
    if 'bboxes_3d' not in sample_preds or 'scores_3d' not in sample_preds:
        print("ERROR: Prediction JSON is missing 'bboxes_3d' or 'scores_3d' keys.", file=sys.stderr)
        return

    # Filter boxes by score threshold
    bboxes_3d_list = sample_preds['bboxes_3d']
    scores_3d_list = sample_preds['scores_3d']
    
    filtered_boxes = [
        box for box, score in zip(bboxes_3d_list, scores_3d_list) 
        if score >= score_threshold
    ]

    if not filtered_boxes:
        print(f"WARNING: No boxes passed the score threshold ({score_threshold}). Saving image only.")
        mmcv.imwrite(mmcv.imconvert(img, 'rgb', 'bgr'), output_path)
        return

    # Convert to MMDet3D structure (LiDAR frame)
    predicted_boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
    if predicted_boxes_tensor.dim() == 1:
        predicted_boxes_tensor = predicted_boxes_tensor.unsqueeze(0)
        
    pred_bboxes_3d_lidar = LiDARInstance3DBoxes(predicted_boxes_tensor)


    # --- 3. Projection and Visualization ---
    
    print(f"Found {len(predicted_boxes_tensor)} boxes (>= {score_threshold}). Projecting...")

    # Convert LiDAR Boxes to Camera Boxes (required for 3D to 2D projection)
    pred_bboxes_3d_camera = pred_bboxes_3d_lidar.convert_to(
        Box3DMode.CAM, np.linalg.inv(lidar2cam_trans)
    )

    # Prepare metadata for visualization (Camera Projection Matrix)
    input_meta = {'cam2img': cam2img} 

    # Initialize and run the visualizer
    visualizer = Det3DLocalVisualizer()
    visualizer.set_image(img)

    # Project the 3D boxes onto the 2D image plane and draw them
    visualizer.draw_proj_bboxes_3d(pred_bboxes_3d_camera, input_meta)
    
    # Save the result
    visualized_img = mmcv.imconvert(visualizer.get_image(), 'rgb', 'bgr')
    mmcv.imwrite(visualized_img, output_path)
    
    print(f"SUCCESS: Projected image saved to {output_path}")


if __name__ == '__main__':
    # --- Command Line Argument Setup ---
    parser = argparse.ArgumentParser(description='MMDet3D 3D BBox Projection Visualizer')
    parser.add_argument('--pred-file', type=str, required=True,
                        help='Path to the prediction JSON file.')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the target 2D image file (e.g., data/kitti/testing/image_2/000000.png)')
    parser.add_argument('--kitti-info', type=str, required=True,
                        help='Path to the KITTI infos PKL file (e.g., data/kitti/kitti_infos_test.pkl)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Directory to save the visualized image.')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for displaying 3D bounding boxes.')

    args = parser.parse_args()

    visualize_projected_boxes(
        info_file_path=args.kitti_info, 
        image_file_path=args.image_path, 
        prediction_json_path=args.pred_file,
        output_dir=args.out_dir,
        score_threshold=args.score_thr
    )