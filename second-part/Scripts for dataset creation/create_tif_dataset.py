import os
import argparse
from PIL import Image

# Constants
START_TIFF_FILE = 15
END_TIFF_FILE = 50

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Cut out areas from multiple image files based on bounding boxes.')
parser.add_argument('--parent_path', type=str, required=True, help='Parent path containing multiple label directories')
args = parser.parse_args()

PARENT_PATH = args.parent_path

# Process each folder in the parent directory
for folder_name in os.listdir(PARENT_PATH):
    folder_path = os.path.join(PARENT_PATH, folder_name)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directories

    print(f"Processing folder: {folder_path}")

    # Paths to additional images
    mask_img_path = os.path.join(folder_path, 'mask.png')
    inklabels_img_path = os.path.join(folder_path, 'inklabels.png')
    inklabels_raw_img_path = os.path.join(folder_path, 'inklabels_raw.png')

    # Load additional images if they exist
    mask_img = Image.open(mask_img_path) if os.path.exists(mask_img_path) else None
    inklabels_img = Image.open(inklabels_img_path) if os.path.exists(inklabels_img_path) else None
    inklabels_raw_img = Image.open(inklabels_raw_img_path) if os.path.exists(inklabels_raw_img_path) else None

    # Read bounding boxes from JSON file
    json_file_path = os.path.join(folder_path, f"{folder_name}_results", f"{folder_name}_annotations.json")
    if not os.path.exists(json_file_path):
        print(f"JSON file not found at {json_file_path}, skipping folder.")
        continue

    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)

    # Process each bounding box
    for annotation in coco_data["annotations"]:
        bbox = annotation["bbox"]
        bbox_save_dir = os.path.join(folder_path, f"{folder_name}_results", "cutouts", f"bbox_{annotation['id']}")
        os.makedirs(bbox_save_dir, exist_ok=True)

        # Cutout from additional images
        for img, img_name in zip([mask_img, inklabels_img, inklabels_raw_img], ['mask', 'inklabels', 'inklabels_raw']):
            if img:
                cutout = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                cutout_save_path = os.path.join(bbox_save_dir, f"{img_name}_cutout.png")
                cutout.save(cutout_save_path)

        # Process TIFF files from 15 to 50 and retain their format
        layers_save_dir = os.path.join(bbox_save_dir, 'layers')
        os.makedirs(layers_save_dir, exist_ok=True)

        for i in range(START_TIFF_FILE, END_TIFF_FILE + 1):
            tiff_file_path = os.path.join(folder_path, 'layers', f"{i:02d}.tif")
            if not os.path.exists(tiff_file_path):
                print(f"TIFF file {tiff_file_path} not found.")
                continue

            tiff_img = Image.open(tiff_file_path)
            tiff_cutout = tiff_img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            tiff_cutout_save_path = os.path.join(layers_save_dir, f"layer_{i:02d}.tif")
            tiff_cutout.save(tiff_cutout_save_path, format='TIFF')

    print(f"Finished processing folder: {folder_path}")

print("All folders processed.")

