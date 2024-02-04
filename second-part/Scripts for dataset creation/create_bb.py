import cv2
import os
import argparse
import json

# Constants for minimum contour dimensions
MIN_WIDTH = 128
MIN_HEIGHT = 384

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Process label images in all folders within a specified directory.')
parser.add_argument('--parent_path', type=str, required=True, help='Parent path containing multiple label directories')
args = parser.parse_args()

PARENT_PATH = args.parent_path

# Iterate over each folder in the parent directory
for segment_id in os.listdir(PARENT_PATH):
    LABELS_PATH = os.path.join(PARENT_PATH, segment_id)
    if not os.path.isdir(LABELS_PATH):
        continue  # Skip if it's not a directory

    print(f"Processing folder: {LABELS_PATH}")

    # Construct the path to the label image
    img_path = os.path.join(LABELS_PATH, f"inklabels.png")
    if not os.path.exists(img_path):
        print(f"Label image not found at {img_path}, skipping folder.")
        continue

    # Read the image
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Failed to load image from {img_path}")
        continue

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to draw bounding boxes and JSON data
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bbox_list = []  # List to store bbox for txt file
    coco_annotations = []  # List to store annotations for JSON

    # Create a new directory for saving results
    save_dir = os.path.join(LABELS_PATH, f"{segment_id}_results")
    os.makedirs(save_dir, exist_ok=True)

    for cnt, annotation_id in zip(contours, range(1, len(contours) + 1)):
        x1, y1, w, h = cv2.boundingRect(cnt)

        # Skip contours smaller than specified dimensions
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        # Determine square bounding box
        side_length = max(w, h)
        side_length = ((side_length + max(MIN_WIDTH, MIN_HEIGHT)) // max(MIN_WIDTH, MIN_HEIGHT)) * max(MIN_WIDTH, MIN_HEIGHT)

        square_x1 = max(0, x1 + w // 2 - side_length // 2)
        square_y1 = max(0, y1 + h // 2 - side_length // 2)
        bbox = (square_x1, square_y1, square_x1 + side_length, square_y1 + side_length)

        # Draw the bounding box and add to list
        cv2.rectangle(color_img, (square_x1, square_y1), (square_x1 + side_length, square_y1 + side_length), (0, 255, 0), 2)
        bbox_list.append(bbox)

        # Add annotation to JSON
        coco_annotations.append({
            "id": annotation_id,
            "image_id": int(segment_id),
            "category_id": 1,
            "bbox": [square_x1, square_y1, side_length, side_length],
            "area": side_length * side_length,
            "segmentation": [],
            "iscrowd": 0
        })

    # Save the image with bounding boxes
    image_file_path = os.path.join(save_dir, f"{segment_id}_with_bboxes.png")
    cv2.imwrite(image_file_path, color_img)

    # Save the bounding boxes to a txt file
    bbox_txt_file_path = os.path.join(save_dir, f"{segment_id}_bboxes.txt")
    with open(bbox_txt_file_path, 'w') as file:
        for bbox in bbox_list:
            file.write(f"{bbox}\n")

    # Save the JSON file in COCO format
    coco_json = {
        "images": [{"id": int(segment_id), "width": img.shape[1], "height": img.shape[0], "file_name": f"{segment_id}.png"}],
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "object"}]
    }
    json_file_path = os.path.join(save_dir, f"{segment_id}_annotations.json")
    with open(json_file_path, 'w') as file:
        json.dump(coco_json, file, indent=4)

    print(f"Image with bounding boxes saved to {image_file_path}")
    print(f"Bounding boxes saved to {bbox_txt_file_path}")
    print(f"Bounding boxes in COCO format saved to {json_file_path}")

print("All folders processed.")
