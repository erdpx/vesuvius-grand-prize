import os
from PIL import Image
import random
import numpy as np

Image.MAX_IMAGE_PIXELS = None

region_folder_names = []

def process_folder(dataset_path, num_regions=15):
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")

    inklabels_file = f"{dataset_path}/inklabels.png"
    mask_file = f"{dataset_path}/mask.png"
    layers_folder = f"{dataset_path}/layers/"

    print("Opening inklabels and mask images")
    inklabels_img = Image.open(inklabels_file).convert('L')
    mask_img = Image.open(mask_file)

    width, height = inklabels_img.size
    assert width >= 1024 and height >= 1024, "Images must be at least 512x512 in size"

    selected_regions = []

    def create_unique_folder(base_path, base_name):
        counter = 0
        new_path = os.path.join(base_path, base_name)
        while os.path.exists(new_path):
            counter += 1
            new_path = os.path.join(base_path, f"{base_name}-{counter}")
        os.makedirs(new_path, exist_ok=True)
        return new_path

    def contains_roi(inklabels_arr, region, threshold=0.15, brightness_threshold=200):
        region_arr = inklabels_arr[region[1]:region[3], region[0]:region[2]]
        white_pixels = np.sum(region_arr >= brightness_threshold)
        total_pixels = region_arr.size
        white_ratio = white_pixels / total_pixels
        return white_ratio >= threshold

    inklabels_arr = np.array(inklabels_img)

    for i in range(num_regions):
        print(f"Selecting region {i+1} of {num_regions}")
        attempt_count = 0
        while attempt_count < 100:
            x = random.randint(0, width - 1024)
            y = random.randint(0, height - 1024)
            region = (x, y, x + 1024, y + 1024)

            if contains_roi(inklabels_arr, region, threshold=0.15, brightness_threshold=200) and region not in selected_regions:
                selected_regions.append(region)
                print(f"Region {i+1} selected at ({x}, {y})")
                break
            attempt_count += 1

        if attempt_count == 100:
            print(f"Could not find a suitable region for {i+1}, moving to the next region")
            continue

        rotation_angle = random.choice([90, 180])
        flip_type = random.choice(['horizontal', 'vertical'])

        def apply_transformation(image):
            rotated_image = image.rotate(rotation_angle)
            if flip_type == 'horizontal':
                return rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == 'vertical':
                return rotated_image.transpose(Image.FLIP_TOP_BOTTOM)
            return rotated_image

        region_folder = create_unique_folder(dataset_path, f"{dataset_name}-region_{i}")
        print(f"Created region folder: {region_folder}")
        layers_region_folder = os.path.join(region_folder, "layers")
        os.makedirs(layers_region_folder, exist_ok=True)

        inklabels_crop = inklabels_img.crop(region)
        mask_crop = mask_img.crop(region)

        inklabels_crop = apply_transformation(inklabels_crop)
        mask_crop = apply_transformation(mask_crop)

        inklabels_crop.save(os.path.join(region_folder, "inklabels.png"))
        mask_crop.save(os.path.join(region_folder, "mask.png"))
        print(f"Saved cropped and transformed images for region {i+1}")

        for layer_file in os.listdir(layers_folder):
            if layer_file.lower().endswith(".tif"):
                layer_path = os.path.join(layers_folder, layer_file)
                layer_img = Image.open(layer_path)
                layer_crop = layer_img.crop(region)
                layer_crop = apply_transformation(layer_crop)
                cropped_layer_file = f"{os.path.splitext(layer_file)[0]}.tif"
                layer_crop.save(os.path.join(layers_region_folder, cropped_layer_file))
                print(f"Processed and saved transformed layer file: {cropped_layer_file}")

        region_folder_names.append(os.path.basename(region_folder))

    print(f"Completed processing dataset: {dataset_name}")

def combine_regions_complete(dataset_path):
    if len(region_folder_names) < 2:
        print("Not enough regions to combine. Please process more regions first.")
        return

    def stitch_images(image1, image2):
        # Ensure both images are in the same mode
        if image1.mode != image2.mode:
            if "A" in image1.mode:  # If image1 has an alpha channel
                image1 = image1.convert("RGBA")
            else:
                image1 = image1.convert("RGB")
            if "A" in image2.mode:  # If image2 has an alpha channel
                image2 = image2.convert("RGBA")
            else:
                image2 = image2.convert("RGB")

        total_width = image1.size[0] + image2.size[0]
        max_height = max(image1.size[1], image2.size[1])
        combined_image = Image.new(image1.mode, (total_width, max_height))
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (image1.size[0], 0))
        return combined_image
    
    def process_file(folder1, folder2, file_name, output_folder):
            # Create a subdirectory for layers if needed
            if "layers" in file_name:
                layers_output_folder = os.path.join(output_folder, "layers")
                os.makedirs(layers_output_folder, exist_ok=True)
                save_path = os.path.join(layers_output_folder, os.path.basename(file_name))
            else:
                save_path = os.path.join(output_folder, file_name)

            image1 = Image.open(os.path.join(dataset_path, folder1, file_name))
            image2 = Image.open(os.path.join(dataset_path, folder2, file_name))
            combined_image = stitch_images(image1, image2)
            combined_image.save(save_path)

    shuffled_regions = random.sample(region_folder_names, len(region_folder_names))
    for i in range(0, len(shuffled_regions), 2):
        if i + 1 < len(shuffled_regions):
            folder1 = shuffled_regions[i]
            folder2 = shuffled_regions[i + 1]
            output_folder = os.path.join(dataset_path, f"combined_{folder1}_{folder2}")
            os.makedirs(output_folder, exist_ok=True)

            process_file(folder1, folder2, "inklabels.png", output_folder)
            process_file(folder1, folder2, "mask.png", output_folder)

            layers_folder = os.path.join(dataset_path, folder1, "layers")
            for layer_file in os.listdir(layers_folder):
                if layer_file.lower().endswith(".tif"):
                    process_file(folder1, folder2, os.path.join("layers", layer_file), output_folder)

            print(f"Combined regions saved in {output_folder}")

parent_folder = "/home/sean/Desktop/new_cuts"  # Replace with your actual parent folder path

for folder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, folder_name)
    if os.path.isdir(subfolder_path):
        print(f"Processing folder: {subfolder_path}")
        process_folder(subfolder_path)
        combine_regions_complete(subfolder_path)

print("All folders processed successfully.")
