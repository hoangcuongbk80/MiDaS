import cv2
import numpy as np
import os

# Path to the folders
input_folder = "output"
output_folder = "uncertain"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process a pair of images
def process_images(image_left_path, image_right_path):
    # Load the images in RGB
    image_left_rgb = cv2.imread(image_left_path)
    image_right_rgb = cv2.imread(image_right_path)

    # Convert images to grayscale for computing disparity (assuming disparity requires grayscale)
    image_left_gray = cv2.cvtColor(image_left_rgb, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right_rgb, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(image_left_gray, image_right_gray)

    # Define a threshold for significant differences
    threshold = 20

    # Identify uncertain areas based on disparity differences
    uncertain_regions = np.where(disparity > threshold)

    # Remove uncertain regions from the images
    image_left_rgb[uncertain_regions] = 0
    image_right_rgb[uncertain_regions] = 0

    # Calculate mean values for non-uncertain areas
    left_mean = np.mean(image_left_rgb[image_left_rgb != 0])
    right_mean = np.mean(image_right_rgb[image_right_rgb != 0])

    # Replace values within other areas with their mean values
    image_left_rgb[image_left_rgb == 0] = left_mean
    image_right_rgb[image_right_rgb == 0] = right_mean

    return image_left_rgb, image_right_rgb

# Process all image pairs in the input folder
for file in os.listdir(input_folder):
    if file.endswith("-dpt_beit_large_512.png"):
        left_image_path = os.path.join(input_folder, file)
        right_image_path = os.path.join(
            input_folder, file.replace("-dpt_beit_large_512.png", "-dpt_swin2_large_384.png")
        )

        if os.path.isfile(right_image_path):
            modified_left, modified_right = process_images(left_image_path, right_image_path)

            # Save modified images to the uncertain folder
            output_left_path = os.path.join(output_folder, file.replace(".png", "_modified.png"))
            output_right_path = os.path.join(
                output_folder,
                file.replace("-dpt_beit_large_512.png", "-dpt_swin2_large_384_modified.png"),
            )

            cv2.imwrite(output_left_path, modified_left)
            cv2.imwrite(output_right_path, modified_right)
