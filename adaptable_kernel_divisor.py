import cv2
import numpy as np
import os

def determine_optimal_kernel_size(grayscale_image, divisor=30):
    # Analyze the image to determine an appropriate kernel size
    height, width = grayscale_image.shape
    average_dimension = (height + width) // 2
    
    # Experiment with different divisors
    optimal_kernel_size = average_dimension // divisor
    
    # Ensure the kernel size is odd and at least 3
    return max(3, optimal_kernel_size | 1)

def remove_hair(image_path, divisor=30):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Determine optimal kernel size
    kernel_size = determine_optimal_kernel_size(grayscale, divisor)
    print(f"Using kernel size: {kernel_size} with divisor: {divisor}")

    # Step 2: Apply the Blackhat operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

    # Step 3: Thresholding to create a binary mask
    _, binary_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Step 4: Inpainting to remove hair
    inpainted_image = cv2.inpaint(image, binary_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inpainted_image

# Input and output directories
input_directory = 'trial_images'
output_directory = 'adaptable_kernel_divisor_output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each image in the input directory
for image_filename in os.listdir(input_directory):
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        image_path = os.path.join(input_directory, image_filename)
        
        # Experiment with different divisors
        for divisor in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print(f"Processing {image_filename} with divisor: {divisor}")
            result_image = remove_hair(image_path, divisor=divisor)
            
            # Save the result image
            output_filename = f'{os.path.splitext(image_filename)[0]}_divisor_{divisor}.jpg'
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, result_image)

print("Processing complete. Images saved in the 'adaptable_kernel_divisor_output' directory.")
