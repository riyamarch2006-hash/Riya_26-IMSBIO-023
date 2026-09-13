import os
os.makedirs("Task_3/input", exist_ok=True)
os.makedirs("Task_3/output", exist_ok=True)
import os
import cv2
import numpy as np

# Folder Paths
input_dir = "Task_3/input"
output_dir = "Task_3/output"
os.makedirs(output_dir, exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

print(f"Total Images Found for Task 3: {len(image_files)}\n")

def detect_obstacles_and_potholes(img):
    """Detects potholes (white circles) and obstacles (colored cylinders/crates)
    while ignoring continuous lane boundary lines."""
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. COLOR MASKS FOR OBSTACLES (Yellow, Green, Blue)
    # Yellow Cylinders & Crates
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Green Cylinders
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Blue Cylinders
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine obstacle masks
    obstacle_mask = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_green, mask_blue))
    
    # 2. POTHOLE MASK (Pure White Circular/Oval Patches)
    # White threshold for bright patches
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Combine all potential detections
    combined_mask = cv2.bitwise_or(obstacle_mask, white_mask)
    
    # Clean noise using Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = img.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Area filter for small noise or large background
        if 80 < area < 40000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # Filter out thin long lane boundary lines (lane lines have extreme aspect ratio or low solidity)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Valid obstacles/potholes are compact shapes (not thin winding lines)
            if solidity > 0.55 and (0.2 < aspect_ratio < 4.0):
                # Ignore UI elements at bottom corner if any
                if y > img.shape[0] - 30 and x > img.shape[1] - 120:
                    continue
                
                count += 1
                
                # Draw Bounding Box
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Label Pixel Coordinates (X, Y)
                coord_text = f"({x},{y})"
                cv2.putText(output, coord_text, (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                
    # Overlay Total Detected Count
    banner_text = f"Total Detected: {count}"
    cv2.putText(output, banner_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return output, count

# Batch Execution
for filename in image_files:
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        continue
    
    processed_img, total_found = detect_obstacles_and_potholes(img)
    
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, processed_img)
    print(f"Processed: {filename} | Detected: {total_found}")

print("\nTask 3 pipeline executed successfully!")
