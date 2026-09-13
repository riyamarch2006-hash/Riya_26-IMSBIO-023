#task 2: Lane Detection and drivable area
import os

# Set directory paths
input_dir = "Task_2/input"
output_dir = "Task_2/output"

# Create directories if they don't exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("Folders created successfully!")
import os
import cv2
import numpy as np

# Folder paths
input_dir = "Task_2/input"
output_dir = "Task_2/output"
os.makedirs(output_dir, exist_ok=True)

# Image list loading
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

print(f"Total Images Found: {len(image_files)}\n")

def preprocess_frame(img):
    """Isolate lane lines under varying light and tunnel conditions using HSV and LAB color spaces."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # White lane detection (L-channel from LAB + HSV)
    l_channel = lab[:, :, 0]
    l_thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Yellow lane detection (HSV space)
    lower_yellow = np.array([13, 70, 80], dtype=np.uint8)
    upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    combined_binary = cv2.bitwise_or(l_thresh, yellow_mask)
    
    # Sobel Gradient (Edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else abs_sobelx)
    sobel_binary = cv2.threshold(scaled_sobel, 35, 255, cv2.THRESH_BINARY)[1]
    
    # Final combined binary map
    final_mask = cv2.bitwise_or(combined_binary, sobel_binary)
    
    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

def process_lane_detection(image):
    h, w = image.shape[:2]
    binary = preprocess_frame(image)
    
    # Dynamic Region of Interest (Masking dashboard & sky)
    roi_mask = np.zeros_like(binary)
    y_top = int(h * 0.52)
    y_bottom = int(h * 0.90)  # Excludes hood/dashboard at the bottom
    
    roi_vertices = np.array([[
        (int(w * 0.05), y_bottom),
        (int(w * 0.42), y_top),
        (int(w * 0.58), y_top),
        (int(w * 0.95), y_bottom)
    ]], dtype=np.int32)
    
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_binary = cv2.bitwise_and(binary, roi_mask)
    
    # Bird's Eye View Warp Matrix
    src = np.float32([
        [w * 0.43, h * 0.55],
        [w * 0.57, h * 0.55],
        [w * 0.90, h * 0.88],
        [w * 0.10, h * 0.88]
    ])
    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.80, h],
        [w * 0.20, h]
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(masked_binary, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Histogram peak detection
    histogram = np.sum(warped[int(h / 2):, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window Setup
    nwindows = 9
    window_height = int(h / nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current, rightx_current = leftx_base, rightx_base
    margin = int(w * 0.08)
    minpix = 40
    
    left_lane_inds, right_lane_inds = [], []
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]
        
        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)
        
        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    ploty = np.linspace(0, h - 1, h)
    
    # Fit 2nd degree polynomial curve
    if len(left_lane_inds) > 50 and len(right_lane_inds) > 50:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
        
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Warp back overlay onto original image
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = cv2.merge((warp_zero, warp_zero, warp_zero))
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw green drivable region and red lane boundaries
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), False, (0, 0, 255), 10)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), 10)
        
        new_warp = cv2.warpPerspective(color_warp, Minv, (w, h))
        return cv2.addWeighted(image, 1.0, new_warp, 0.35, 0)
    else:
        # Contour Fallback
        contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:
                hull = cv2.convexHull(c)
                cv2.fillConvexPoly(overlay, hull, (0, 255, 0))
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# Batch Execution
for filename in image_files:
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        continue
    
    result = process_lane_detection(img)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, result)
    print(f"Processed: {filename}")

print("\nAll 10 images processed cleanly!")
