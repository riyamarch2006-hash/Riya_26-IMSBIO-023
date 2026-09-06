#task 2: Lane Detection and drivable area 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory structure
os.makedirs('Task_2/input', exist_ok=True)
os.makedirs('Task_2/output', exist_ok=True)

# Find uploaded raw image
files = [f for f in os.listdir('/content') if f.endswith(('.jpg', '.png', '.jpeg'))]

if files:
    raw_img_path = os.path.join('/content', files[0])
    img = cv2.imread(raw_img_path)
    
    # Save input image
    cv2.imwrite('Task_2/input/input_task2.jpg', img)

    height, width = img.shape[:2]

    # OpenCV Pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    overlay = img.copy()
    cv2.fillPoly(overlay, polygon, (0, 255, 0))
    output = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Save output image
    cv2.imwrite('Task_2/output/output_task2.jpg', output)

    # Display Input and Output Images side-by-side inline
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Task 2: Input Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Task 2: Output Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
