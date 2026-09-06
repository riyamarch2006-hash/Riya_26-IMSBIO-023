import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Left panel folder mein se automatic .jpg/.png image dhoondne ke liye:
files = [f for f in os.listdir('/content') if f.endswith(('.jpg', '.png', '.jpeg'))]

if not files:
    print("❌ Koi image nahi mili! Kripya left panel 📁 mein image upload karein.")
else:
    # Sabse pehli uploaded image pick karega
    img_path = os.path.join('/content', files[0])
    print(f"✅ Image mil gayi: {img_path}")
    
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI Masking
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Overlay & Lines
    overlay = img.copy()
    cv2.fillPoly(overlay, polygon, (0, 255, 0))
    output_task2 = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_task2, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Display & Save
    cv2.imwrite('output_task2.jpg', output_task2)
    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(output_task2, cv2.COLOR_BGR2RGB))
    plt.title("Task 2: Lane Detection")
    plt.axis('off')
    plt.show()
