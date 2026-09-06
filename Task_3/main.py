# Task 3: Obstacle and Pothole Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('Task_3/input', exist_ok=True)
os.makedirs('Task_3/output', exist_ok=True)

files = [f for f in os.listdir('/content') if f.endswith(('.jpg', '.png', '.jpeg'))]

if files:
    raw_img_path = os.path.join('/content', files[0])
    img = cv2.imread(raw_img_path)
    
    # Save input image
    cv2.imwrite('Task_3/input/input_task3.jpg', img)

    # Create a copy for processing output
    output_img = img.copy()

    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_img, f"({x},{y})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.putText(output_img, f"Total Obstacles/Potholes: {count}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save output image
    cv2.imwrite('Task_3/output/output_task3.jpg', output_img)

    # Display Input and Output Images sequentially
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Task 3: Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title('Task 3: Output Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
