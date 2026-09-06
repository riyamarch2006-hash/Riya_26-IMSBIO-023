# Task 4: Aerial Path Planning
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('Task_4/input', exist_ok=True)
os.makedirs('Task_4/output', exist_ok=True)

files = [f for f in os.listdir('/content') if f.endswith(('.jpg', '.png', '.jpeg'))]

if files:
    raw_img_path = os.path.join('/content', files[0])
    img = cv2.imread(raw_img_path)
    
    # Save input image
    cv2.imwrite('Task_4/input/input_task4.jpg', img)

    output_img = img.copy()
    h, w, _ = output_img.shape
    waypoints = np.array([
        [int(w * 0.25), int(h * 0.75)],
        [int(w * 0.25), int(h * 0.25)],
        [int(w * 0.75), int(h * 0.25)],
        [int(w * 0.75), int(h * 0.75)],
        [int(w * 0.25), int(h * 0.75)]
    ], np.int32)

    cv2.polylines(output_img, [waypoints], isClosed=True, color=(255, 0, 0), thickness=4)
    for pt in waypoints:
        cv2.circle(output_img, tuple(pt), 6, (0, 255, 255), -1)

    # Save output image
    cv2.imwrite('Task_4/output/output_task4.jpg', output_img)

    # Display Input and Output Images sequentially
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Task 4: Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title('Task 4: Output Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
