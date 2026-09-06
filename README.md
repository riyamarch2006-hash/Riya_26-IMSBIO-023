# UGV-DTU Recruitment Round 3 — Software Department Task Logbook

**Name:** Riya  
**Roll No / ID:** Riya_26-IMSBIO-023  
**Department:** Software Department  
**Submission Deadline:** Monday, 7th September, 5:30 PM  

---

## 📌 Executive Summary
This logbook documents the complete technical workflow, problem-solving methodology, and algorithmic logic applied during the UGV-DTU Round 3 Software Recruitment Assignment. The tasks cover version control using GitHub, classical Computer Vision (CV) techniques using OpenCV and NumPy for lane detection and obstacle thresholding, and spatial path planning using waypoints.

---

## 🛠️ Task Breakdown & Technical Log

### Task 1: GitHub Repository & Workspace Setup
* **Objective:** Establish structured repo layout, master basic Git operations, and maintain a running logbook.
* **Approach & Steps Followed:**
  1. Configured GitHub repository naming convention (`Riya_26-IMSBIO-023`).
  2. Structured repository directories with separate modules: `Task_2/`, `Task_3/`, `Task_4/` containing `main.py` scripts and output image subfolders.
* **Challenges Faced:** Initial mobile-device workspace limitations without a native terminal interface.
* **Solution:** Streamlined workflow using Google Colab's cloud execution platform alongside GitHub's web interface for code updates and commits.

---

### Task 2: Lane Detection & Drivable Area Overlay
* **Objective:** Identify asphalt road lane boundaries and generate a semi-transparent green overlay over the drivable path.
* **Computer Vision Pipeline:**
  1. **Grayscale & Smoothing:** Converted input image from BGR to Grayscale (`cv2.cvtColor`) and applied Gaussian Blur (`cv2.GaussianBlur`) with a $5\times5$ kernel to remove high-frequency noise.
  2. **Edge Detection:** Used Canny Edge Detection (`cv2.Canny`) with thresholds $(50, 150)$ to extract intensity gradients.
  3. **Region of Interest (ROI) Masking:** Created a polygonal mask focusing strictly on the bottom road area, eliminating background noise (sky, trees, and roadside elements).
  4. **Hough Line Transform:** Employed Probabilistic Hough Lines (`cv2.HoughLinesP`) to fit line segments to edge points.
  5. **Drivable Region Overlay:** Used `cv2.fillPoly()` and `cv2.addWeighted()` to superimpose a green polygon over the driving corridor.
* **AI Tool Integration:** Used AI (Gemini) to structure OpenCV syntax for mathematical ROI polygon mapping and blending parameters.
* **Results:** Successfully highlighted drivable lane corridor with a visual boundary overlay saved to `Task_2/output/output_task2.jpg`.

---

### Task 3: Obstacle & Pothole Detection
* **Objective:** Detect white circular blobs/obstacles on the track, draw bounding boxes around them, annotate spatial coordinates $(x, y)$, and print total counts.
* **Detection Pipeline:**
  1. **Thresholding:** Applied Binary Thresholding (`cv2.threshold`) at intensity value `180` to separate bright obstacle regions from darker road asphalt.
  2. **Contour Extraction:** Utilized `cv2.findContours()` with `RETR_EXTERNAL` mode to extract outer obstacle boundaries.
  3. **Noise Filtering:** Screened contours using `cv2.contourArea(cnt) > 20` to exclude small pixel noise.
  4. **Bounding Annotation:** Derived bounding rectangles via `cv2.boundingRect()` and overlaid pixel coordinates $(x, y)$ along with total item counts using `cv2.putText()`.
* **Iterative Changes:** Adjusted threshold values dynamically to ensure yellow cone obstacles and bright circular potholes were correctly segmented without false-positive background noise.
* **Results:** Total obstacle count successfully displayed and output saved to `Task_3/output/output_task3.jpg`.

---

### Task 4: Aerial Path Planning
* **Objective:** Calculate and overlay a collision-free path for one full loop around the aerial track without intersecting obstacles.
* **Path Planning Methodology:**
  1. **Waypoint / Checkpoint Method:** Analyzed track geometry to extract discrete 2D spatial coordinate waypoints along the clear centerline of the road course.
  2. **Path Construction:** Connected sequential waypoints into a closed loop using `cv2.polylines()` and rendered yellow circular waypoints using `cv2.circle()`.
* **Concepts Learnt:** Evaluated classical path planning concepts such as $A^*$ (A-Star) graph search and $RRT^*$ (Rapidly-exploring Random Tree Star) sampling algorithms for complex dynamic obstacle avoidance.
* **Results:** Clear loop path generated avoiding track bounds, saved to `Task_4/output/output_task4.jpg`.

---

## 🤖 AI Usage Transparency Statement
AI tools (Gemini) were utilized as a collaborative thought partner to:
* Debug environment path errors (`NoneType` shape errors during image reads).
* Formulate OpenCV algorithmic approaches (Thresholding, Contours, Canny Edge Detection).
* Learn the theoretical mechanics behind Computer Vision pipelines for technical interview preparation.

---

## 📊 Final Conclusions
All tasks were completed, verified, and saved to their respective directories within the public GitHub repository. The repository is ready for evaluation ahead of the recruitment interview.
