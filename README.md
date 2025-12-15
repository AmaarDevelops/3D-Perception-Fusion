# yolo-midas-3d-fusion

## 🎯 Real-Time 3D Target Acquisition (Depth + Segmentation Fusion)

This project represents the final stage of the perception stack, fusing two powerful deep learning models to achieve real-time, 3D target acquisition. It identifies an object, segments it out, determines its pixel location ($C_x, C_y$), and calculates its relative distance ($C_z$) using monocular depth estimation.

The output is a complete $C_x, C_y, C_z$ coordinate set, ready to be sent to a robotic controller for grasping or manipulation tasks.

---

## ✨ Core Technology Stack

| Component | Role in Project | Output Data |
| :--- | :--- | :--- |
| **YOLOv8-Seg** | **Segmentation & Detection.** Provides a mask and pixel coordinates for the target object. | Binary Mask (0/1), $C_x, C_y$ |
| **MiDaS** | **Monocular Depth Estimation.** Provides a relative depth map for the entire scene. | Depth Map ($\text{D}_{\text{rel}}$) |
| **OpenCV/NumPy** | **Fusion Logic.** Cleans the mask and performs the matrix multiplication for fusion. | 3D Centroid ($C_x, C_y, C_z$) |

---

## 🛠️ Implementation Details

### 1. Model Synchronization
Both the YOLO segmentation mask and the MiDaS depth map are resized to the same dimensions as the input frame to ensure pixel-to-pixel correspondence.

### 2. Binary Masking
The object mask from YOLO is converted into a clean binary mask (0s and 255s) using `cv2.threshold`.

### 3. Depth Fusion (The Core Algorithm)
The final depth isolation is achieved by multiplying the depth map by a normalized (0/1) binary mask. This removes all background depth data, leaving only the depth values associated with the target object.

$$
\text{D}_{\text{masked}} = \left(\frac{\mathbf{M}_{\text{binary}}}{255}\right) \odot \mathbf{D}_{\text{MiDaS}}
$$
*Where $\mathbf{M}_{\text{binary}}$ is the 0/255 mask, and $\odot$ is the element-wise (Hadamard) product.*

### 4. 3D Centroid Calculation
The final $C_z$ value (relative distance) is calculated by taking the average of all non-zero depth values in the masked depth array.

$$
C_z = \text{Average}(\text{D}_{\text{masked}}[\text{D}_{\text{masked}} > 0])
$$

---

## ⚙️ How to Run

### Prerequisites

* Python 3.8+
* PyTorch (with CUDA support recommended for performance)
* `ultralytics` (for YOLOv8)
* `opencv-python`
* `numpy`

### Execution

``bash

python depth.py

Console Output Example

The console will output the detected 3D coordinates in real-time:

3D Grap Target :- X: 331 , Y: 363 , Z: 756.03
3D Grap Target :- X: 332 , Y: 364 , Z: 760.12
