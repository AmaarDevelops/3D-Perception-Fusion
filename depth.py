import numpy as np
import torch
import cv2
from ultralytics import YOLO
import time
from segmentaton_helper import calculate_centroid,run_live_analysis,find_center_and_speed

yolo = YOLO('yolov8n-seg.pt')

model_type = 'MiDaS_small'

midas = torch.hub.load('intel-isl/MiDaS',model_type)

device = "cuda" if torch.cuda.is_available() else "cpu"

midas.to(device)
midas.eval()

# Load transforms to apply resize and normalize the images
midas_transforms = torch.hub.load('intel-isl/MiDaS',"transforms")

# load relevant transforms
if model_type == "DPT_Large" or model_type == "DPT_hybrid":
    transforms = midas_transforms.dpt_transform
else:
    transforms = midas_transforms.small_transform



cap = cv2.VideoCapture(0)

while True:


    ret,frame = cap.read()

    if not ret:
        print('Error. could not read from your camera frame.')
        break

    results = yolo(frame,stream=False,verbose=False,conf=0.5)

    # Midas setup

    start = time.time()

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    input_batch = transforms(img).to(device)

    with torch.no_grad():
        predictions = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            predictions.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy() # Calculate the depth_map in numpy array form



    # Best mask for YOLO
    if results and results[0].masks is not None:
        mask = results[0].masks.data.cpu().numpy()

        largest_mask_area = 0
        best_mask = None

        for mask_tensor in mask:
            mask = cv2.resize(mask_tensor,(frame.shape[1],frame.shape[0]))
            area = np.sum(mask)

            if area > largest_mask_area:
                largest_mask_area = area
                best_mask = mask * 255


    if best_mask is not None:
        mask_uint8 = best_mask.astype(np.uint8)

        ret_val,binary_mask = cv2.threshold(mask_uint8,127,255,cv2.THRESH_BINARY)

        cX,cY = calculate_centroid(binary_mask)

        masked_depth = (binary_mask / 255.0) * depth_map

        non_zero = masked_depth[masked_depth > 0]

        if len(non_zero) > 0:
            cZ = np.mean(non_zero)

            print(f"3D Grap Target :-  X: {cX} , Y: {cY} , Z: {cZ}")

            cv2.circle(frame,(int(cX),int(cY)),10,(0,255,0),-1)

            depth_text = f'Depth (Cz) : {cZ:.3f}'

            cv2.putText(frame,depth_text,(int(cX+20),int(cY)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
            cv2.imshow('3D target acquistion',frame)

        else:
            cZ = 0.0
            print("Target not found or the msak is empty")

        if cv2.waitKey(20) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()













