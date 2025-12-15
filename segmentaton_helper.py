from ultralytics import YOLO
import cv2
import numpy as np


def calculate_centroid(mask):

    M = cv2.moments(mask)

    if M["m00"] == 0:
        return None,None

    # Centroid coordinates are calculated from the moments:
    # Cx = m10 / m00 , describes how the area of mask is spread horizontally
    # Cy = m01 / m00 , describes how the area of mask is spread vertically
    Cx = int(M['m10']) / int(M['m00'])
    Cy = int(M['m01']) / int(M['m00'])

    return Cx,Cy




# ---------- Calculate orientation ------

def calculate_orientation(mask):
    M = cv2.moments(mask)

    if M["m00"] == 0: # m00 the total area / pixel intensities of the entire mask
        return None

    mu_20 = M['mu20']
    mu_02 = M['mu02']
    mu_11 = M['mu11']

    # Formula for Orientation (theta) based on central moments:
    # theta = 0.5 * arctan2( 2 * mu_11, (mu_20 - mu_02) )

    angle_rad = 0.5 * np.arctan2(2*mu_11,mu_20 - mu_02)

    angle_deg = np.degrees(angle_rad) # Converting into degrees from radians

    if angle_deg < 0: # Adjust the angle to be intuitive b/w 0 and 180
        angle_deg += 180

    return angle_deg



# ---- Analyse mask metric ----


def analyse_metric_mask(mask):
    cX,cY = calculate_centroid(mask)
    angle = calculate_orientation(mask)

    if cX is None:
        return {"Centroid_X" : None , "Centriod_Y" : None, "Angle_degrees" : None}

    return {
        "Centroid_X" : cX,
        "Centroid_Y" : cY,
        "Angle_Degrees" : angle
    }



# ----------- Robot control function ------------

def find_center_and_speed(cx_position,frame_width):
    if cx_position is None:
        return 0.0

    target_x = frame_width / 2 # The target (half of the frame)
    kp_gain = 0.05

    max_command = 100 # Maximum speed of the robotic motor
    error_x = cx_position - target_x # How off it is from the target

    command_speed = kp_gain * error_x # the more off is it from cX, the faster it shall move

    command_speed = np.clip(command_speed,-max_command,max_command)

    return command_speed


# --- Live Analysis of the robot ---
def run_live_analysis():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Couldn't find camera")
        return

    model = YOLO('yolov8n-seg.pt')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Extract the width of the current camera capture

    while True:
        ret,frame = cap.read()

        if not ret:
            break

        results = model(frame,stream=False,verbose=False,conf=0.5)

        current_mask = np.zeros(frame.shape[:2],dtype=np.uint8)

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
                metrics = analyse_metric_mask(best_mask)
                cX = metrics['Centroid_X']

                motor_control_command = find_center_and_speed(cX,frame_width)

                print(f'Error :- {cX - frame_width/2:.2f} | Command :- {motor_control_command:.2f}')

                # Visualization
                if cX is not None:
                    cv2.line(frame,(frame_width // 2 , 0) , (frame_width // 2, frame.shape[0]),(0,255,0))

                    cv2.circle(frame,(int(cX),int(metrics['Centroid_Y'])) , 5 , (0,255,0),-1)
                    cv2.putText(frame,f"CMD : {motor_control_command:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            else:
                motor_control_command = 0.0
                print('Object Lost, command : 0.0')

            cv2.imshow('Robot Tracking System',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    run_live_analysis()







