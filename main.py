import cv2
import time
import numpy as np
from Sign_Detector import HandSignDetector

def main():
    cap = cv2.VideoCapture(1)
    detector = HandSignDetector()
    
    # UI State variables
    mode = "DETECT" 
    recording_label = ""
    recording_count = 0
    REQUIRED_SAMPLES = 50 

    print("--- Controls ---")
    print("'r': Start Recording a new sign")
    print("'s': Save data to file")
    print("'q': Quit")
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        results = detector.process_frame(frame)

        # UI Text
        cv2.putText(frame, f"Mode: {mode}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            
            # 1. Calculate ONE bounding box for ALL hands
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                detector.mp_draw.draw_landmarks(
                    frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS,
                    detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    detector.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # Update box boundaries
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y

            # Add padding
            x_min -= 20; y_min -= 20; x_max += 20; y_max += 20
            
            # 2. Extract combined landmarks
            # (The detector now handles combining 1 or 2 hands into a single data list)
            lm_list = detector.extract_landmarks(results)

            if mode == "DETECT":
                sign, confidence = detector.predict(lm_list)
                
                # Intelligent Hiding
                if sign == "Nothing" or confidence < 50:
                    pass # Do nothing
                else:
                    # Draw ONE big box
                    color = (255, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.rectangle(frame, (x_min, y_min - 45), (x_max, y_min), color, cv2.FILLED)
                    label_text = f"{sign} {int(confidence)}%"
                    cv2.putText(frame, label_text, (x_min + 5, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif mode == "RECORD":
                if recording_count < REQUIRED_SAMPLES:
                    detector.add_data(lm_list, recording_label)
                    recording_count += 1
                    
                    cv2.putText(frame, f"RECORDING: {recording_label}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Progress bar
                    bar_width = int((recording_count / REQUIRED_SAMPLES) * 200)
                    cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(frame, (10, 60), (210, 80), (255, 255, 255), 2)
                else:
                    mode = "DETECT"
                    detector.train_model()
                    print(f"Finished recording {recording_label}.")
                    recording_label = ""
                    recording_count = 0

        cv2.imshow("SignBridge 2.0 - Combined Hands", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            detector.save_data()
        elif key == ord('r'):
            recording_label = input("Enter name for the new sign: ")
            print(f"Get ready to pose for '{recording_label}' in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            print("GO!")
            mode = "RECORD"

    cap.release()
    cv2.destroyAllWindows()

# Run the app
# Set-ExecutionPolicy Unrestricted -Scope Process
# .\venv\Scripts\Activate.ps1

if __name__ == "__main__":
    main()