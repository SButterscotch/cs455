import cv2
import datetime
import time

# Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = 'gamer_rage_output.avi'
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

rage_level = 0
max_rage = 100
motion_timeout = 10  # If still for 10s, assume you've left
text_display_time = 7  # Display message 3 seconds before stopping
last_motion_time = time.time()
recording_active = False
away_from_screen = False  

print("Gamer Rage Detector started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame.")
        break

    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    fgmask = fgbg.apply(gray_blurred)
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_motion = False
    total_movement = 0  

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  
            continue

        frame_motion = True
        total_movement += area  
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  

    # Adjust rage level based on movement
    if frame_motion:
        last_motion_time = time.time()
        away_from_screen = False  
        rage_level = min(max_rage, int(total_movement / 2000))  

        if not recording_active:
            recording_active = True
            print("Movement detected! Recording started.")

    else:
        time_since_last_motion = time.time() - last_motion_time

        # Show "Poor guy needed a break" at 7s of no motion
        if text_display_time <= time_since_last_motion < motion_timeout:
            away_from_screen = True
            print("Poor guy needed a break.")

        # Stop recording at 10s of no motion
        if time_since_last_motion >= motion_timeout:
            rage_level = max(0, rage_level - 5)
            if recording_active:
                print("No movement detected. Stopping recording.")
                recording_active = False
            
    # Rage expressions
    if rage_level < 25:
        rage_expression = "?"
    elif rage_level < 50:
        rage_expression = "!?"
    elif rage_level < 75:
        rage_expression = "!??"
    else:
        rage_expression = "!!??"

    # Rage meter display
    meter_height = int((rage_level / max_rage) * (frame_height - 50))
    cv2.rectangle(frame, (frame_width - 50, frame_height - meter_height), (frame_width - 30, frame_height),
                  (0, 0, 255), -1)  

    # Overlay text
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Rage Level: {rage_level}% {rage_expression}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, timestamp, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show "Poor guy needed a break" in red at 7 seconds of no motion
    if away_from_screen:
        text = "Poor guy needed a break"
        font_scale = min(frame_width, frame_height) / 700
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height // 2)  
        
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 3, cv2.LINE_AA)

    if recording_active:
        out.write(frame)

    cv2.imshow("Gamer Rage Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User exited early.")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as '{}'.".format(output_filename))
