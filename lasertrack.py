import cv2
import numpy as np
import board
import digitalio
import adafruit_hid.mouse

# monitor bounding box:
(x1, y1) = (40, 50)
(x2, y2) = (60, 70)

# camera screen resolution:
camera_screen_width = x2 - x1
camera_screen_height = y2 - y1

# screen resolution
screen_width = 1920
screen_height = 1080


# Initialize the mouse
mouse = adafruit_hid.mouse.Mouse(board.USB)

def coordinate_transform(x, y, x1, y1, x2, y2):
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return (x - x1, y - y1)
    else:
        return None

def get_mouse_position(x, y):
    # mouse position fed to computer
    x = x * 32767 / camera_screen_width
    y = y * 32767 / camera_screen_height
    return (x, y)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the lower and upper bounds of the laser color in the HSV color space
lower_laser = np.array([150, 0, 0])
upper_laser = np.array([190, 50, 250])

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only the laser color
    mask = cv2.inRange(hsv, lower_laser, upper_laser)
    
    # Find the contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours were found
    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find the moments of the largest contour
        M = cv2.moments(largest_contour)
        
        # Find the centroid of the largest contour
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # coordinate_transformed = coordinate_transform(cX, cY, x1, y1, x2, y2)

            # Get the HSV value of the centroid
            (h, s, v) = hsv[cY, cX]
            
            # Print the HSV value
            print("HSV:", (h, s, v))
            
            # Draw a circle at the centroid on the original frame
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    
    # Show the original frame with the laser position
    cv2.imshow("Laser Detection", frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
