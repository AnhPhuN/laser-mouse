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

# Define the threshold for how long the laser can be gone before clicking the mouse
threshold = 10 # 10 frames
# Define a timer to keep track of how long the laser has been gone
disappearance_timer = 11


# Initialize the mouse
mouse = adafruit_hid.mouse.Mouse(board.USB)

# takes in position of laser on camera screen and returns position on computer screen
def coordinate_transform(x, y, x1, y1, x2, y2):
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return (x - x1, y - y1)
    else:
        return None

# def get_mouse_position(x, y):
#     # mouse position fed to computer
#     x = x * 32767 / camera_screen_width
#     y = y * 32767 / camera_screen_height
#     return (x, y)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the lower and upper bounds of the laser color in the HSV color space
lower_laser = np.array([150, 0, 0])
upper_laser = np.array([190, 50, 250])

# move mouse to top left
mouse.move(-screen_width, -screen_height)
old_position = (0, 0)

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

        # set time of laser gone to 0
        disappearance_timer = 0
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find the moments of the largest contour
        M = cv2.moments(largest_contour)
        
        # Find the centroid of the largest contour
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            coordinate_transformed = coordinate_transform(cX, cY, x1, y1, x2, y2)
            if coordinate_transformed is not None:
                mouse.move(coordinate_transformed[0] - old_position[0], coordinate_transformed[1] - old_position[1])
                old_position = coordinate_transformed
            


            # Get the HSV value of the centroid (for debugging)
            (h, s, v) = hsv[cY, cX]
            print("HSV:", (h, s, v))
            
            # Draw a circle at the centroid on the original frame
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    else:
        # Increment the disappearance timer
        disappearance_timer += 1
        # If the laser has been gone for the threshold number of frames, print "hello"
        if disappearance_timer == threshold:
            mouse.click(Mouse.LEFT_BUTTON)
            disappearance_timer = 0

    # Show the original frame with the laser position
    cv2.imshow("Laser Detection", frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
