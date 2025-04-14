import datetime
import cv2
import numpy as np
import time
import sys

from functools import partial
import math

def sort_clockwise(items, get_point=lambda x: x):
    # Extract coordinates from each item using the get_point function
    points = [get_point(item) for item in items]

    # Calculate the centroid of the points
    center = np.mean(points, axis=0)

    # Function to calculate the angle between a point and the centroid
    def calculate_angle(item):
        point = get_point(item)
        return math.atan2(point[1] - center[1], point[0] - center[0])

    # Sort the items based on their angle with respect to the centroid
    return sorted(items, key=calculate_angle)

def create_binary_image(marker_grid, image_size=200):
    # Create a binary image from a grid
    cell_size = image_size // marker_grid.shape[0]
    binary_marker = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Fill in the white cells (value 1)
    for i in range(marker_grid.shape[0]):
        for j in range(marker_grid.shape[1]):
            if marker_grid[i, j] == 1:
                # Calculate pixel coordinates
                y_start = i * cell_size
                y_end = (i + 1) * cell_size
                x_start = j * cell_size
                x_end = (j + 1) * cell_size
                
                # Set the region to white (255)
                binary_marker[y_start:y_end, x_start:x_end] = 255
                
    return binary_marker

def perspective_transform(image, corners):
    """
    Transform the perspective of an image based on four corners.
    
    Args:
        image: Original image (numpy array)
        corners: List of four points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] representing the corners
                 of the document in the original image (top-left, top-right, bottom-right, bottom-left)
    
    Returns:
        Transformed image with corrected perspective
    """
    if corners is None or len(corners) != 4:
       return image
    # Convert corners to numpy array
    corners = np.array(corners, dtype=np.float32)
    
    # Get width and height of the document
    # Calculate the width as the max distance between corner pairs
    width_top = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
    width_bottom = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
    max_width = int(max(width_top, width_bottom))
    
    # Calculate the height as the max distance between corner pairs
    height_left = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + ((corners[3][1] - corners[0][1]) ** 2))
    height_right = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + ((corners[2][1] - corners[1][1]) ** 2))
    max_height = int(max(height_left, height_right))
    
    # Define the destination points for the transformed image
    dst_points = np.array([
        [0, 0],               # top-left
        [max_width - 1, 0],   # top-right
        [max_width - 1, max_height - 1], # bottom-right
        [0, max_height - 1]   # bottom-left
    ], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return warped

def main():
    # Open the webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam: {frame_width}x{frame_height} at {fps} FPS")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    current_fps = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is not received successfully, break the loop
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            current_fps = frame_count / elapsed_time
            # Print FPS without newline, overwriting previous output
            sys.stdout.write(f"\rCurrent FPS: {current_fps:.2f}")
            sys.stdout.flush()
            frame_count = 0
            start_time = time.time()

        # ===========PROCESSING============== #

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_eq = cv2.equalizeHist(frame_gray)
        thresh = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # ===========CONTOURS============== #

        frame_cont = frame.copy()

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            # Approximate the contour to find squares
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 4:
              area = cv2.contourArea(contour)
              if (area < 25 or area > 1000):
                  continue

              if not cv2.isContourConvex(approx):
                  continue

              cv2.drawContours(frame_cont, [approx], 0, (0, 255, 0), 2)
              candidates.append(approx)

        # ===========MARKER DETECTION============== #

        candidates_masks = []
        for cand in candidates:
          marker_corners = np.squeeze(cand).astype(np.float32)

          canonical_size = 200  # Size of the canonical marker (square)
          dst_points = np.array([
              [0, 0],
              [canonical_size, 0],
              [canonical_size, canonical_size],
              [0, canonical_size]
          ], dtype=np.float32)

          # 4. Apply perspective transformation
          perspective_matrix = cv2.getPerspectiveTransform(marker_corners, dst_points)
          canonical_marker = cv2.warpPerspective(frame_eq, perspective_matrix, (canonical_size, canonical_size))

          # 5. Apply Otsu thresholding to separate black and white bits
          _, bin_marker = cv2.threshold(canonical_marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

          # Calculate moments of the contour
          M = cv2.moments(cand)
          
          # Check if the contour has area to avoid division by zero
          if M["m00"] != 0:
            # Calculate center coordinates (x, y)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            candidates_masks.append({"image": bin_marker, "center": (center_x, center_y), "contour": cand})

        matching_masks = []
        for cand in candidates_masks:
            max_iou = 0
            best_match = None

            # Find the marker with maximum IoU for this candidate
            for mrk in binary_markers:
                intersection = np.logical_and(mrk, cand["image"])
                union = np.logical_or(mrk, cand["image"])
                iou = np.sum(intersection) / np.sum(union)

                # Update best match if this has higher IoU
                if iou > max_iou:
                    max_iou = iou
                    best_match = cand

            # Only add if we found a match with IoU above threshold
            if max_iou > 0.5:
                matching_masks.append({"cand": best_match, "iou": max_iou})

        matching_masks.sort(key=lambda e: e["iou"])
        matching_masks = matching_masks[:4]

        # ===========CORNER DETECTION============== #

        frame_corn = frame.copy()
        if matching_masks:
            matching_masks = [c["cand"] for c in matching_masks]
            
            centers = sort_clockwise(matching_masks, lambda x: x["center"])
            corners = []
            for i, c in enumerate(centers):
                cv2.circle(frame_corn, c["center"], 5, (0, 0, 255), -1)

                cnt = sort_clockwise([point[0] for point in c["contour"]])
                cv2.circle(frame_corn, cnt[i], 5, (0, 255, 255), -1)
                corners.append(cnt[i])

        # ===========DOCUMENT DETECTION============== #

        frame_doc = frame.copy()
        if 'corners' in locals() and corners and len(corners) == 4:
            num_points = len(corners)
            for j in range(num_points):
                # Connect each point to the next point (with wrap-around)
                start_point = tuple(corners[j])
                end_point = tuple(corners[(j + 1) % num_points])  # Modulo ensures wrapping around to the first point
                cv2.line(frame_doc, start_point, end_point, (0, 255, 0), 2)  # Green line with thickness 2

            warped = perspective_transform(frame, corners)

        # Display the resulting frame
        cv2.imshow('Camera', frame)
        cv2.imshow("Contours", frame_cont)
        cv2.imshow("Corners", frame_corn)
        cv2.imshow("Document", frame_doc)
        if 'warped' in locals(): cv2.imshow("Warped", warped)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Press 'c' to capture document
        if cv2.waitKey(1) & 0xFF == ord('c') and 'warped' in locals():
            # Get current timestamp
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

            # Create filename with timestamp
            filename = f"captures/doc_{timestamp_str}.png"
            cv2.imwrite(filename, warped, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    # Add a newline after breaking out of the loop
    print()
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
  # Original marker
  marker = [0, 0, 0, 0,
            0, 1, 0, 0,
            0, 1, 1, 0,
            0, 0, 0, 0]

  # Convert list to 4x4 grid
  marker_grid = np.array(marker).reshape(4, 4)

  # Create a list to store all rotations
  binary_markers = []

  # Generate all four rotations (0°, 90°, 180°, 270°)
  for k in range(4):
      # Rotate the grid k times by 90 degrees
      rotated_grid = np.rot90(marker_grid, k)
      
      # Create binary image from the rotated grid
      binary_image = create_binary_image(rotated_grid)
      
      # Add to the list
      binary_markers.append(binary_image)
      
  main()