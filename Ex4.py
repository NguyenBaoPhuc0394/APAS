import cv2
import numpy as np
import math

def draw_line(line, img, color=(0, 0, 255)):
    rho, theta = line[0]
    if theta != 0:
        m = -1 / math.tan(theta)
        c = rho / math.sin(theta)
        cv2.line(img, (0, int(c)), (img.shape[1], int(m * img.shape[1] + c)), color)
    else:
        cv2.line(img, (int(rho), 0), (int(rho), img.shape[0]), color)

sudoku = cv2.imread("./images/sudoku.png", 0)
sudoku_blurred = cv2.GaussianBlur(sudoku, (11, 11), 0)
outer_box = cv2.adaptiveThreshold(sudoku_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 5, 2)
outer_box = cv2.bitwise_not(outer_box)
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
outer_box = cv2.dilate(outer_box, kernel)
max_area = -1
max_pt = None
for y in range(outer_box.shape[0]):
    for x in range(outer_box.shape[1]):
        if outer_box[y, x] >= 128:
            _, _, _, area = cv2.floodFill(outer_box, None, (x, y), 64)
            if area[3] > max_area:
                max_pt = (x, y)
                max_area = area[3]
                
if max_pt:
    cv2.floodFill(outer_box, None, max_pt, 255)

for y in range(outer_box.shape[0]):
    for x in range(outer_box.shape[1]):
        if outer_box[y, x] == 64 and (x, y) != max_pt:
            cv2.floodFill(outer_box, None, (x, y), 0)
            
outer_box = cv2.erode(outer_box, kernel)

cv2.imshow("Thresholded", outer_box)

lines = cv2.HoughLines(outer_box, 1, np.pi / 180, 200)
if lines is not None:
    for line in lines:
        draw_line(line, outer_box)

cv2.imshow("Detected Lines", outer_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
