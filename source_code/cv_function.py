import cv2
import numpy as np
import math


def lower_value(hsv_list):
    sum_li = []
    for i in range(122, 505):
        for j in range(260,290):
            sum_li.append(hsv_list[j][i][2])
    return sum(sum_li) / len(sum_li)


def not_inrange(img,lbound, ubound):
    img_inrange = cv2.inRange(img, lbound, ubound)
    is_img = (img_inrange == img)
    img[is_img] = 0

    return img


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def hough_lines2(img, cimg):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 40)

    angle_list = []
    line_list = []
    line_list01 = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if theta > np.pi/2:
                theta = theta - np.pi
            a = math.cos(theta) # -1 ~ 1
            b = math.sin(theta) # 0 ~ 1
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            #line_list.append(pt2)
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            line_list.append(theta)
            degree = (math.atan2(dy, dx) * 180)/math.pi
            angle_list.append(rho)
            cv2.line(cimg, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    #print(line_list)
    return (angle_list, line_list)


def roi_vertices(image, vertices):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        ignore_mask_color = (255,) * image.shape[2]
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image

