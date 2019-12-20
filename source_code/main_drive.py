#!/usr/bin/env python

import rospy, time
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import cv_function as cf

from scipy.spatial import distance as dist

image_hi = np.zeros((5, 5, 3), dtype="uint8")

angle = 90
now_angle = 90
bridge = CvBridge()
cv_image = np.empty(shape=[0])

speed = 140
current_state = 0
usonic_data = None
motor_pub = None

face_cascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')

lower = {'red': (-10, 30, 30), 'green': (50, 30, 30)}
upper = {'red': (10, 255, 255), 'green': (70, 255, 255)}
traffic_colors = {'red': (0, 140, 255), 'green': (0, 255, 0)}
isRed = False
isPeople = False
isCross = False


doro_colors = [[255, 255, 255]]
colorNames = ["white"]

lab = np.zeros((len(doro_colors), 1, 3), dtype="uint8")
for i in range(len(doro_colors)):
    lab[i] = doro_colors[i]
lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)


def distance(us_idx):
    global usonic_data
    return usonic_data[us_idx]


def check_collision(us_idx):
    default_speed = 10  # 10m/s
    dist = distance(us_idx)
    return dist / default_speed


def stop_the_car():
    drive(90, 90)


def change_the_gear():
    for stop_cnt in range(2):
        drive(90, 90)
        time.sleep(0.1)
        drive(90, 60)


def go_forward_slowly(angle=90):
    drive(angle, 120)


def back_the_car_slowly(angle=90):
    drive(angle, 65)


def first_turn():
    global usonic_data
    previous = 0
    temp = usonic_data[6] - previous
    print(usonic_data[6])
    while temp > -10:
        go_forward_slowly()
        temp = usonic_data[6] - previous
        previous = usonic_data[6]
    duration = 2.5
    start = time.time()
    end = start + duration
    max_angle, min_angle = [130, 50]
    while (time.time() < end):
        drive(35, 115)
    stop_the_car()


def soft_steering(x):
    y = x * x
    return y


def soft_drive_in():
    duration = 5
    start = time.time()
    end = start + duration
    max_angle, min_angle = [130, 50]
    while (time.time() < end):
        x = (time.time() - start) / duration
        y = soft_steering(x)
        angle = 90 - (40 * y)
        # back_the_car_slowly(angle)
        # drive(angle, 70)
        drive(35, 70)
        if distance(4) < 10:
            break


def calibrate():
    print(distance(2), ' ', distance(5))
    while (abs(distance(2) - distance(5)) > 3):
        # print(distance(1))
        if distance(1) < 15:
            for stop_cnt in range(2):
                drive(90, 90)
                time.sleep(0.1)
                drive(90, 60)
                time.sleep(0.1)
            print("calibrate-go-back")
            # a_while = time.time() + 1.2
            # while time.time() < a_while:
            while distance(4) < 10:
                drive(90, 70)
        else:
            if distance(2) > distance(5):

                drive(120, 110)
            else:
                drive(60, 110)


def move_to_center():
    while (abs(distance(1) - distance(4)) < 3):
        if distance(1) > distance(4):
            drive(90, 115)
        else:
            change_the_gear()
            drive(90, 65)

def parallel_parking():
    first_turn()
    print('finished turn left')
    for stop_cnt in range(2):
        drive(90, 90)
        time.sleep(0.1)
        drive(90, 60)
        time.sleep(0.1)
    while distance(4) > 60:
        back_the_car_slowly()
    print('stop')
    stop_the_car()
    print('soft drive in')
    soft_drive_in()
    print('calibrate')
    #calibrate()
    print('move to center')
    move_to_center()

def label(image, contour):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(image, mask=mask)[:3]

    minDist = (np.inf, None)

    for (i, row) in enumerate(lab):
        d = dist.euclidean(row[0], mean)

        if d < minDist[0]:
            minDist = (d, i)
    return colorNames[minDist[1]]

def hdbd(cam_img):
    #rospy.init_node("camtest_node")
    #rospy.Subscriber("/usb_cam/image_raw", Image, callback)

    #time.sleep(1)

    #cam_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)


    frame = cam_img

    pts1 = np.float32([[210, 248], [400, 248], [20, 314], [600, 314]])
    pts2 = np.float32([[[0, 0], [320, 0], [0, 240], [320, 240]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    birdeye = cv2.warpPerspective(frame, M, (320, 240))

    img_gray = cv2.cvtColor(birdeye, cv2.COLOR_BGR2GRAY)


    ret, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)



    test = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)


    _, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    reccnt = 0
    for cnt in contours:
        #size = len(cnt)
        # print(size)

        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        size = len(approx)
        # print(size)

        cv2.line(birdeye, tuple(approx[0][0]), tuple(approx[size - 1][0]), (0, 255, 255), 3)
        for k in range(size - 1):
            cv2.line(birdeye, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 100 + k * 10), 3)

        if cv2.isContourConvex(approx):
            if size == 4 and label(test, cnt) == "white":
                reccnt += 1
            else:
                pass
    global isCross
    if reccnt >= 5:
        isCross = True

def traffic_cal(frame):
    global isRed
    img_color = frame

    height, width = img_color.shape[:2]
    img_color = cv2.resize(img_color, (width, height), interpolation=cv2.INTER_AREA)

    img_blurred = cv2.GaussianBlur(img_color, (7, 7), 0)
    img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    for key, value in upper.items():
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(img_hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 15:
                cv2.circle(img_color, (int(x), int(y)), int(radius), traffic_colors[key], 2)
                cv2.putText(img_color, key + " ball", (int(x - radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            traffic_colors[key], 2)
                print(key + " ball")

                if key == 'red':
                    isRed = True
                elif key == 'green':
                    isRed = False


def cross_cal(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('result', img_gray)
    # cv.waitKey(0)

    ret, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('result', img_binary)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    reccnt = 0
    for cnt in contours:
        size = len(cnt)
        print(size)

        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        size = len(approx)
        print(size)

        cv2.line(frame, tuple(approx[0][0]), tuple(approx[size - 1][0]), (0, 255, 255), 3)
        for k in range(size - 1):
            cv2.line(frame, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 100 + k * 10), 3)

    if reccnt >= 5:
        print("traffic {}".format(reccnt))
        return True
    else:
        return False


def face_cal(frame):
    global usonic_data
    global speed
    global face_cascade
    global isPeople

    faces = face_cascade.detectMultiScale(frame, 1.03, 5)
    if len(faces) > 0 and usonic_data[1] < 100:
        print('People!')
        isPeople = True

def init_node():
    global motor_pub
    rospy.init_node('sample')
    rospy.Subscriber('ultrasonic', Int32MultiArray, callback2)
    motor_pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw", Image, callback)


def callback2(data):
    global usonic_data
    usonic_data = data.data


def callback(img_data):
    global bridge
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")


def drive(angle, speed):
    global motor_pub
    drive_info = [angle, speed]
    pub_data = Int32MultiArray(data=drive_info)
    motor_pub.publish(pub_data)

car_mod = 0
if __name__ == '__main__':
    init_node()
    global speed
    global angle
    global usonic_data
    #out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30,(640, 480))
    time.sleep(3)
    global image_hi
    image_hi = cv_image[100:101,100:101]
    rate = rospy.Rate(20)

    init_time = time.time()
    while not rospy.is_shutdown():
        #cv2.imshow("camera", cv_image)
        frame = cv_image
        key_num = cv2.waitKey(1)
        if key_num == 27:
            break
        elif key_num == 32:
            while True:
                if cv2.waitKey(1) & 0xFF == 32:
                    break

        cv2.imshow('cg', image_hi)

        ####################################################
        pts1 = np.float32([[59, 308], [562, 308], [6, 330], [609, 330]])
        pts2 = np.float32([[10, 0], [310, 0], [10, 240], [310, 240]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        birdeye = cv2.warpPerspective(frame, M, (320, 240))

        gray = cv2.cvtColor(birdeye, cv2.COLOR_BGR2GRAY)
        gausian = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Canny(gray, 100, 255)
        edgeC = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        try:
            degree_list, theta_list = cf.hough_lines2(edge, edgeC)
            target_theta = {}
            for i in theta_list:
                target_theta[i] = target_theta.get(i, 0) + 1
            sort_theta_list = sorted(target_theta.items(), key=lambda x: x[1], reverse=True)
            # print(sort_theta_list[0][0])
            target_angle = 95 + sort_theta_list[0][0] * 430
            #angle = target_angle
            #print(target_angle)

            if target_angle < angle:
                angle -= 2.4
            if target_angle > angle:
                angle += 1.4

        #############################################
            speed = 135
            if cross_cal(frame):
                speed = 115
            else:
                speed = 120

            # people
            face_cal(frame)
            if isPeople:
                speed = 90
            else:
                speed = 120


            # traffic
            traffic_cal(frame)
            if isRed:
                speed = 90
            else:
                speed = 135

        except ZeroDivisionError:
            print('')
        except IndexError:
            print('')
        drive(angle, speed)

        if time.time - init_time > 25:
            parallel_parking()
            break

# roslaunch usb_cam usb_cam-test.launch
# rosrun xycar_b2 xycar_b2_motor.py