import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_contour(img):
    # img = cv2.imread('pic.png')
    # img = cv2.imread('./output/crops/photo_2025-03-21_19-10-25_crossline-2.jpg')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for red color 
    red_lower = np.array([0, 100, 70], np.uint8)
    red_upper = np.array([5, 255, 255], np.uint8)

    # define mask
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    #green color
    green_lower = np.array([40, 80, 20], np.uint8)
    green_upper = np.array([85, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    #blue color
    blue_lower = np.array([94, 100, 50], np.uint8)
    blue_upper = np.array([120, 255, 180], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    #gray color
    gray_lower = np.array([90, 5, 50], np.uint8)
    gray_upper = np.array([110, 20, 200], np.uint8)
    gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between img and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8") 

    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(img, img, 
                            mask = red_mask) 

    # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(img, img, 
                                mask = green_mask) 

    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(img, img, 
                            mask = blue_mask) 

    # For gray color
    gray_mask = cv2.dilate(gray_mask, kernel)
    res_gray = cv2.bitwise_and(img, img,
                            mask = gray_mask)

    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 

    list_x = []
    list_y = []
    list_z = [25, 50, 75]
    gray_y = 0

    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        print("Area: ", area)
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            img = cv2.rectangle(img, (x, y), 
                                    (x + w, y + h), 
                                    (0, 0, 255), 2) 
            
            cv2.putText(img, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))     
            print("Red coordinates: ", x, y, x+w, y+h)
            print("Red center: ", x+w/2, y+h/2)
            list_x.append(x+w/2)
            list_y.append(y+h/2)

    # Creating contour to track blue color 
    contours, hierarchy = cv2.findContours(blue_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if (area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            img = cv2.rectangle(img, (x, y), 
                                    (x + w, y + h), 
                                    (255, 0, 0), 2) 
            
            cv2.putText(img, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0)) 
            print("Blue coordinates: ", x, y, x+w, y+h)
            print("Blue center: ", x+w/2, y+h/2)
            list_x.append(x+w/2)
            list_y.append(y+h/2)

    # Creating contour to track green color 
    contours, hierarchy = cv2.findContours(green_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 

    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 150): 
            x, y, w, h = cv2.boundingRect(contour) 
            img = cv2.rectangle(img, (x, y), 
                                    (x + w, y + h), 
                                    (0, 255, 0), 2) 
            
            cv2.putText(img, "Green Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0)) 
            print("Green coordinates: ", x, y, x+w, y+h)
            print("Green center: ", x+w/2, y+h/2)
            list_x.append(x+w/2)
            list_y.append(y+h/2)

    # Creating contour to track gray color
    contours, hierarchy = cv2.findContours(gray_mask,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y),
                                    (x + w, y + h),
                                    (128, 128, 128), 2)

            cv2.putText(img, "Gray Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (128, 128, 128))
            print("Gray coordinates: ", x, y, x+w, y+h)
            print("Gray center: ", x+w/2, y+h/2)
            gray_y = y+h/2
    
    if True:
        poly = np.poly1d(np.polyfit(list_y, list_z, 2))
        print(poly)
        print("Estimated distance: ", poly(gray_y))
        plt.plot(list_y, list_z, 'yo', list_y, poly(list_y), '--k')
        plt.show()

    cv2.namedWindow("Color Tracking", cv2.WINDOW_NORMAL)
    cv2.imshow("Color Tracking", img)
    # cv2.imwrite("./output/crops_all/{img_name}_{label}.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()