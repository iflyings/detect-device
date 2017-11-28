# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_image(image):
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 800, 600)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def show_histogram(image):
    # opencv方法读取-cv2.calcHist（速度最快）
    # 图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
    hist_cv = cv2.calcHist([image], [0], None, [256], [0, 256])
    # numpy方法读取-np.histogram()
    hist_np, bins = np.histogram(image.ravel(), 256, [0, 256])
    # numpy的另一种方法读取-np.bincount()（速度=10倍法2）
    hist_np2 = np.bincount(image.ravel(), minlength=256)
    plt.subplot(221), plt.imshow(image, 'gray')
    plt.subplot(222), plt.plot(hist_cv)
    plt.subplot(223), plt.plot(hist_np)
    plt.subplot(224), plt.plot(hist_np2)
    cv2.waitKey(0)


def detect_contour(image):
    show_image(image)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    show_image(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

    show_image(thresh)
    '''
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 构造正方形，消除间隙

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=2)  # 4次腐蚀，去掉细节
    closed = cv2.dilate(closed, None, iterations=2)  # 4次膨胀，让轮廓突出

    show_image(closed)
    '''
    # find the contours in the thresholded image
    (_, cnts, __) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours were found, return None
    if len(cnts) == 0:
        return None

    # otherwise, sort the contours by area and compute the rotated
    # bounding box of the largest contour
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(image, [cnt], -1, (255, 255, 255), 10)
    show_image(image)

    # rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # return box # the bounding box of the barcode
    hull = cv2.convexHull(cnt)
    return hull


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    image = cv2.imread("IMG_20171118_094101.jpg")
    # rect = detect_contour(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    rect = detect_contour(np.array(image[:, :, 0]))
    cv2.drawContours(image, [rect], -1, (0, 0, 255), 10)
    show_image(image)


