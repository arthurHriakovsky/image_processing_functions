import cv2
import numpy as np
import matplotlib.pyplot as plt


def diff_by_treshold(img_1, img_2, treshold=25):
    diff = cv2.absdiff(img_1, img_2)
    _, threshold_image = cv2.threshold(diff, treshold, 255, cv2.THRESH_BINARY)
    return threshold_image


def detect_changes(img_1, img_2, threshold_value=25, min_area=50, rotated_bounding_boxes=True):
    # define contoured image for visualisation and result image for the result with bounding boxes
    contured_image = img_1.copy()
    rslt_image = img_1.copy()

    threshold_image = diff_by_treshold(img_1, img_2, threshold_value)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Filter out small changes
            if rotated_bounding_boxes:
                # finds a rotated rectangle of the minimum area
                rect = cv2.minAreaRect(contour)
                # finds the four vertices of a rotated rectangle
                box = cv2.boxPoints(rect)
                # convert points to np array
                box = box.astype(int)
                cv2.drawContours(rslt_image, [box], 0, (0, 0, 0), 2)
            else:
                # boundingRect function calculates the minimal up-right bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rslt_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.drawContours(contured_image, contours, -1, (0, 0, 0), 2)

    return rslt_image, contured_image


if __name__ == '__main__':
    # Load images as grayscale
    image1 = cv2.imread("1.jpg")
    image2 = cv2.imread("2.jpg")
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    bin_image = diff_by_treshold(image1_gray, image2_gray)
    result_image, contoured_image = detect_changes(image1_gray, image2_gray)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.title('Thresholded Difference')
    plt.imshow(bin_image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Changes with Bounding Boxes')
    plt.imshow(result_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Contoured image')
    plt.imshow(contoured_image, cmap='gray')

    plt.show()

