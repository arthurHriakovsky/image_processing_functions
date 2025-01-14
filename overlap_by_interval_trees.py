import numpy as np
import cv2
import random
from intervaltree import Interval, IntervalTree

# define the Box class
class Box:
    def __init__(self, first_point=(0, 0), second_point=(0, 0)):
        self.top_left = (min(first_point[0], second_point[0]), min(first_point[1], second_point[1]))
        self.bottom_right = (max(first_point[0], second_point[0]), max(first_point[1], second_point[1]))

# Function to check if two boxes overlap
def find_overlaps(box1, box2):
    x_left = max(box1.top_left[0], box2.top_left[0])
    y_top = max(box1.top_left[1], box2.top_left[1])
    x_right = min(box1.bottom_right[0], box2.bottom_right[0])
    y_bottom = min(box1.bottom_right[1], box2.bottom_right[1])
    return not (x_right < x_left or y_bottom < y_top)

# function to find overlapping rectangles using interval trees
def find_overlapping_rectangles(rectangles):
    overlapped = []
    x_tree = IntervalTree()
    y_tree = IntervalTree()
    for i, box in enumerate(rectangles):
        # Create x overlapped and y overlapped trees
        x_tree.add(Interval(box.top_left[0], box.bottom_right[0], str(i)))
        y_tree.add(Interval(box.top_left[1], box.bottom_right[1], str(i)))
    for i, box in enumerate(rectangles):
        # we want to find all intervals that overlapped on x and y axis both
        x_overlapped_intervals = x_tree[box.top_left[0]:box.bottom_right[0]]
        y_overlapped_intervals = y_tree[box.top_left[1]:box.bottom_right[1]]
        # getting the only indices of overlapped rectangles except of the rectangle itself
        x_data = set([interval.data for interval in x_overlapped_intervals])
        y_data = set([interval.data for interval in y_overlapped_intervals])
        common_values = x_data.intersection(y_data) - {str(i)}
        if common_values:
            overlapped.append(box)

    return overlapped

def testing(number_of_rectangles):
    width, height = 800, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    default_color = (255, 255, 255)  # white
    overlapped_color = (0, 255, 0)  # green
    thickness = 2
    rectangles = []

    for _ in range(number_of_rectangles):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        box = Box((x1, y1), (x2, y2))
        rectangles.append(box)

    overlapping_rectangles = find_overlapping_rectangles(rectangles)

    # draw rectangles
    for box in rectangles:
        if box in overlapping_rectangles:
            cv2.rectangle(image, box.top_left, box.bottom_right, overlapped_color, thickness)
        else:
            cv2.rectangle(image, box.top_left, box.bottom_right, default_color, thickness)

    # show the image
    cv2.imshow("Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    testing(8)  # test with N random rectangles
