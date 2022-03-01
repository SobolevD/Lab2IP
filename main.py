import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_RANGE_BORDER = 0
MAX_RANGE_BORDER = 256

INPUT_IMAGE_PATH = 'C:\\Users\\soboi\\Desktop\\01_ap.jpg'
OUTPUT_IMAGE_PATH = 'C:\\Users\\soboi\\Desktop\\01_ap2.jpg'


def get_image(filename):
    return plt.imread(filename)


def create_histogram(cum_sum, img):
    plt.plot(cum_sum, color='red')
    plt.hist(img.flatten(), MAX_RANGE_BORDER, [MIN_RANGE_BORDER, MAX_RANGE_BORDER], color='blue')
    plt.xlim([MIN_RANGE_BORDER, MAX_RANGE_BORDER])
    plt.legend(('Integral distribution function', 'Histogram'), loc='upper left')
    plt.show()


def create_single_histogram(img):
    plt.hist(img.flatten(), MAX_RANGE_BORDER, [MIN_RANGE_BORDER, MAX_RANGE_BORDER], color='blue')
    plt.xlim([MIN_RANGE_BORDER, MAX_RANGE_BORDER])
    plt.legend('Histogram', loc='upper left')
    plt.show()


def manual_equalization():
    # Read image
    img = get_image(INPUT_IMAGE_PATH)

    # Show start image
    plt.title("Start image")
    plt.imshow(img)
    plt.show()

    hist, bins = np.histogram(img.flatten(), MAX_RANGE_BORDER, [MIN_RANGE_BORDER, MAX_RANGE_BORDER])

    # Calculate cumulative sum
    cum_sum = hist.cumsum()
    cdf_normalized = cum_sum * float(hist.max()) / cum_sum.max()

    create_histogram(cdf_normalized, img)

    # Integral function
    cum_sum = (cum_sum - 1) * 255 / (cum_sum[-1] - 1)
    cum_sum = cum_sum.astype(np.uint8)

    # Img = indexes for cum_sum = result image
    img_result = cum_sum[img]

    # Show result image
    plt.title("Changed image")
    plt.imshow(img_result)
    plt.show()

    create_histogram(cdf_normalized, img_result)


def cv_equalization():
    # Read image
    img = cv.imread(INPUT_IMAGE_PATH, 0)

    # Show start image
    plt.title("Start image")
    plt.imshow(img)
    plt.show()

    # Create histogram
    create_single_histogram(img)

    hist, bins = np.histogram(img.flatten(), MAX_RANGE_BORDER, [MIN_RANGE_BORDER, MAX_RANGE_BORDER])
    img_result = cv.equalizeHist(img)

    # Show result image
    plt.title("Changed image")
    plt.imshow(img_result)
    plt.show()

    # Create histogram
    create_single_histogram(img_result)


if __name__ == '__main__':
    manual_equalization()
    cv_equalization()







