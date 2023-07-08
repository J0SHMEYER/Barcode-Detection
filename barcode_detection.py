import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

def standard_deviation_filter(image):
    height = len(image)
    width = len(image[0])

    result = [[0 for x in range(width)] for y in range(height)]

    for i in range(height):
        for j in range(width):
            if i >= 2 and i < height - 2 and j >= 2 and j < width - 2:

                neighborhood = []
                for m in range(i - 2, i + 3):
                    for n in range(j - 2, j + 3):
                        neighborhood.append(image[m][n])

                mean = sum(neighborhood) / 25
                variance = sum([(x - mean) ** 2 for x in neighborhood]) / 25
                stdDev = variance ** 0.5

                result[i][j] = stdDev

    return result

def gaussianFilter(image, imageWid, imageHei):
    filterSize = 3
    gaussianFilter = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    output = []
    rowEp = []
    for i in range(imageWid):
        rowEp.append(image[0][i])
    output.append(rowEp)

    for i in range(filterSize // 2, len(image) - filterSize // 2):
        row = []
        row.append(image[i][0])
        for j in range(filterSize // 2, len(image[0]) - filterSize // 2):
            pixel_sum = 0
            for k in range(filterSize):
                for l in range(filterSize):
                    pixel_sum += image[i + k - filterSize//2][j + l - filterSize//2] * gaussianFilter[k][l]
            row.append(pixel_sum // 16)
        
        row.append(image[i][imageWid - 1])
        output.append(row)
    
    rowEp = []
    for i in range(imageWid):
        rowEp.append(image[imageHei - 1][i])
    output.append(rowEp)

    return output

def erosion(image):
    height = len(image)
    width = len(image[0])
    result = [[0] * width for x in range(height)]
    
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            minVal = float('inf')
            for x in range(-2, 3):
                for y in range(-2, 3):
                    minVal = min(minVal, image[i + x][j + y])
            result[i][j] = minVal
    
    return result

def dilation(image):
    height = len(image)
    width = len(image[0])
    result = [[0] * width for x in range(height)]
    
    for i in range(height):
        for j in range(width):
            maxVal = 0
            for x in range(i-2, i+3):
                for y in range(j-2, j+3):
                    if 0 <= x < height and 0 <= y < width:
                        maxVal = max(maxVal, image[x][y])
            result[i][j] = maxVal
    
    return result

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
def computeConnectedComponent(pixelArr, imageWid, imageHei):

    image = createInitializedGreyscalePixelArray(imageWid, imageHei)
    visited = createInitializedGreyscalePixelArray(imageWid, imageHei)

    label = 1
    info = {}

    for i in range(imageHei):
        for j in range(imageWid):

            if pixelArr[i][j] != 0 and visited[i][j] != 1:

                q = Queue()
                q.enqueue((i, j))
                visited[i][j] = 1
                count = 0

                while not q.isEmpty():

                    (y, x) = q.dequeue()
                    image[y][x] = label
                    count += 1

                    neighbours = [
                        (y + 1, x),
                        (y - 1, x),
                        (y, x + 1),
                        (y, x - 1)
                    ]

                    for px in neighbours:

                        if -1 < px[0] < imageHei and -1 < px[1] < imageWid:

                            if pixelArr[px[0]][px[1]] != 0 and visited[px[0]][px[1]] != 1:
                                q.enqueue((px[0], px[1]))
                                visited[px[0]][px[1]] = 1

                info[label] = count
                label += 1

    return image, info

def computeBoundingBox(pixelArr, imageWid, imageHei, dict):

    aspectRatio = 0
    index = -1
    
    while not (0 < aspectRatio < 1.8) and index < len(dict) - 1:

        index += 1
        label = dict[index][0]

        min_x = imageWid - 1
        max_x = 0
        min_y = imageHei - 1
        max_y = 0

        for i in range(imageHei):
            for j in range(imageWid):

                if pixelArr[i][j] == label:

                    if j < min_x:
                        min_x = j

                    if j > max_x:
                        max_x = j

                    if i < min_y:
                        min_y = i

                    if i > max_y:
                        max_y = i

        # Avoid division by zero
        if max_y != min_y:
            aspectRatio = (max_x - min_x) / (max_y - min_y)

    return min_x, max_x, min_y, max_y

def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode2"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    (imageWid, imageHei, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    grayImg = [[0 for j in range(imageWid)] for i in range(imageHei)]

    for i in range(imageHei):
        for j in range(imageWid):
            grayImg[i][j] = 0.299 * float(px_array_r[i][j]) + 0.587 * float(px_array_g[i][j]) + 0.114 * float(px_array_b[i][j])

    minVal = 0
    maxVal = 0

    for row in grayImg:
        for value in row:
            if value < minVal:
                minVal = value
            if value > maxVal:
                maxVal = value

    stretchedImg = [[0 for j in range(imageWid)] for i in range(imageHei)]

    for i in range(imageHei):
        for j in range(imageWid):
            value = int(round(((grayImg[i][j] - minVal) / (maxVal - minVal)) * 255))
            value = max(0, min(255, value))
            stretchedImg[i][j] = value

    minVal = 0
    maxVal = 0

    for row in stretchedImg:
        for value in row:
            if value < minVal:
                minVal = value
            if value > maxVal:
                maxVal = value


    result = standard_deviation_filter(stretchedImg)

    # having this higher(multiple runs) caused words to start being detected sometimes
    gau_result = gaussianFilter(result, imageWid, imageHei)

    minVal = 0
    maxVal = 0

    for row in gau_result:
        for value in row:
            if value < minVal:
                minVal = value
            if value > maxVal:
                maxVal = value

    stretched_image_data_for_threshold = []

    for row in gau_result:
        thresholded_row = []
        for pixel in row:
            thresholded_row.append((pixel - minVal) * 255 / (maxVal - minVal))
        stretched_image_data_for_threshold.append(thresholded_row)

    minVal = 0
    maxVal = 0

    for row in stretched_image_data_for_threshold:
        for value in row:
            if value < minVal:
                minVal = value
            if value > maxVal:
                maxVal = value

    thresholded_image = []

    for row in stretched_image_data_for_threshold:
        thresholded_row = []
        for pixel in row:
            if pixel >= 75:
                ## mean + 2.5 * std:
                thresholded_row.append(255)
            else:
                thresholded_row.append(0)
        thresholded_image.append(thresholded_row)

    step5 = erosion(thresholded_image)
    step5 = erosion(step5)
    step5 = erosion(step5)
    step5 = dilation(step5)
    step5 = dilation(step5)
    step5 = dilation(step5)
    step5 = dilation(step5)
    step5 = erosion(step5)


    # Connected Component Analysis
    px_array, cc_dict = computeConnectedComponent(step5, imageWid, imageHei)

    # Sort to find largest connect component
    cc_dict = sorted(cc_dict.items(), key=lambda item: item[1], reverse=True)

    # Compute bounding box
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y = computeBoundingBox(px_array, imageWid, imageHei, cc_dict)
    
    px_array = px_array_r

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()