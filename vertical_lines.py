import cv2
import math
import numpy as np
import time
import matplotlib.pyplot as plt

from functions import *


N_PIXELS_PER_CHUNK = 20
HORIZONTAL_MARGIN = -2000
VERTICAL_MARGIN = -2000

display_progress = False
horizontal_lines = []
vertical_lines = []

#CHOOSE A PIC
# img = cv2.imread('photo_samples/jun_2018.jpg')
# img = cv2.imread('photo_samples/4.jpg')
img = cv2.imread('data/56.jpg')
print(img.shape)
rows = img.shape[0]
cols = img.shape[1]
print(rows*cols)

# start = time.time()

computer_shot = discriminate(img)

num_of_hor_chunks = math.ceil(cols / N_PIXELS_PER_CHUNK)
num_of_vert_chunks = math.ceil(rows / N_PIXELS_PER_CHUNK)
vert_lines_x = np.linspace(0, cols - 1, num_of_hor_chunks, dtype=np.uint16)
hor_lines_y = np.linspace(0, rows - 1, num_of_vert_chunks, dtype=np.uint16)
if display_progress:
    print(vert_lines_x)

# APPLY FILTERS
if not computer_shot:
    initial_filter = cv2.bilateralFilter(img, 9, 55, 55)
else:
    initial_filter = cv2.GaussianBlur(img, (11, 11), 0)

grayscale = cv2.cvtColor(initial_filter, cv2.COLOR_BGR2GRAY)
adaptive_thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
adaptive_thresh_dil = cv2.dilate(adaptive_thresh, np.ones((2, 2), np.uint16), iterations=1)

# REMOVE SMALL BLOBS
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(adaptive_thresh_dil, connectivity=8,
                                                                           ltype=cv2.CV_32S)
sizes = stats[1:, -1]
nb_components = nb_components - 1
max_size = 201
largest_chunk_num = 0

largest_chunk_img = np.zeros(output.shape, dtype=np.uint8)

for i in range(0, nb_components):
    if sizes[i] >= max_size:
        max_size = sizes[i]
        largest_chunk_num = i

largest_chunk_img[output == largest_chunk_num + 1] = 255
# cv2.imshow('a', largest_chunk_img)
# cv2.waitKey()


#LEAVE HORIZONTAL LINES
img_horizontal = cv2.erode(adaptive_thresh_dil, np.ones((1, 5), np.uint16), iterations=5)
img_horizontal = cv2.dilate(img_horizontal, np.ones((1, 5), np.uint16), iterations=5)
if display_progress:
    cv2.imshow('horizontal', img_horizontal)
    cv2.waitKey()

# left_bottom = np.where(img_horizontal == 255)
# print(left_bottom)

right_bottom_point = get_lower_right(img_horizontal)
if display_progress:
    if right_bottom_point is None:
        print("NONE")
    else:
        print(right_bottom_point)

bottom_line_initial = get_bottom_line(right_bottom_point, img_horizontal, vert_lines_x, N_PIXELS_PER_CHUNK)
# opposite = math.fabs(bottom_line_initial[0][0] - bottom_line_initial[-1][0])
opposite = math.fabs(bottom_line_initial[-1][0] - bottom_line_initial[0][0])
print(opposite)
anti_clockwise = bottom_line_initial[-1][0] < bottom_line_initial[0][0]
if anti_clockwise:
    angle = (math.degrees(math.atan(opposite/cols)))
else:
    angle = - (math.degrees(math.atan(opposite/cols)))

print('angle', angle)
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
cv2.imshow('img', img)
cv2.waitKey()
start = time.time()

#APPLY FILTERS TO CORRECTED IMAGE
if not computer_shot:
    bilateral_filter = cv2.bilateralFilter(img, 9, 55, 55)
else:
    bilateral_filter = cv2.GaussianBlur(img, (11, 11), 0)

bilateral_filter_gray = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2GRAY)
adaptive_thresh = cv2.adaptiveThreshold(bilateral_filter_gray, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

adaptive_thresh_dil = cv2.dilate(adaptive_thresh, np.ones((2, 2), np.uint16), iterations=1)

# REMOVE SMALL BLOBS
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(adaptive_thresh_dil, connectivity=8,
                                                                           ltype=cv2.CV_32S)
sizes = stats[1:, -1]
nb_components = nb_components - 1

max_size = 201
largest_chunk_num = 0

small_chunks_img = np.zeros(output.shape)
largest_chunk_img = np.zeros(output.shape, dtype=np.uint8)

for i in range(0, nb_components):
    if sizes[i] < max_size:
        small_chunks_img[output == i + 1] = 255
    if sizes[i] >= max_size:
        max_size = sizes[i]
        largest_chunk_num = i

largest_chunk_img[output == largest_chunk_num + 1] = 255
# cv2.imshow('a', largest_chunk_img)
# cv2.waitKey()


# LEAVE HORIZONTAL LINES######################################################################################
img_horizontal = cv2.erode(largest_chunk_img, np.ones((1, 5), np.uint16), iterations=5)
img_horizontal = cv2.dilate(img_horizontal, np.ones((1, 5), np.uint16), iterations=5)
if display_progress:
    cv2.imshow('horizontal', img_horizontal)
    cv2.waitKey()

# left_bottom = np.where(img_horizontal == 255)
# print(left_bottom)

right_bottom_point = get_lower_right(img_horizontal)
print('right bottom point', right_bottom_point)
if display_progress:
    if right_bottom_point is None:
        print("NONE")
    else:
        print(right_bottom_point)

bottom_line = get_bottom_line(right_bottom_point, img_horizontal, vert_lines_x, N_PIXELS_PER_CHUNK)

# print(bottom_line)
horizontal_lines.append(bottom_line)

lowest_row = img_horizontal.shape[0]
for i in bottom_line:
    if i[0] < lowest_row:
        lowest_row = i[0]

single_line_img = np.zeros(img_horizontal.shape)
difference = np.count_nonzero(single_line_img == img_horizontal)
correlation_list = []

line = copy.deepcopy(bottom_line)

after_line = True
first_bot = True
local_max = []
local_ind = []
i = 0
line_img = np.zeros(img_horizontal.shape, dtype=np.uint8)

pic = np.zeros(img_horizontal.shape)

while lowest_row - i > 0:

    line = raise_line(line, 1)
    new_im = copy.deepcopy(img_horizontal)

    for n in range(1, len(line)):
        cv2.line(line_img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [255, 255, 255], 2)
        cv2.line(new_im, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [255, 255, 255], 2)

    if display_progress:
        cv2.imshow('horizontal', new_im)
        cv2.waitKey()

    correlation = np.count_nonzero(line_img == img_horizontal) - difference

    if correlation > HORIZONTAL_MARGIN:
        sel = np.logical_and(line_img == img_horizontal, line_img != 0)
        pic_chunk = np.where(sel == True, np.uint8(255), np.uint8(0))
        pic = pic + pic_chunk
        pic[pic > 255] = 255
        # cv2.imshow('qwert', pic)
        # cv2.waitKey()

    correlation_list.append(correlation)

    for n in range(1, len(line)):
        cv2.line(line_img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [0, 0, 0], 2)

    # print("correlation", correlation, 'coords', lowest_row - i)
    if first_bot and correlation > HORIZONTAL_MARGIN:
        i += 1
        line = raise_line(line)
        continue
    elif first_bot and correlation < HORIZONTAL_MARGIN:
        first_bot = False
        i += 1
        continue

    if correlation > HORIZONTAL_MARGIN: #and not after_line:    # still in ROI

        local_max.append(correlation)
        local_ind.append(i)
        after_line = False

    if correlation < HORIZONTAL_MARGIN and not after_line:  # ROI is over
        maximum = max(local_max)
        max_index = local_ind[local_max.index(maximum)]

        # print('initial', lowest_row - max_index)
        # print('passed', lowest_row - max_index + (line[0][0]-line[-1][0]))
        # print('line', line[0][0], line[-1][0])

        # print('coords', lowest_row - max_index)

        # cv2.imshow('pic', pic)
        # cv2.waitKey()

        line = get_next_hor_line(lowest_row - max_index, pic, N_PIXELS_PER_CHUNK, vert_lines_x)
        # print(line)
        l = copy.deepcopy(line)
        horizontal_lines.append(l)
        line = raise_line(line, 15)

        local_max = []
        local_ind = []
        pic = np.zeros(img_horizontal.shape)
        after_line = True
        first_bot = True

        i = lowest_row - line[1][0] - 1 # for future raise line at the start of the cycle
        # print(i)
        continue

    i += 1

# line_img_ = copy.deepcopy(img)
for line in horizontal_lines:
    for n in range(1, len(line)):
        cv2.line(img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [0, 0, 255], 1)

#####################################################################################################

# LEAVE VERTICAL LINES
img_vertical = cv2.erode(largest_chunk_img, np.ones((5, 1), np.uint16), iterations=5)
img_vertical = cv2.dilate(img_vertical, np.ones((8, 1), np.uint16), iterations=7)

left_bottom_point = get_lower_left(img_vertical)
if display_progress:
    if left_bottom_point is None:
        print("NONE")
    else:
        print(left_bottom_point)
        cv2.imshow('vertical', img_vertical)
        cv2.waitKey()


leftmost_line = get_leftmost_line(left_bottom_point, img_vertical, hor_lines_y)
# print(leftmost_line)

# for i in range(1, len(leftmost_line)):
#     cv2.line(img, (leftmost_line[i-1][1], leftmost_line[i-1][0]), (leftmost_line[i][1], leftmost_line[i][0]), [0, 0, 255], 1)

vertical_lines.append(leftmost_line)

leftmost_row = img_vertical.shape[1]
for i in leftmost_line:
    if i[1] < leftmost_row:
        leftmost_row = i[1]

single_line_img = np.zeros(img_vertical.shape)
difference = np.count_nonzero(single_line_img == img_vertical)

line = copy.deepcopy(leftmost_line)

after_line = True
first_bot = True
local_max = []
local_ind = []
correlation_list = []

pic = np.zeros(img_horizontal.shape)

i = 1
line_img = np.zeros(img_vertical.shape)
while leftmost_row + i < img_vertical.shape[1]:

    line = shift_right_line(line, 1)

    new_im = copy.deepcopy(img_vertical)
    for n in range(1, len(line)):
        cv2.line(line_img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [255, 255, 255], 2)
        cv2.line(new_im, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [255, 255, 255], 2)
    if display_progress:
        cv2.imshow('vertical', new_im)
        cv2.waitKey()

    correlation = np.count_nonzero(line_img == img_vertical) - difference


    if correlation > VERTICAL_MARGIN:
        sel = np.logical_and(line_img == img_vertical, line_img != 0)
        pic_chunk = np.where(sel is True, np.uint8(255), np.uint8(0))
        pic = pic + pic_chunk
        pic[pic > 255] = 255
        # cv2.imshow('qwert', pic)
        # cv2.waitKey()


    correlation_list.append(correlation)

    for n in range(1, len(line)):
        cv2.line(line_img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [0, 0, 0], 2)

    # print("correlation", correlation, 'coords', leftmost_row + i)
    if first_bot and correlation > VERTICAL_MARGIN:
        i += 1
        line = shift_right_line(line)
        continue
    elif first_bot and correlation < VERTICAL_MARGIN:
        first_bot = False
        i += 1
        continue

    if correlation > VERTICAL_MARGIN: #and not after_line:    # still in ROI
        # get local maximum
        local_max.append(correlation)
        local_ind.append(i)
        after_line = False

    if correlation < VERTICAL_MARGIN and not after_line:  # ROI is over
        maximum = max(local_max)
        max_index = local_ind[local_max.index(maximum)]

        # print('initial', leftmost_row + max_index)
        # print('passed', leftmost_row + max_index + (line[0][1]-line[-1][1]))
        # print('line', line[0][1])

        # if line[0][1]>line[-1][1]:

        # pic = cv2.erode(pic, np.ones((1, 2), np.uint16), iterations=2)
        # cv2.imshow('pic', pic)
        # cv2.waitKey()


        line = get_next_vert_line(leftmost_row + max_index + (line[0][1]-line[-1][1]), pic, hor_lines_y)
        # line = get_next_vert_line(leftmost_row + max_index, pic, hor_lines_y)
        # else:
        #     line = get_next_vert_line(leftmost_row + max_index, img_vertical, hor_lines_y)
        l = copy.deepcopy(line)
        vertical_lines.append(l)
        # vertical_lines.append(line)
        local_max = []
        local_ind = []
        pic = np.zeros(img_horizontal.shape)
        after_line = True
        first_bot = True
        # if line[0][1]>line[-1][1]:
        line = shift_right_line(line, 20)
        i = line[-1][1]-leftmost_row+1
        # else:
        #     i = l[0][1]-leftmost_row
        continue
    i += 1

finish = time.time() - start
print(finish)
# small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow('img', img)
# cv2.waitKey()

for line in vertical_lines:
    for n in range(1, len(line)):
        cv2.line(img, (line[n - 1][1], line[n - 1][0]), (line[n][1], line[n][0]), [0, 0, 255], 1)
cv2.imshow('img', img)
cv2.waitKey()

if display_progress:
    plt.plot(correlation_list)
    plt.show()
