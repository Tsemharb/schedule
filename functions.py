import copy


def discriminate(img):
    computer_shot = True
    cropped = img[int(img.shape[0] / 2 - 200):int(img.shape[0] / 2 + 200),
               int(img.shape[1] / 2 - 200):int(img.shape[1] / 2 + 200)]
    same_pix_qty = 0

    for row in range(cropped.shape[0]):
        for col in range(cropped.shape[1]):
            same_pix_qty += int(cropped[row][col][0] == cropped[row][col][1] == cropped[row][col][2])

    if int(same_pix_qty/(cropped.shape[0]*cropped.shape[1]*cropped.shape[2])*100) >= 5:
        computer_shot = False
    return computer_shot


def get_lower_left(img):
    point = None
    for j in range(int(img.shape[1]*0.25)):
        if point is not None:
            break
        for i in range(img.shape[0]-1, int(img.shape[0]*0.7), -1):
            if img[i][j] == 255:
                point = [i, j]
                break
    return point


def get_lower_right(img):
    point = None
    for row in range(img.shape[0] - 1, int(img.shape[0] * 0.7), -1):
        if point is not None:
            break
        for col in range(img.shape[1] - 1, int(img.shape[1] * 0.75), -1):
            if img[row][col] == 255:
                point = [row, col]
                break
    for row in range(point[0]-1, point[0]-8, -1):
        for col in range(point[1], img.shape[1]-1):
            if img[row][col] == 255:
                point = [row, col]
    while True:
        if point[0] < img.shape[0]-1 and img[point[0]+1][point[1]] == 255:
            point[0] = point[0]+1
        else:
            break
    return point


def get_bottom_line(right_bottom_point, img, vert_lines_x, a):
    x_coords = [x for x in vert_lines_x if x < right_bottom_point[1]]
    # print(x_coords)
    point = right_bottom_point
    bottom_line = [point]
    for x in x_coords[::-1]:
        if img[point[0], x] != 255:
            for i in range(1, 5): #search within vertical range

                if img[point[0]-i, x] != 255:
                    pass
                else:
                    next_point = [point[0]-i, x]
                    break

                if point[0]+i < img.shape[0] and img[point[0]+i, x] != 255:
                    pass
                else:
                    if point[0]+i < img.shape[0]-1:
                        next_point = [point[0]+i, x]
                    break

                if i == 4:
                    try:
                        vertical_difference = bottom_line[-1][0] - bottom_line[-2][0]
                        next_point = [point[0] + vertical_difference, x]
                    except:
                        next_point = [point[0], x]
        else:
            next_point = [point[0], x]

        bottom_line.append(next_point)
        point = next_point

    # initial = math.ceil(left_bottom_point[1]/chunk_length) - 1
    # initial_vert_distance = bottom_line[initial][0] - bottom_line[0][0]

    last_point = [bottom_line[0][0], img.shape[1]-1]
    bottom_line.insert(0, last_point)

    return bottom_line


def get_next_hor_line(row_number, img, chunk_length, x_vert_coords):
    # print("row number", row_number)
    # print(x_vert_coords)
    # with open('lines2', 'a') as file:
    #     file.write(str(row_number))
    #     file.write('\n')
    rightmost_point_x = 0
    shift = 0
    neg = False

    for i in range(img.shape[1]-1, int(img.shape[1] * 0.8), -1):
        for n in range(20):

            if img[row_number - n][i] == 255:
                rightmost_point = img[row_number - n][i]
                rightmost_point_x = i
                neg = True
                break

            if img[row_number + n][i] == 255:
                rightmost_point = img[row_number + n][i]
                rightmost_point_x = i
                break

            if n == 19:
                rightmost_point = img[row_number][i]

        if rightmost_point == 255:
            shift = copy.deepcopy(n)
            if neg:
                shift = - shift
            break

    rightmost_point = [row_number + shift, rightmost_point_x]
    # print("rightmost",  rightmost_point)
    x_coords = [x for x in x_vert_coords if x < rightmost_point[1]]
    # print(x_coords)
    point = rightmost_point
    line = [point]
    for x in x_coords[::-1]:
        if img[point[0], x] != 255:
            for i in range(1, 15): #search within vertical range #5

                if img[point[0]-i, x] != 255:
                    pass
                else:
                    next_point = [point[0]-i, x]
                    break

                if img[point[0]+i, x] != 255:
                    pass
                else:
                    next_point = [point[0]+i, x]
                    break

                if i == 14:
                    try:
                        vertical_difference = line[-1][0] - line[-2][0]
                        next_point = [point[0] + 0, x]#vertical_difference, x]
                    except:
                        next_point = [rightmost_point[0], x]

        else:
            next_point = [point[0], x]

        line.append(next_point)
        point = next_point

    # initial = math.ceil(leftmost_point[1]/chunk_length) - 1
    # initial_vert_distance = line[initial][0] - line[0][0]
    last_point = [line[0][0], img.shape[1]-1]
    line.insert(0, last_point)

    return line


def get_leftmost_line(left_bottom_point, img, hor_lines_y):
    y_coords = [y for y in hor_lines_y if y < left_bottom_point[0]]
    # print(y_coords)
    point = left_bottom_point
    leftmost_line = [left_bottom_point]
    for y in y_coords[::-1]:
        if img[y, point[1]] != 255:
            for i in range(1, 10): #search within horizontal range

                if img[y, point[1]-i] != 255:
                    pass
                else:
                    next_point = [y, point[1]-i]
                    break

                if img[y, point[1]+i] != 255:
                    pass
                else:
                    next_point = [y, point[1]+i]
                    break

                if i == 9:
                    # try:
                    #     # horizontal_difference = leftmost_line[-1][1] - leftmost_line[-2][1]
                    #     next_point = [y, point[1] + 0]#horizontal_difference]
                    # except:
                    next_point = [y, point[1]]
        else:
            next_point = [y, point[1]]

        leftmost_line.append(next_point)
        point = next_point

    last_point = [img.shape[0] - 1, leftmost_line[0][1]]
    leftmost_line.insert(0, last_point)

    return leftmost_line


def get_next_vert_line(column_number, img, y_hor_coords):
    leftmost_point_y = 0
    shift = 0
    neg = False

    for i in range(img.shape[0] - 1, int(img.shape[0] * 0.7), -1):
        for n in range(30):
            # print(column_number+n)
            if column_number + n < img.shape[1] and img[i][column_number + n] == 255:
                leftmost_point = img[i][column_number + n]
                leftmost_point_y = i
                break

            if img[i][column_number - n] == 255:
                leftmost_point = img[i][column_number - n]
                leftmost_point_y = i
                neg = True
                break

            if n == 29:
                leftmost_point = img[i][column_number]

        if leftmost_point == 255:
            shift = copy.deepcopy(n)
            if neg:
                shift = - shift
            break

    leftmost_point = [leftmost_point_y, column_number + shift]
    # print("leftmost",  leftmost_point)
    y_coords = [y for y in y_hor_coords if y < leftmost_point[0]]
    point = leftmost_point
    line = [point]
    for y in y_coords[::-1]:
        if img[y, point[1]] != 255:
            for i in range(1, 20):  # search within horizontal range

                # add sideways walk

                if img[y, point[1]-i] != 255:
                    pass
                else:
                    next_point = [y, point[1]-i]
                    break

                if point[1]+i < img.shape[1]-1:
                    if img[y, point[1]+i] != 255:
                        pass
                    else:
                        next_point = [y, point[1]+i]
                        break

                if i == 19:
                    # next_point = [y, point[1]]
                    try:
                        horizontal_difference = line[-1][1] - line[-2][1]
                        next_point = [y, point[1] +0]# horizontal_difference]
                    except:
                        next_point = [y, point[1]]


        else:
            next_point = [y, point[1]]

        line.append(next_point)
        point = next_point

    last_point = [img.shape[0] - 1, line[0][1]]
    line.insert(0, last_point)

    # print(line)

    return line


def raise_line(line, shift=1):
    for point in line:
        point[0] = point[0] - shift
    return line


def shift_right_line(line, shift=1):
    for point in line:
        point[1] = point[1] + shift
    return line
