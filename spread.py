import xlrd
from pylab import *
import math
import random

path = "2021MCMProblemC_DataSet.xlsx"
const_x = 23
const_y = 27


def hornets():
    data = xlrd.open_workbook(path)
    sheet = data.sheets()[0]
    date_rows_num = sheet.nrows
    positive_list = []
    for i in range(1, date_rows_num):
        if sheet.cell(i, 3).value == "Positive ID":
            positive_list.append((xlrd.xldate_as_tuple(sheet.cell(i, 1).value, 0)[:3],
                                  sheet.cell(i, 6).value, sheet.cell(i, 7).value))

    positive_list = sorted(positive_list, key=lambda element: element[0])

    for i in range(len(positive_list)):
        theta = np.linspace(0, 2 * np.pi, 800)

        x, y = np.cos(theta) * 0.1782, np.sin(theta) * 0.27
        x = np.add(positive_list[i][2], x)
        y = np.add(positive_list[i][1], y)
        print(positive_list[i][0])
        if positive_list[i][0][0] == 2020:
            plt.scatter(positive_list[i][2], positive_list[i][1], c='r')
            plot(x, y, color='r', linewidth=2.0)

        else:
            plt.scatter(positive_list[i][2], positive_list[i][1], c='g')
            plot(x, y, color='g', linewidth=2.0)

    plt.ylim((47.105, 50.895))
    plt.xlim((-124.5, -122))
    plt.show()


def queen_grid():
    rgb = (235 / 255, 78 / 255, 6 / 255, 1)
    figure(figsize=(10, 10), dpi=80)

    theta = np.linspace(0, 2 * np.pi, 800)

    x, y = np.cos(theta) * 10.7193, np.sin(theta) * 10.7193

    num = 1000
    v = np.linspace(0, 1, num)
    v.shape = (num, 1)
    x1 = v * x
    y1 = v * y

    for i in range(num):
        plot(10.5 + x1[i], 10.5 + y1[i], color=(rgb[0] * v[i][0] * 0.4 + 0.6, rgb[1], rgb[2], 1 - v[i][0]),
             linewidth=1.0)

    plot(10.5 + x, 10.5 + y, color=(rgb[0], rgb[1], rgb[2], 0.3))
    plt.xticks(np.arange(0, 23, 1))
    plt.yticks(np.arange(0, 23, 1))
    plt.grid(True)
    plt.show()


def latitude_transform(latitude):
    deta = 49.061 - latitude
    km = 111 * deta
    deta_num = km / 1.4
    return math.floor(deta_num)


def longitude_transform(longitude):
    deta = longitude + 122.72
    km = deta * 73.5375
    deta_num = km / 1.4
    return math.floor(deta_num)


def fill_queen(mtx, center):
    for i in range(-10, 10):
        for j in [4, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 7, 6, 4]:
            for k in range(-j, j):
                if const_x > center[0] + i >= 0 and const_y > center[1] + k >= 0:
                    x = math.sqrt(math.pow(i, 2) + math.pow(j - 10, 2)) * 1.4
                    mtx[center[0] + i][center[1] + k] += 0.0008 * x * x - 0.057 * x + 1
    return mtx


def fill_queen_2(mtx, center):
    for i in range(-10, 10):
        for j in [4, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 7, 6, 4]:
            for k in range(-j, j):
                if const_x > center[0] + i >= 0 and const_y > center[1] + k >= 0:
                    x = math.sqrt(math.pow(i, 2) + math.pow(j - 10, 2)) * 1.4
                    mtx[center[0] + i][center[1] + k] += 0.0006 * x * x - 0.048 * x + 0.9
    return mtx


def fill_others(mtx, center):
    for i in range(-1, 1):
        for j in range(-1, 1):
            if const_x > center[0] + i >= 0 and const_y > center[1] + j >= 0:
                mtx[center[0] + i][center[1] + j] += 1
    return mtx


def hornets_mark(mtx, now_time):
    data = xlrd.open_workbook(path)
    sheet = data.sheets()[0]
    date_rows_num = sheet.nrows

    for i in range(1, date_rows_num):
        longitude = sheet.cell(i, 7).value
        latitude = sheet.cell(i, 6).value
        if -122.206 > longitude >= -122.72 and 49.061 >= latitude > 48.771:
            if sheet.cell(i, 3).value == "Positive ID" \
                    and xlrd.xldate_as_tuple(sheet.cell(i, 1).value, 0)[0] == now_time:
                if xlrd.xldate_as_tuple(sheet.cell(i, 1).value, 0)[:3] == (2020, 5, 27) \
                        or xlrd.xldate_as_tuple(sheet.cell(i, 1).value, 0)[:3] == (2020, 6, 7):
                    mtx[0] = fill_queen(mtx[0], (latitude_transform(latitude), longitude_transform(longitude)))
                else:
                    mtx[1][latitude_transform(latitude)][longitude_transform(longitude)] += 1
                    mtx[1] = fill_others(mtx[1], (latitude_transform(latitude), longitude_transform(longitude)))
            elif sheet.cell(i, 3).value == "Negative ID":

                mtx[2] = fill_others(mtx[2], (latitude_transform(latitude), longitude_transform(longitude)))

    return mtx


def plant_mark(mtx):
    data = xlrd.open_workbook(path)
    sheet = data.sheets()[1]
    for i in range(const_x):
        for j in range(const_y):
            if sheet.cell(i, j).value == 1:
                mtx[3] = fill_others(mtx[3], (i, j))
            elif sheet.cell(i, j).value == 2:
                mtx[4] = fill_others(mtx[4], (i, j))
            elif sheet.cell(i, j).value == 4:
                mtx[5] = fill_others(mtx[5], (i, j))
            elif sheet.cell(i, j).value == 6:
                mtx[6] = fill_others(mtx[6], (i, j))
            elif sheet.cell(i, j).value != 0:
                mtx[7] = fill_others(mtx[7], (i, j))
    return mtx


def heat_draw(mtx):
    fig, ax = plt.subplots(2, 4)
    ax = ax.flatten()
    titles = ["Queen", "Worker", "Other Hornets", "Berry", "Cereal Grain", "Hay/Silage", "Pasture", "Other Plant"]
    for i in range(8):
        im = ax[i].imshow(mtx[i], cmap=plt.cm.summer)
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    fig.colorbar(im, ax=[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7]], fraction=0.03, pad=0.05)
    plt.suptitle("2020 Attributes Heat Map")
    plt.show()


def file_write(mtx, file_num, now_time):
    with open('tempdata/data_'+str(now_time)+'_'+str(file_num)+'.txt', 'w') as f:
        for i in range(const_x):
            for j in range(const_y):
                for k in range(8):
                    f.write(str(mtx[k][i][j]) + " ")
                f.write('\n')


def result_file_write():
    data = xlrd.open_workbook(path)
    sheet = data.sheets()[0]
    date_rows_num = sheet.nrows

    mtx = np.zeros([const_x, const_y])
    for i in range(1, date_rows_num):
        longitude = sheet.cell(i, 7).value
        latitude = sheet.cell(i, 6).value
        if -122.206 > longitude >= -122.72 and 49.061 >= latitude > 48.771:
            if sheet.cell(i, 3).value == "Positive ID" and xlrd.xldate_as_tuple(sheet.cell(i, 1).value, 0)[0] == 2020:
                mtx[latitude_transform(latitude)][longitude_transform(longitude)] = 1

    for i in range(const_x):
        for j in range(const_y):
            with open("tempdata/data_result_2020.txt", 'a') as f:
                f.write(str(mtx[i][j])+'\n')


def more_attribute():
    mtx = np.ones([9, const_x, const_y])
    mtx = hornets_mark(mtx, 2019)
    mtx = plant_mark(mtx)
    mtx[8] = np.random.rand(const_x, const_y)
    file_write(mtx, 4, 2019)

    mtx2 = np.ones([9, const_x, const_y])
    mtx2 = hornets_mark(mtx2, 2020)
    mtx2 = plant_mark(mtx2)
    mtx2[8] = np.random.rand(const_x, const_y)
    file_write(mtx2, 4, 2020)


def main():
    mtx = np.ones([8, const_x, const_y])
    mtx = hornets_mark(mtx, 2021)
    mtx = plant_mark(mtx)
    file_write(mtx, "all_dead", 2021)
    # heat_draw(mtx)


if __name__ == '__main__':
    main()
