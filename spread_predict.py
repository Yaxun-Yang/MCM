import numpy as np
import pandas as pd
import xlrd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math


class MyDataset(Dataset):
    def __init__(self, file_path, attri_num):
        self.file_path = file_path
        self.train_x = np.empty([621, attri_num], float)
        self.train_y = np.empty([621, 1], float)
        self.x_std = np.empty([1, attri_num], float)
        self.x_mean = np.empty([1, attri_num], float)
        self.data_process()

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return len(self.train_x)

    def data_process(self):

        csv_x_data = pd.read_excel(self.file_path, sheet_name=0, header=None)
        # csv_x_data = pd.read_table(self.file_path, sep=' ', header=None)
        self.train_x = csv_x_data.to_numpy()
        # self.train_x = np.delete(self.train_x, -1, axis=1)
        csv_y_data = pd.read_excel("tempdata/2019_1.xlsx", sheet_name=1, header=None)
        self.train_y = csv_y_data.to_numpy()


        # 标准化训练集
        self.x_mean = np.mean(self.train_x, axis=0)
        self.x_std = np.std(self.train_x, axis=0)
        self.train_x = np.divide(np.subtract(self.train_x, self.x_mean), self.x_std)
        # self.train_x = np.concatenate((np.ones([32561, 1]), self.train_x), axis=1).astype(float)


def sigmod(z):
    return 1 / (1 + np.exp(z))


def train(file_str, file_num, attri_num):
    data_set = MyDataset('tempdata/'+file_str+'.xlsx', attri_num)
    # data_set = MyDataset('tempdata/'+file_str+'.txt', attri_num)
    class_0_id = []
    class_1_id = []

    for i in range(data_set.train_y.shape[0]):
        if data_set.train_y[i] == 0:
            class_0_id.append(i)
        else:
            class_1_id.append(i)

    class_0 = data_set.train_x[class_0_id]
    class_1 = data_set.train_x[class_1_id]

    # print(class_0, class_1)

    mean_0 = np.mean(class_0, axis=0)
    mean_1 = np.mean(class_1, axis=0)

    # print(mean_0)
    n = class_0.shape[1]

    cov_0 = np.zeros([n, n])
    cov_1 = np.zeros([n, n])

    for i in range(class_0.shape[0]):
        cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [class_0[i] - mean_0]) / class_0.shape[0]
        # print(np.dot(np.transpose(class_0[i] - mean_0), class_0[i] - mean_0) / class_0.shape[0])

    for i in range(class_1.shape[0]):
        cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [class_1[i] - mean_1]) / class_1.shape[0]

    cov = (cov_0*class_0.shape[0] + cov_1*class_1.shape[0])/(class_0.shape[0] + class_1.shape[0])

    w = np.transpose(mean_0 - mean_1).dot(np.linalg.inv(cov))
    b = (-0.5) * mean_0.dot(np.linalg.inv(cov)).dot(mean_0) + 0.5 * mean_1.dot(np.linalg.inv(cov)).dot(mean_1)\
        + np.log(float(class_0.shape[0]) / float(class_1.shape[0]))
    # print(w.shape)
    w = np.append(w, b)
    # print(w.shape)

    np.save('tempdata/'+str(file_num)+'.npy', w)

    return data_set.x_mean, data_set.x_std


def draw(result, file_str):
    im = plt.imshow(result, cmap=plt.cm.summer)
    if file_str == "2020":
        plt.title("Asian Giant Hornets Spread Prediction")
    elif file_str == "2021":
        plt.title("Asian Giant Hornets Extinction Threshold")
    elif file_str == "2019":
        plt.title("Areas need to be investigated")
    # plt.colorbar(im)
    plt.axis("off")
    plt.show()


def test(x_mean, x_std, file_str, file_num):
    csv_data = pd.read_excel('tempdata/'+file_str+'.xlsx', header=None)
    # csv_data = pd.read_table('tempdata/'+file_str+'.txt', sep=' ', header=None)
    test_x = csv_data.to_numpy()
    w = np.load('tempdata/'+str(file_num)+'.npy')

    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            test_x[i][j] = (test_x[i][j] - x_mean[j]) / x_std[j]

    test_x = np.concatenate((np.ones([621, 1]), test_x), axis=1).astype(float)
    result = sigmod(np.dot(test_x, np.transpose(w)))
    result = np.resize(result, [23, 27])
    re_result = np.zeros_like(result)
    for i in range(len(result)):
        for j in range(len(result[i])):
            re_result[i][j] = 1 if result[i][j] > 0.5 else 0
    print(np.sum(re_result))

    return result
    # with open('testResult.csv', 'w') as file:
    #     for i in range(16281):
    #         file.write(str(result[i]) + '\n')


def draw_four(result):
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    titles = ["Not Updated", "Prior Knowledge", "Food Reduction", "Add Attributes"]
    for i in range(4):
        im = ax[i].imshow(result[i], cmap=plt.cm.summer)
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    fig.colorbar(im, ax=[ax[0], ax[1], ax[2], ax[3]], fraction=0.03, pad=0.05)
    plt.suptitle("Heat Map of Changes After Model Update")
    plt.show()


def model_update():
    result = []
    x_mean_1, x_std_1 = train("2019_1", 1, 5)
    result.append(test(x_mean_1, x_std_1, "2020_1", 1))

    x_mean_2, x_std_2 = train("2019_2", 2, 5)
    result.append(test(x_mean_2, x_std_2, "2020_2", 2))

    x_mean_3, x_std_3 = train("2019_3", 3, 5)
    result.append(test(x_mean_3, x_std_3, "2020_3", 3))

    x_mean_4, x_std_4 = train("2019_4", 4, 6)
    result.append(test(x_mean_4, x_std_4, "2020_4", 5))

    for i in range(4):
        print(result[i].mean())
    draw_four(result)


def inv_count(result):
    mtx = np.zeros([23, 27])
    class_one = [1, 1, 1]
    class_one_list = [1, 1, 1]
    class_two = [-1, -1, -1]
    class_two_list = [0, 0, 0]
    for i in range(23):
        for j in range(27):
            if 0.6 > result[i][j] > 0.4:
                mtx[i][j] += math.fabs(result[i][j] - 0.5) - 1
                if mtx[i][j] < class_one[2]:
                    if mtx[i][j] < class_one[1]:
                        if mtx[i][j] < class_one[0]:
                            class_one[2] = class_one[1]
                            class_one_list[2] = class_one_list[1]
                            class_one[1] = class_one[0]
                            class_one_list[1] = class_one_list[0]
                            class_one[0] = mtx[i][j]
                            class_one_list[0] = [i, j]

                        else:
                            class_one[2] = class_one[1]
                            class_one_list[2] = class_one_list[1]
                            class_one[1] = mtx[i][j]
                            class_one_list[1] = [i, j]

                    else:
                        class_one[0] = mtx[i][j]
                        class_one_list[0] = [i, j]
            elif result[i][j] > 0.9:
                mtx[i][j] += result[i][j]
                if mtx[i][j] > class_two[2]:
                    if mtx[i][j] > class_two[1]:
                        if mtx[i][j] > class_two[0]:
                            class_two[2] = class_two[1]
                            class_two_list[2] = class_two_list[1]
                            class_two[1] = class_two[0]
                            class_two_list[1] = class_two_list[0]
                            class_two[0] = mtx[i][j]
                            class_two_list[0] = [i, j]

                        else:
                            class_two[2] = class_two[1]
                            class_two_list[2] = class_two_list[1]
                            class_two[1] = mtx[i][j]
                            class_two_list[1] = [i, j]
                    else:
                        class_two[0] = mtx[i][j]
                        class_two_list[0] = [i, j]

    print(class_two, class_two_list)
    print(class_one, class_one_list)
    draw(mtx, "2019")


def main():
    x_mean, x_std = train("2019_1", 1, 5)
    result = test(x_mean, x_std, "2020_1", 1)
    # draw(result, 2020)

    inv_count(result)


if __name__ == '__main__':
    main()
