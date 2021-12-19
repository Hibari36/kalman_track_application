
import json
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from scipy.optimize import curve_fit

# 预测矩阵，模型为匀速直线运动模型，不考虑加速度
A = np.array([[1., 0., 0., 1., 0., 0.],
              [0., 1., 0., 0., 1., 0.],
              [0., 0., 1., 0., 0., 1.],
              [0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0., 1.]])
# 观测矩阵，仅能从位置，速度信息中观测到位置信息
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])


class TrackData:
    def __init__(self, json_path):
        with open(json_path, 'rt') as file:
            data = json.load(file)
            track_data = data['kml']['Document']['Folder']['Placemark'][2]['LineString']['coordinates']
            track_coordinates = track_data.split(' ')
            coordinates = []
            for coordinate in track_coordinates:
                coordinates.append(coordinate.split(','))
            coordinates = np.array(coordinates).astype(float)
            self.track = coordinates
        self.floor_track = np.zeros_like(self.track)
        self.len = len(self.track)
        self.predict_state = np.zeros((self.len, 6))
        self.update_state = np.zeros((self.len, 6))
        self.predict_error = np.zeros((self.len, 6, 6))
        self.update_error = np.zeros((self.len, 6, 6))
        self.kalman_gain = np.zeros((self.len, 6, 3))

    def fit_track(self):
        x = self.floor_track[:, 0]
        y = self.floor_track[:, 1]
        z = self.floor_track[:, 2]
        times = np.arange(0, len(x), 1)
        x_poly = np.polyfit(times, x, deg=1)
        y_poly = np.polyfit(times, y, deg=1)
        z_poly = np.polyfit(times, z, deg=1)
        fit_x = np.polyval(x_poly, times)
        fit_y = np.polyval(y_poly, times)
        fit_z = np.polyval(z_poly, times)
        return np.array([fit_x, fit_y, fit_z]).transpose()

    def kalman(self):
        # 为初始状态赋值
        fit_data = self.fit_track()     # 线性最小二乘拟合解
        self.predict_state[0, 0:3] = self.update_state[0, 0:3] = fit_data[0]
        self.update_state[0, 3:6] = (fit_data[-1] - fit_data[0]) / (self.len - 1)
        measure_error_vector = np.array([225, 225, 900, 450, 450, 1800])
        measure_error = np.diag(measure_error_vector[0:3])
        state_error = np.zeros(6)
        state_error[0:3] = measure_error_vector[0:3] / self.len
        state_error[3:6] = measure_error_vector[3:6] / (self.len - 1)
        self.predict_error[0] = self.update_error[0] = np.diag(state_error)

        for i in range(self.len - 1):
            # 使用初始状态 预测下一状态 x_k+1_k = A * x_k_k
            self.predict_state[i+1] = np.dot(A, self.update_state[i])
            print('predict state:\n', self.predict_state[i+1])
            print('observe state:\n', self.floor_track[i+1])
            # 状态预测值的方差  P_k+1_k = A * P_k_k * A_t + Q_k  模型准确，预测方差Q_k为0
            self.predict_error[i+1] = np.dot(np.dot(A, self.update_error[i]), A.transpose())
            # print('predict error:\n', self.predict_error[i+1])
            # 计算卡尔曼增益 K_k+1 = P_k+1_k * H_t * (H * P_k+1_k * H_t + R_k+1)^-1   H为观测矩阵，R_k+1 为观测误差，即GPS定位误差
            self.kalman_gain[i+1] = np.dot(np.dot(self.predict_error[i+1], H.transpose()),
                                           np.linalg.inv(np.dot(np.dot(H, self.predict_error[i+1]),
                                                                H.transpose()) + measure_error))
            # print('kalman gain:\n', self.kalman_gain[i+1])
            # 计算预测残差    y_k+1 = observe_k+1 - H * x_k+1_k
            residual = self.floor_track[i+1] - np.dot(H, self.predict_state[i+1])
            print('residual:\n', residual)
            # 更新预测状态 x_k+1_k+1 =  x_k+1_k + K_k+1 * y_k+1
            self.update_state[i+1] = self.predict_state[i+1] + np.dot(self.kalman_gain[i+1], residual)
            print('update state:\n', self.update_state[i+1])
            # 计算更新预测方差 P_k+1_k+1 = (I - K_k+1) * P_k+1_k
            self.update_error[i+1] = np.dot((np.eye(6) - np.dot(self.kalman_gain[i+1], H)),
                                            self.predict_error[i+1])
            # print('update error:\n', self.update_error[i+1])
        print('original states')
        print(self.floor_track)
        print('filter states')
        print(self.update_state)
        self.draw_figure(90, 0, self.update_state[:, 0], self.update_state[:, 1], self.update_state[:, 2])

    def draw_figure(self, elev_angle, hor_angle, x=None, y=None, z=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if x is not None:
            ax.plot(x, y, z, c='red')
        x = self.floor_track[:, 0]
        y = self.floor_track[:, 1]
        z = self.floor_track[:, 2]
        ax.plot(x, y, z, c='green')
        ax.scatter(x[0], y[0], z[0], c='red')
        ax.scatter(x[-1], y[-1], z[-1], c='green')
        ax.view_init(elev=elev_angle, azim=hor_angle)
        plt.show()

    # 将经纬度转换为平面坐标（粗略转换）
    def convert_track(self):
        coordinates = np.zeros((self.len, 3))
        for i in range(self.len):
            coord = self.track[i]
            x, y = get_coordinate(coord[1], coord[0])
            coordinates[i] = np.array([x, y, coord[2]])
        self.floor_track = coordinates


def get_coordinate(latitude, longitude):
    # 仅针对经度114 纬度30左右的位置进行坐标转换，同时以经度114，纬度30为坐标原点
    a = 6378137.0000    # 地球赤道半径
    b = 6356752.3142    # 地球极地半径
    # 纬度30度处的纬圈半径
    a_30 = a * cos(30/180*pi)
    x = a_30 * (longitude - 114) / 180 * pi
    y = b * (latitude - 30) / 180 * pi
    return x, y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trace_data = TrackData('trace.json')
    trace_data.convert_track()
    # trace_data.draw_figure(90, 0)
    trace_data.kalman()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
