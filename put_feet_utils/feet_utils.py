import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import scipy.signal
import os
import pandas as pd
from skimage.transform import resize


def get_mean(signal: np.ndarray, axis=0):
    return signal.mean(axis=axis)

def get_std_dev(signal: np.ndarray, axis=0):
    return signal.std(axis=axis)

def get_power(signal: np.ndarray, axis=0, fs=1000):
    f_welch, S_xx_welch = scipy.signal.welch(signal, fs=fs, axis=0)
    df_welch = f_welch[1] - f_welch[0]
    return np.sum(S_xx_welch, axis=axis) * df_welch

def get_energy(signal: np.ndarray, axis=0):
    N = signal.shape[0]
    Xk = np.fft.fft(signal)
    return np.sum(np.abs(Xk) ** 2, axis=axis) / N

def pca(data: np.ndarray, labels: np.ndarray, n_components=3):
    X = data.copy()
    y = labels.copy()


    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig=fig,auto_add_to_figure=False, rect=[0, 0, .95, 1], elev=48, azim=134)
    fig.add_axes(ax)

    plt.cla()
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [('box', 0), ('pufa', 1), ('profil', 2), ('gasnica',3)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def check_pca(path_to_data):
    data_dirs = []
    class_names = []
    class_counter = 0
    for subdir, dirs, files in os.walk(path_to_data):
        data_dirs.append(subdir)
    data_dirs.pop(0)
    column_names = ["class", "force_x_mean", "force_y_mean", "force_z_mean"]
    df_all = pd.DataFrame(columns=column_names)

    for subdir in data_dirs:
        df_log_file = pd.read_csv(os.path.join(subdir, 'log.csv'))
        print(subdir)
        for i in range(len(df_log_file)):
            forces_path = os.path.join(subdir, os.path.basename(df_log_file.loc[i, 'forces_path']))
            df_forces = pd.read_csv(os.path.join('', forces_path))

            # standaryzacja
            # df_forces['0'] = (df_forces['0']-df_forces['0'].mean())/df_forces['0'].std()
            # df_forces['1'] = (df_forces['1'] - df_forces['1'].mean()) / df_forces['1'].std()
            # df_forces['2'] = (df_forces['2'] - df_forces['2'].mean() )/ df_forces['2'].std()
            #

            df_f_x = df_forces['0'].mean()
            df_f_y = df_forces['1'].mean()
            df_f_z = df_forces['2'].mean()

            df_all = df_all.append(
                {"class": int(class_counter), "force_x_mean": float(df_f_x), "force_y_mean": float(df_f_y),
                 "force_z_mean": float(df_f_z)}, ignore_index=True)
        class_counter += 1

    print(df_all)
    labels = df_all['class'].to_numpy()
    data = df_all.loc[:, df_all.columns != 'class'].to_numpy()
    pca(data,labels)

def df_resample(df1, num=1):
    df2 = pd.DataFrame()
    for key, value in df1.iteritems():
        temp = value.to_numpy() / value.abs().max()  # normalize
        resampled = resize(temp, (num, 1), mode='edge') * value.abs().max()  # de-normalize
        df2[key] = resampled.flatten().round(2)
    return df2

def preprocess_data(path_to_data):
    data_dirs = []
    class_names = []
    class_counter = 0
    for subdir, dirs, files in os.walk(path_to_data):
        data_dirs.append(subdir)
    data_dirs.pop(0)
    column_names = ["force_x_mean", "force_y_mean", "force_z_mean",
                    "quat_0_x", "quat_0_y", "quat_0_z", "quat_0_w",
                    "quat_1_x", "quat_1_y", "quat_1_z", "quat_1_w",
                    "quat_2_x", "quat_2_y", "quat_2_z", "quat_2_w",
                    "quat_3_x", "quat_3_y", "quat_3_z", "quat_3_w"]


    for subdir in data_dirs:
        df_log_file = pd.read_csv(os.path.join(subdir, 'log.csv'))
        for i in range(len(df_log_file)):
            forces_path = os.path.join(subdir, os.path.basename(df_log_file.loc[i, 'forces_path']))
            df_forces = pd.read_csv(os.path.join('', forces_path))
            quat_path = os.path.join(subdir, os.path.basename(df_log_file.loc[i, 'quat_path']))
            df_quat = pd.read_csv(os.path.join('', quat_path))
            df_forces = df_forces.rename(columns={"0": "force_x_mean",
                                                  "1": "force_y_mean",
                                                  "2": "force_z_mean"})
            df_quat = df_quat.rename(columns={"0": "quat_0_x",
                                              "1": "quat_0_y",
                                              "2": "quat_0_z",
                                              "3": "quat_0_w",
                                              "4": "quat_1_x",
                                              "5": "quat_1_y",
                                              "6": "quat_1_z",
                                              "7": "quat_1_w",
                                              "8": "quat_2_x",
                                              "9": "quat_2_y",
                                              "10": "quat_2_z",
                                              "11": "quat_2_w",
                                              "12": "quat_3_x",
                                              "13": "quat_3_y",
                                              "14": "quat_3_z",
                                              "15": "quat_3_w"
                                              })
            df_forces = df_forces.drop(['Unnamed: 0'], axis=1)
            df_quat = df_quat.drop(['Unnamed: 0'], axis=1)
            df_forces = df_resample(df_forces, 3000)
            df_quat = df_resample(df_quat, 3000)
            df_combined = pd.concat([df_forces,df_quat], axis=1)
            df_combined.plot()
            plt.show()


def plot_data(path_to_data):
    data_dirs = []
    class_names = []
    class_counter = 0
    for subdir, dirs, files in os.walk(path_to_data):
        data_dirs.append(subdir)
    data_dirs.pop(0)
    column_names = ["class", "force_x_mean", "force_y_mean", "force_z_mean",
                    "quat_0_x", "quat_0_y", "quat_0_z", "quat_0_w",
                    "quat_1_x", "quat_1_y", "quat_1_z", "quat_1_w",
                    "quat_2_x", "quat_2_y", "quat_2_z", "quat_2_w",
                    "quat_3_x", "quat_3_y", "quat_3_z", "quat_3_w"]
    df_data = pd.DataFrame(columns=column_names)

    for subdir in data_dirs:
        df_log_file = pd.read_csv(os.path.join(subdir, 'log.csv'))
        for i in range(len(df_log_file)):
            forces_path = os.path.join(subdir, os.path.basename(df_log_file.loc[i, 'forces_path']))
            df_forces = pd.read_csv(os.path.join('', forces_path))
            quat_path = os.path.join(subdir, os.path.basename(df_log_file.loc[i, 'quat_path']))
            df_quat = pd.read_csv(os.path.join('', quat_path))

            df_forces = df_resample(df_forces,3000)
            df_quat = df_resample(df_quat,3000)

            plt.plot(df_forces['0'])
            plt.plot(df_quat['0'])
            plt.plot(df_quat['1'])
            plt.plot(df_quat['2'])
            plt.plot(df_quat['3'])
            plt.show()





