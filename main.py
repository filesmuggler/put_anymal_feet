import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import re
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal

PATH = './logs/1203/log.csv'
classes_names = [('_cup', 0), ('_bottle', 1), ('_apple', 2)]

def plot_experiment(forces,acc,gyro, quat):
    fig_width = 20
    fig_height = 20
    f, axes = plt.subplots(nrows=4, ncols=1, figsize=(fig_width, fig_height))
    axes[0].plot(forces[:-1000])
    axes[1].plot(acc[:-1000])
    axes[2].plot(gyro[:-1000])
    axes[3].plot(quat[:-1000])
    plt.show()

def process_data(data):
    Fs = 400
    M = 1024
    dx = data['x'].to_numpy()
    dy = data['y'].to_numpy()
    dz = data['z'].to_numpy()
    dxf = fft(dx)
    nxf = fftfreq(len(dx),1/Fs)
    plt.stem(nxf, np.abs(dxf), use_line_collection=True)
    plt.grid()
    plt.show()

    freqs, times, Sx = signal.spectrogram(dx, fs=Fs)

    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(times, freqs, Sx, cmap='viridis')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]');
    print("xdd")
    plt.plot()
    plt.show()

def process_poke_point(pokepoint):
    pokes = re.findall(r'\d+\.\d+', pokepoint)
    for index, item in enumerate(pokes):
        pokes[index] = float(item)
    return pokes

def process_quat(quat):
    pass

def analyze_data(forces=None,acc=None,gyro=None,pokepoint=None,quat=None):
    if forces is not None:
        process_data(forces)
    # if acc is not None:
    #     process_data(acc)
    # if gyro is not None:
    #     process_data(gyro)
    # if pokepoint is not None:
    #     process_poke_point(pokepoint)
    # if quat is not None:
    #     process_quat(quat)

def old_analyze_data(forces, acc, gyro, poke_point):
    pokes = re.findall(r'\d+\.\d+', poke_point)
    for index, item in enumerate(pokes):
        pokes[index] = float(item)
    forces_avg = forces.mean()
    acc_avg = acc.mean()
    gyro_avg = gyro.mean()
    return forces_avg, acc_avg, gyro_avg, pokes

def read_experiment(row_path):
    obj_class = re.findall(r'_\D\w+',row_path['forces_path'].split("/")[-1])
    obj_id = ''
    for name, label in classes_names:
        if obj_class[0] == name:
            obj_id = label

    forces = pd.read_csv(os.path.join("logs/1203",row_path['forces_path'].split("/")[-1]), index_col=0)
    acc = pd.read_csv(os.path.join("logs/1203",row_path['acc_path'].split("/")[-1]), index_col=0)
    gyro = pd.read_csv(os.path.join("logs/1203",row_path['gyro_path'].split("/")[-1]), index_col=0)
    poke_point = row_path["poke_point"]
    return forces,acc,gyro,poke_point,obj_id

def main():
    cl = ['fx_avg', 'fy_avg', 'fz_avg', 'ax_avg', 'ay_avg', 'az_avg', 'gx_avg','gy_avg','gz_avg','p_x','p_y','p_z','class']
    combined_data = pd.DataFrame(columns=cl)

    mylog = pd.read_csv(PATH)
    number_of_experiments = len(mylog)
    for i in range(number_of_experiments):
        forces_raw,acc_raw,gyro_raw,poke_point_raw, obj_id = read_experiment(mylog.loc[i,:])
        forces_p, acc_p, gyro_p, pokes_p = old_analyze_data(forces_raw, acc_raw, gyro_raw, poke_point_raw)
        #analyze_data(forces_raw, acc_raw, gyro_raw, poke_point_raw)
        a_row = pd.Series([forces_p[0],forces_p[1],forces_p[2],acc_p[0],acc_p[1],acc_p[2],gyro_p[0],gyro_p[1],gyro_p[2],pokes_p[0],pokes_p[1],pokes_p[2],obj_id],index=cl)
        row_df = pd.DataFrame([a_row])
        combined_data = pd.concat([row_df,combined_data],ignore_index=True)


    cm_data = combined_data.loc[:,combined_data.columns !='class']
    #
    pca = PCA(n_components=2)
    pca.fit(cm_data)
    cm_data = pca.transform(cm_data)
    y = combined_data['class']

    target_ids = range(len(classes_names))

    plt.figure(figsize=(6, 5))
    for i, c, label in zip(target_ids, 'rgbcmykw', classes_names):
        plt.scatter(cm_data[y == i, 0], cm_data[y == i, 1],
                    c=c, label=label)
    plt.legend()
    plt.show()


    print('xd')

if __name__ == '__main__':
    main()