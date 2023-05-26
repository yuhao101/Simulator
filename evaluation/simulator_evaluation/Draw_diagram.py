from evaluation.simulator_evaluation.path import *
from evaluation.simulator_evaluation.Evaluate import *
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle
from tqdm import tqdm
import sys


def get_best_model(result,q):
    model_error = []
    real_data = np.array(list(get_real_data(result).values()))
    model_error.append(list(abs(np.array(list(get_model_result_perfect_matching(result,q).values()))-real_data+np.array([0, 1000, 1000, 1000]))))
    model_error.append(list(abs(np.array(list(get_model_result_fcfs(result,q).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_production_function(result,q).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mm1(result,q).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mm1k(result,q).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_mmn(result,q).values()))-real_data)))
    model_error.append(list(abs(np.array(list(get_model_result_batch_matching(result,q).values()))-real_data)))
    model_error = np.array(model_error)
    best_model = np.argmin(model_error, axis=0)
    best_model_mape = model_error[best_model, list(np.arange(len(real_data)))]/real_data
    return best_model, best_model_mape


def get_model_name(labels):
    res = []
    for la in labels:
        model_num = int(re.split(r'[{}]',la)[1])
        res.append(model_list[model_num])
    return res


def draw_best_model(type, x, y, labels, errors, name, trip_time):
    fs = 20

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    t = trip_time
    # x_stack = np.array(
    #     [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    x_stack = np.array(
        [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*7.275/5
    # if True:
    #     x_stack = np.array([0.0, x.max()])
    #     # y1 = np.array(
    #     #     [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])*0
    #     y1 = np.array([0, 0])
    #     y2 = x_stack * t / 1 - y1
    #     y3 = x_stack * t / 0.7 - y1 - y2
    #     # y5 = np.array(
    #     #     [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]) - y1 - y2 - y3
    #     y5 = np.array([2000, 2000]) - y1 - y2 - y3
    #     y_stack = [y1, y2, y3, y5]
    # else:
    #     x_stack = np.array([0.0, 2000 * 0.7 / t, x.max()])
    #     # y1 = np.array(
    #     #     [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])*0
    #     y1 = np.array([0, 0, 0])
    #     y2 = x_stack * t / 1 - y1
    #     y3 = x_stack * t / 0.7 - y1 - y2
    #     y3[2] = 2000 - y1[2] - y2[2]
    #     # y5 = np.array(
    #     #     [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]) - y1 - y2 - y3
    #     y5 = np.array([2000, 2000, 2000]) - y1 - y2 - y3
    #     y_stack = [y1, y2, y3, y5]
    y1 = np.array(
            [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])*0
    y2 = x_stack * t / 1 - y1
    y3 = x_stack * t / 0.7 - y1 - y2
    y5 = np.array(
        [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]) - y1 - y2 - y3
    y_stack = [y1, y2, y3, y5]
    # x_stack[1] = 2000/t
    plt.stackplot(x_stack, y_stack, colors=['w', "w", "#808080", "#016795"], alpha=0.08)
    ax.plot(x_stack, x_stack * t / 1, linestyle='-', color="red", lw=1, label='$Q = N/t$', alpha=0.5)
    scatter = ax.scatter(x, y, c=labels, cmap='Dark2', s=errors * 80 + 10)
    handles, labels = scatter.legend_elements(prop="colors", alpha=1.0)

    labels = get_model_name(labels)

    legend1 = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.01, 1.0), title="Models", fontsize=fs)
    legend2 = ax.legend(loc=2, bbox_to_anchor=(1.01, 0.6), markerscale=0.5, fontsize=fs)
    plt.setp(legend1.get_title(), fontsize=fs)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    kw = dict(prop="sizes", fmt="{x:.0f}%", alpha=0.6,
              func=lambda s: ((s - 10) / 0.8))
    legend2 = ax.legend(*scatter.legend_elements(**kw),
                        loc=2, bbox_to_anchor=(1.01, 0.5), title="MAPE", fontsize=fs)
    plt.setp(legend2.get_title(), fontsize=fs)
    plt.xlabel('Arrival rate of orders (pax/sec)', fontsize=28)

    plt.ylabel('Number of drivers (veh)', fontsize=28)
    if not os.path.exists(load_path + 'Figures/' + type):
        os.mkdir(load_path + 'Figures/' + type)
    plt.savefig(load_path + 'Figures/' + type + '/' + name + '.jpg', dpi=600, bbox_inches='tight')
    plt.clf()


def get_file_list(num, mode, files_list):
    if mode == 'fix_driver':
        para_select = 3
        para_sort = 1
    elif mode == 'fix_order':
        para_select= 1
        para_sort = 3
    file_list = []
    for file in files_list:
        if float(file.split('_')[para_select]) == num:
            file_list.append(file)
    file_list.sort(key=lambda ele:float(re.split(r'[____]',ele)[para_sort]))
    return file_list


def get_data(result, q):
    res = []
    res.append(list(get_real_data(result).values()))
    res.append(list(get_model_result_perfect_matching(result, q).values()))
    res.append(list(get_model_result_fcfs(result,q).values()))
    res.append(list(get_model_result_production_function(result,q).values()))
    res.append(list(get_model_result_mm1(result,q).values()))
    res.append(list(get_model_result_mm1k(result,q).values()))
    res.append(list(get_model_result_mmn(result,q).values()))
    res.append(list(get_model_result_batch_matching(result,q).values()))
    res = np.array(res).T.tolist()
    return res


def filter_line(x, y):
    new_x  = []
    new_y = []
    for i in range(len(y)):
        if y[i]>=0:
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y


def get_draw_data(y, name, mode):
    location = "lower right"
    model_list = ['Real data', 'Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model','M/M/1/k Queuing Model','M/M/N Queuing Model',  'Batch Matching']
    if name == 'Matching rate':
        if mode == 'fix_driver':
            location = "upper left"
        model_list = ['Real data', 'Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 and M/M/N Queuing Model','M/M/1/k Queuing Model',  'Batch Matching']
        
        y = pd.DataFrame(y)[[0, 1, 2, 3, 4, 5, 7]].values.tolist()
    if name == 'Pick-up time':
        model_list = ['Real data','Cobb-Douglas and M/M/1/k','FCFS','M/M/1, M/M/N Queuing Model',  'Batch Matching']
        # df = pd.DataFrame(y)
        y = pd.DataFrame(y)[[0, 1, 2, 4, 7]].values.tolist()
        if mode == 'fix_driver':
            location = "upper left"
        else:
            location = "upper right"
    if name == 'Matching time' or name == 'Waiting time':
        model_list = ['Real data', 'FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model','M/M/1/k Queuing Model','M/M/N Queuing Model',  'Batch Matching']
        y = pd.DataFrame(y)[[0, 2, 3, 4, 5, 6, 7]].values.tolist()
        if mode == 'fix_driver':
            location = "upper left"
        else:
            location = "upper right"
    return y, model_list, location


def draw_picture(x, y, name, type, mode, num):
    fs = 20
    y, model_list, location = get_draw_data(y, name, mode)
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    if mode == 'fix_driver':
        ax.axvspan(0,num/633*0.7 , facecolor="#016795", alpha=0.08)
        ax.axvspan(num/633*0.7, num/633, facecolor="#808080", alpha=0.08)
        x_label = 'Arrival rate of orders (pax/sec)'
        x_label = 'Arrival rate of orders (pax/sec)'
    elif mode == 'fix_order':
        ax.axvspan(num*7.275/5*633, num*7.275/5*633/0.7, facecolor="#808080", alpha=0.08)
        ax.axvspan(num*7.275/5*633/0.7, 2000, facecolor="#016795", alpha=0.08)
        x_label = 'Number of drivers (veh)'
    # plt.show()
    # exit()
    for i in range(len(y[0])):
        current_line = np.array(y)[:, i:i + 1].T[0]
        if i == 0:
            m = '*'
            m_size = 10
            l_width = 2
        else:
            m = 'o'
            m_size = 5
            l_width = 1
        new_x, new_y = filter_line(x, current_line)
        l = plt.plot(new_x, new_y, label=model_list[i], marker=m, linewidth=l_width, markersize=m_size)
    legend1 = plt.legend(loc=location, title="Models", fontsize=fs)

    plt.setp(legend1.get_title(), fontsize=fs)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    plt.xlabel(x_label, fontsize=28)
    if name == 'Matching rate':
        y_label_name = 'Matching rate (pax/sec)'
    else:
        y_label_name = name + ' (sec)'

    plt.ylabel(y_label_name, fontsize=28)
    picture_name = name + '_' + mode + '=' + str(num)

    if not os.path.exists(load_path + 'Figures/' + type):
        os.mkdir(load_path + 'Figures/' + type)
    if not os.path.exists(load_path + 'Figures/' + type + '/' + mode):
        os.mkdir(load_path + 'Figures/' + type + '/' + mode)
    plt.savefig(load_path + 'Figures/' + type + '/' + mode + '/' + picture_name + '.jpg', dpi=600, bbox_inches='tight')
    plt.cla()
    plt.close("all")


def draw_one_picture(type, mode, num, trip_time, files_list, result_path):
    file_list = get_file_list(num, mode, files_list)
    x = []
    matching_rate = []
    matching_time = []
    pickup_time = []
    waiting_time = []
    if type == 'true_data':
        if mode == 'fix_driver':
            para_x = 1
        elif mode == 'fix_order':
            para_x = 3
        for file in file_list:
            result = pickle.load(open(result_path + file, 'rb'))

            if float(file.split('_')[1]) * 7.275 / 5 > result['fleet_size'] / trip_time:
                pass
            else:
                if para_x == 1:
                    x.append(float(file.split('_')[para_x])*7.275)
                else:
                    x.append(float(file.split('_')[para_x]))
                res_data = get_data(result, float(file.split('_')[1])* 7.275 / 5)
                matching_rate.append(res_data[0])
                matching_time.append(res_data[1])
                pickup_time.append(res_data[2])
                waiting_time.append(res_data[3])
    else:
        if mode == 'fix_driver':
            para_x = 1
        elif mode == 'fix_order':
            para_x = 3
        for file in file_list:
            result = pickle.load(open(result_path + file, 'rb'))

            if float(file.split('_')[1])/5 > result['fleet_size']/trip_time:
                pass
            else:
                x.append(float(file.split('_')[para_x]))
                res_data = get_data(result)
                matching_rate.append(res_data[0])
                matching_time.append(res_data[1])
                pickup_time.append(res_data[2])
                waiting_time.append(res_data[3])
    if mode == 'fix_driver':
        x=np.array(x)/5
    if np.array(x).size > 0:
        draw_picture(x, matching_rate, 'Matching rate', type, mode, num)
        draw_picture(x, matching_time, 'Matching time', type, mode, num)
        draw_picture(x, pickup_time, 'Pick-up time', type, mode, num)
        draw_picture(x, waiting_time, 'Waiting time', type, mode, num)


def draw_evaluation():
    result_path = './Results/driver_stable/'
    files = os.listdir(result_path)
    files.sort()
    time = []
    files_list = [item for item in files if 'drivers_' in item]
    for file in files_list:
        result = pickle.load(open(result_path + file, 'rb'))
        time.append(result['trip_time'])
    trip_time = int(np.array(time).mean())
    print(trip_time)
    draw_one_picture('evaluation', 'fix_driver', 2000, trip_time, files_list, result_path)

    # for driver_num in tqdm(range(200, 2200, 200), desc='handling'):
    #     draw_one_picture('evaluation', 'fix_driver', driver_num, trip_time, files_list, result_path)
    # #
    # for order_sample in tqdm(range(1, 11, 1), desc='handling'):
    #     draw_one_picture('evaluation', 'fix_order', order_sample*6.2/10, trip_time, files_list, result_path)


def draw_raw_simulator():
    result_path = '../../hongkong_simulator/Result/'
    files = os.listdir(result_path)
    files.sort()
    time = []
    files_list = [item for item in files if 'drivers_' in item]
    for file in files_list:
        result = pickle.load(open(result_path + file, 'rb'))
        time.append(result['trip_time'])
    trip_time = int(np.array(time).mean())
    print(trip_time)
    # draw_one_picture('true_data', 'fix_driver', 800, trip_time, files_list, result_path)

    # for driver_num in tqdm(range(1000, 1200, 200), desc='handling'):
    #     draw_one_picture('true_data', 'fix_driver', driver_num, trip_time, files_list, result_path)
    for order_sample in tqdm(range(4, 5, 1), desc='handling'):
        draw_one_picture('true_data', 'fix_order', order_sample/10, trip_time, files_list, result_path)


def draw_best_model_raw_simulaor():
    result_path = '../../hongkong_simulator/Result/'
    files = os.listdir(result_path)
    time = []
    files_list = [item for item in files if 'drivers_' in item]
    for file in files_list:
        result = pickle.load(open(result_path + file, 'rb'))
        time.append(result['trip_time'])
    trip_time = int(np.array(time).mean())
    print(trip_time)
    orders = []
    drivers = []
    best_model = []
    best_model_mape = []
    for file in tqdm(files_list, desc='loading best model'):
        result = pickle.load(open(result_path + file, 'rb'))
        if float(file.split('_')[1])*7.275/5 > result['fleet_size']/trip_time:
            pass
        else:
            orders.append(float(file.split('_')[1])*7.275)
            drivers.append(float(file.split('_')[3]))
            m, m_e = get_best_model(result, float(file.split('_')[1])*7.275/5)
            best_model.append(list(m))
            best_model_mape.append(list(m_e))

    x=np.array(orders)/5
    y=np.array(drivers)

    labels=np.array(np.array(best_model)[:,:1].T)[0]
    errors = np.array(np.array(best_model_mape)[:,:1].T)[0]
    draw_best_model('true_data', x, y, labels,errors, 'matching_rate_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,1:2].T)[0]
    errors = np.array(np.array(best_model_mape)[:,1:2].T)[0]
    draw_best_model('true_data', x, y, labels, errors,'matching_time_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,2:3].T)[0]
    errors = np.array(np.array(best_model_mape)[:,2:3].T)[0]
    draw_best_model('true_data', x, y, labels, errors, 'pickup_time_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,3:4].T)[0]
    errors = np.array(np.array(best_model_mape)[:,3:4].T)[0]
    draw_best_model('true_data', x, y, labels, errors, 'waiting_time_best_model', trip_time)


def draw_best_model_evaluation():
    result_path = './Results/'
    files = os.listdir(result_path)
    time = []
    files_list = [item for item in files if 'drivers_' in item]
    for file in files_list:
        result = pickle.load(open(result_path + file, 'rb'))
        time.append(result['trip_time'])
    trip_time = int(np.array(time).mean())
    print(trip_time)
    orders = []
    drivers = []
    best_model = []
    best_model_mape = []
    for file in tqdm(files_list, desc='loading best model'):
        result = pickle.load(open(result_path + file, 'rb'))
        if float(file.split('_')[1])/5 > result['fleet_size']/trip_time:
            pass
        else:
            orders.append(float(file.split('_')[1])/5)
            drivers.append(float(file.split('_')[3]))
            m, m_e = get_best_model(result)
            best_model.append(list(m))
            best_model_mape.append(list(m_e))

    x=np.array(orders)
    y=np.array(drivers)

    print('orders', orders)
    print('drivers', drivers)
    print('m', m)
    print('m_e', m_e)
    print('best_model', best_model)
    print('best_model_mape', best_model_mape)
    labels=np.array(np.array(best_model)[:,:1].T)[0]
    errors = np.array(np.array(best_model_mape)[:,:1].T)[0]
    draw_best_model('evaluation', x, y, labels,errors, 'matching_rate_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,1:2].T)[0]
    errors = np.array(np.array(best_model_mape)[:,1:2].T)[0]
    draw_best_model('evaluation', x, y, labels, errors,'matching_time_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,2:3].T)[0]
    errors = np.array(np.array(best_model_mape)[:,2:3].T)[0]
    draw_best_model('evaluation', x, y, labels, errors, 'pickup_time_best_model', trip_time)
    labels=np.array(np.array(best_model)[:,3:4].T)[0]
    errors = np.array(np.array(best_model_mape)[:,3:4].T)[0]
    draw_best_model('evaluation', x, y, labels, errors, 'waiting_time_best_model', trip_time)


if __name__ == "__main__":
    plt.rc('font', family='Times New Roman')
    # production_func_params = get_production_func_params()
    model_list = ['Perfect Matching','FCFS','Cobb-Douglas Production Function','M/M/1 Queuing Model', 'M/M/1/k Queuing Model', 'M/M/N Queuing Model','Batch Matching']

    # draw_evaluation()
    # draw_raw_simulator()
    draw_best_model_raw_simulaor()
    # draw_best_model_evaluation()

