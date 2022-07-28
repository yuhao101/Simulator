from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [int(2000*i/10) for i in range(1, 11)]
    order_sample_frac = [i/10 for i in range(1, 11)]
    max_distance_num = [1]
    # max_distance_num = [i/2 for i in range(1, 11, 1)]
    # time_interval = [i for i in range(5, 55, 5)]
    time_interval = [5]
    # driver_num = [1000]
    # order_sample_frac = [1]
    print('max_distance_num', max_distance_num)
    print('time_interval', time_interval)
    print('driver_num', driver_num)
    print('order_sample_frac', order_sample_frac)
    cruise_flag = [True]
    pickup_flag = ['ma']
    delivery_flag = ['rg']
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:
                for single_driver_num in driver_num:
                    for single_order_sample_frac in order_sample_frac:
                        for single_max_distance_num in max_distance_num:
                            for single_time_interval in time_interval:
                                env_params['pickup_mode'] = pc_flag
                                env_params['delivery_mode'] = dl_flag
                                env_params['cruise_flag'] = cr_flag
                                env_params['driver_num'] = single_driver_num
                                env_params['maximal_pickup_distance'] = single_max_distance_num
                                env_params['delta_t'] = single_time_interval
                                env_params['request_interval'] = single_time_interval
                                env_params['order_sample_ratio'] = single_order_sample_frac
                                simulator = Simulator(**env_params)
                                simulator.reset()
                                track_record = []
                                t = time.time()
                                for step in tqdm(range(simulator.finish_run_step)):
                                    new_tracks = simulator.step()
                                    track_record.append(new_tracks)

                                match_and_cancel_track_list = simulator.match_and_cancel_track
                                file_path = './final_experiment/' + pc_flag + "_" + dl_flag + "_" + "cruise="+str(cr_flag)
                                if not os.path.exists(file_path):
                                    os.makedirs(file_path)
                                file_path += '/driver_num_' + str(single_driver_num)
                                if not os.path.exists(file_path):
                                    os.makedirs(file_path)
                                file_path += '/sample_frac_' + str(single_order_sample_frac)
                                if not os.path.exists(file_path):
                                    os.makedirs(file_path)
                                if not os.path.exists(file_path+'/records'):
                                    os.makedirs(file_path+'/records')
                                if not os.path.exists(file_path + '/passengers'):
                                    os.makedirs(file_path+'/passengers')
                                if not os.path.exists(file_path + '/match_and_cancel'):
                                    os.makedirs(file_path+'/match_and_cancel')
                                pickle.dump(track_record, open(file_path+'/records' + '/records_max_distance_'+
                                                               str(single_max_distance_num)+ '_time_interval_' +
                                                               str(single_time_interval) + '.pickle', 'wb'))
                                pickle.dump(simulator.requests, open(file_path+'/passengers' + '/passenger_records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))

                                pickle.dump(match_and_cancel_track_list,open(file_path+'/match_and_cancel'+'/match_and_cacel_'+str(single_driver_num)+'.pickle','wb'))
                                file = open(file_path + '/time_statistic.txt', 'a')
                                file.write(str(time.time()-t)+'\n')




