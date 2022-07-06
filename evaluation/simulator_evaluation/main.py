from evaluation.simulator_evaluation.simulator_env import Simulator
from evaluation.simulator_evaluation.config import *
from evaluation.simulator_evaluation.utilities import save_data
from evaluation.simulator_evaluation.Create_Drivers import create_driver
from evaluation.simulator_evaluation.Create_Records import create_records

if __name__ == "__main__":
    driver_num = [int(2000 * i / 10) for i in range(1, 11)]
    order_sample_frac = [i / 10 for i in range(1, 11)]
    # max_distance_num = [i/2 for i in range(1, 11, 1)]
    max_distance_num = [2]
    # time_interval = [i for i in range(5, 55, 5)]
    time_interval = [5]
    print('max_distance_num', max_distance_num)
    print('time_interval', time_interval)
    print('driver_num', driver_num)
    print('order_sample_frac', order_sample_frac)
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for single_driver_num in driver_num:
        for single_order_sample_frac in order_sample_frac:
            for single_max_distance_num in max_distance_num:
                for single_time_interval in time_interval:
                    env_params['ave_order'] = single_order_sample_frac
                    env_params['num_drivers'] = single_driver_num
                    env_params['time_period'] = int((env_params['t_end']) / env_params['delta_t'])
                    env_params['pickup_dis_threshold'] = single_max_distance_num
                    env_params['delta_t'] = single_time_interval
                    env_params['request_interval'] = single_time_interval
                    create_driver()
                    create_records()

                    simulator = Simulator(**env_params)
                    simulator.reset()

                    for k in range(simulator.finish_run_step):
                        simulator.step()
                        if k % 500 == 0:
                            print(simulator.current_step)
                    save_data(simulator, env_params)
