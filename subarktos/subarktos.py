import sys
sys.path.append("..")
from rocket import *
import numpy as np
import pandas as pd
from scipy.integrate import quad
from matplotlib import pyplot as plt

def get_subarktos():
    mass = 4.42 #kg
    motor = Motor(1.223, 0.713, "J450.txt")
    drogue = Parachute(0.3, 1.8, "Drogue")
    main = Parachute(0.9, 2.2, "Main", alt=800)
    altimeter = Sensor(0, 5.54, None)
    accelerometer = Sensor(2, 62.953, None)
    drag = 0.000292
    subarktos = Rocket(mass, [drogue, main], [motor], [altimeter, accelerometer], drag)
    return subarktos

def get_state_data():
    sim = pd.read_csv("subarktos_sim.csv", comment='#')
    read_t = np.array(sim['Time (s)'])
    read_alt = np.array(sim['Altitude (ft)'])
    read_vel = np.array(sim['Vertical velocity (ft/s)'])
    read_acc = np.array(sim['Vertical acceleration (ft/s^2)'])
    return (read_t, read_alt, read_vel, read_acc)

def get_force_data():
    sim = pd.read_csv("subarktos_sim.csv", comment='#')
    read_t = np.array(sim['Time (s)'])
    t_apogee = 16.16 # read off csv
    read_drag = np.array(sim['Drag force (N)'])
    read_thrust = np.array(sim['Thrust (N)'])
    read_gravity = np.array(sim['Gravitational acceleration (ft/s^2)']) * np.array(sim['Mass (oz)']) * 0.008641
    def force(t):
        n = np.where(read_t == t)[0][0]
        if t < t_apogee:
            return read_thrust[n] - read_drag[n] - read_gravity[n]
        else:
            return read_drag[n] - read_gravity[n]
    return (np.vectorize(force))(read_t)

def test_acc_differences(subarktos, read_t, read_acc):
    # question: if you integrate the same acceleration data, do you get the same velocity and altitude?
    # answer: you do!
    def new_ext_input(t):
        if t < np.min(read_t) or t > np.max(read_t):
            return 0
        return (interp.interp1d(read_t, read_acc)(t)/3.28 * subarktos.get_mass(t), "OpenRocket sim data")

    subarktos.ext_input = new_ext_input
    return subarktos

def sim_and_plot(subarktos, read_t, read_alt, read_vel, read_acc, read_force):
    plt.figure()
    times, states, inputs = subarktos.simulate(dt=0.001, timeout=120, verbose=False)

    print(states[1][2300:2400])

    plt.subplot(2,2,1)
    plt.plot(times, 3.28*states[0], label='subarktos.simulate()')
    plt.plot(read_t, read_alt, label='OpenRocket simulation')
    plt.title("Altitude over time from subarktos.simulate() and OpenRocket")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (ft)")

    plt.subplot(2,2,2)
    plt.plot(times, 3.28*states[1], label='subarktos.simulate()')
    plt.plot(read_t, read_vel, label='OpenRocket simulation')
    plt.title("Velocity over time from subarktos.simulate() and OpenRocket")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (ft/s)")

    plt.subplot(2,2,3)
    plt.plot(times, 3.28*states[2], label='subarktos.simulate()')
    plt.plot(read_t, read_acc, label='OpenRocket simulation')
    plt.title("Acceleration over time from subarktos.simulate() and OpenRocket")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (ft/s^2)")

    plt.subplot(2,2,4)
    plt.plot(times, inputs[0], label='subarktos.simulate()')
    plt.plot(read_t, read_force, label='OpenRocket simulation')
    plt.title("Total force over time from subarktos.simulate() and OpenRocket")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.show()

def test_kalman_openrocket(subarktos, read_t, read_alt, read_acc):
    sigma_alt = np.max(read_alt)/64
    sigma_acc = np.max(read_acc)/64
    alt_noise = np.random.normal(0, sigma_alt, np.size(read_alt))
    acc_noise = np.random.normal(0, sigma_acc, np.size(read_acc))
    sensor_data = np.vstack([(read_alt + alt_noise, read_acc + acc_noise)]).T
    filtered_times, filtered_states, _ = subarktos.simulate(dt=0.001, timeout=120, kalman=(read_t, sensor_data))

    def interpolate_state_data(data):
        def toreturn(t):
            if t < np.min(read_t) or t > np.max(read_t):
                return 0
            return interp.interp1d(read_t, data)(t)
        return toreturn

    alt_error_after = 3.28*filtered_states[0] - np.vectorize(interpolate_state_data(read_alt))(filtered_times)
    acc_error_after = 3.28*filtered_states[2] - np.vectorize(interpolate_state_data(read_acc))(filtered_times)

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(read_t, read_alt, label="OpenRocket without noise")
    plt.plot(filtered_times, 3.28*filtered_states[0], label="subarktos.simulate() filtered from OpenRocket + noise")
    plt.legend(loc='upper right')
    plt.title("Kalman filtered rocket altitude against true values.")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (ft)")

    plt.subplot(2,2,2)
    plt.plot(read_t, read_acc, label="OpenRocket without noise")
    plt.plot(filtered_times, 3.28*filtered_states[2], label="subarktos.simulate() filtered from OpenRocket + noise")
    plt.legend(loc='upper right')
    plt.title("Kalman filtered rocket acceleration against true values.")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (ft/s^2)")

    plt.subplot(2,2,3)
    plt.plot(read_t, alt_noise, label="Error before filtering")
    plt.plot(filtered_times, alt_error_after, label="Error after filtering")
    plt.title("Altitude error reduction due to Kalman filtering.")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Error in altitude (ft)")

    plt.subplot(2,2,4)
    plt.plot(read_t, acc_noise, label="Error before filtering")
    plt.plot(filtered_times, acc_error_after, label="Error after filtering")
    plt.title("Acceleration error reduction due to Kalman filtering.")
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.ylabel("Error in acceleration (ft/s^2)")

    plt.show()

subarktos = get_subarktos()
state_data = get_state_data()
force_data = get_force_data()
sim_and_plot(subarktos, *state_data, force_data)
