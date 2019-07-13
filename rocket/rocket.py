import numpy as np
import scipy.interpolate as interp

class Parachute:

    CHUTE_OPEN_DELAY = 0.2
    '''
    diameter : scalar
        Diameter of the parachute.
    cd : scalar
        Drag coefficient of the parachute.
    alt : scalar
        Deployment altitude of the parachute in feet converted to meters.
        If None, considered to be at apogee.
    deployed : bool
        Indicates whether or not the chute has deployed.
    name : string
        Name of the parachute, for reading status.
    t_deployed : scalar
        Time when the parachute was deployed.
    '''
    def __init__(self, diameter, cd, name, alt=None):
        self.b_drag = 1.275 * np.pi * cd * diameter**2/8
        if alt is not None:
            self.alt = alt * 0.3048
        else:
            self.alt = float('inf')
        self.deployed = False
        self.name = name
        self.t_deployed = None

    def drag(self, velocity, t):
        # velocity in m/s
        if not self.deployed or t - self.t_deployed < self.CHUTE_OPEN_DELAY:
            return 0
        return self.b_drag * velocity**2

    def deploy(self, t):
        if not self.deployed:
            print("Deploying", self.name, "at t =", str(t).strip(),"s")
            self.deployed = True
            self.t_deployed = t

class Motor:
    # credit for thrust and mass functions: Aled Cuda
    '''
    mass_init : scalar
        Initial mass of the motor.
    mass_final : scalar
        Final mass of the motor.
    time_burnout : scalar
        Time at which motor burns out in seconds.
    thrust_curve : string
        Name of text file containing thrust curve data.
    time_delay : scalar
        Time between launch and ignition of this motor in seconds.
    '''
    def __init__(self, mass_init, mass_final, thrust_curve, time_delay=0):
        self.mass_init = mass_init
        self.mass_final = mass_final
        self.thrust_data = np.loadtxt(thrust_curve)
        # adding thrust(t = 0) = 0 to help interpolation
        self.thrust_data = np.concatenate((np.array([[0,self.thrust_data[::,1][1]/2]]), self.thrust_data), axis=0)
        self.max_thrust = np.amax(self.thrust_data,0)[1]
        assert time_delay >= 0, "Cannot have a negative delay."
        self.time_delay = time_delay
        # adjust for time delay in thrust data
        self.thrust_data = self.thrust_data + np.vstack([np.array([time_delay,0])]*self.thrust_data.shape[0])
        self.time_burnout = np.max(self.thrust_data[::,0]) + time_delay

    def thrust(self, t):
        # If we ask for a time before or after the range we assume thrust is zero
        if t > self.time_burnout or t < self.time_delay:
            return 0
        # Otherwise we use the interpolate function
        return interp.interp1d(self.thrust_data[::,0], self.thrust_data[::,1])(t)

    def mass(self, t):
        if t < self.time_delay:
            return self.mass_init
        elif t > self.time_burnout:
            return self.mass_final
        return ((self.mass_final - self.mass_init)/(self.time_burnout - self.time_delay))*t + self.mass_init

class Sensor:
    '''
    Object that reads in a rocket state variable.
    select : int
        Indicates which state variable is being read: 0 for altitude, 1 for velocity (not expected to be used),
        2 for acceleration.
    var : scalar
        Variance of the sensor.
    data : ndarray
        Collection of n row vectors with two elements: time and sensor reading.
    convert : scalar
        Multiplicative factor from the state variable to the sensor reading. Mostly m to ft conversion.
    '''

    def __init__(self, select, var, data, convert=3.28):
        self.select = select
        self.var = var
        self.data = data
        self.convert = convert

class Rocket:

    '''
    A rocket with an altimeter and accelerometer.

    dry_mass : scalar
        Takes in time, returns rocket mass.
    parachutes : list
        A list of Parachute objects.
    motors : list
        A list of Motor objects.
    sensors : list
        A list of Sensor objects. Setting this to the empty list will set Kalman filtering to simulation without updates.
    b_drag : scalar
        Coefficient on v^2 in drag equation. (Not the same as cd.)
        Currently being determined via curve fitting on OpenRocket sim data.
    state : ndarray
        1x3 row vector containing position, velocity, and acceleration in the best units.
        State represented as a row vector, to be transposed if it's important for matrix operations.
    apogee : bool
        Boolean to indicate if apogee has been reached yet, to update parachute states.
    '''
    STATE_SIZE = 3
    INPUT_SIZE = 1

    def __init__(self, dry_mass, parachutes, motors, sensors, b_drag):
        self.dry_mass = dry_mass
        self.parachutes = parachutes
        self.motors = motors
        self.sensors = sensors
        self.b_drag = b_drag
        self.H = np.zeros((len(sensors), 3))
        self.R = np.zeros((len(sensors), len(sensors)))
        for i,s in enumerate(sensors):
            self.H[i][s.select] = s.convert
            self.R[i][i] = s.var
        self.P = np.zeros([self.STATE_SIZE, self.STATE_SIZE])
        q = 2
        d = 0.001 # sampling time difference of one of the sensors. Just in here for the noise model, to be updated.
        # constant acceleration approximation: change this later
        self.Q = q * np.array([[d**4/4, d**3/3, d**2/2], [d**3/3, d**2/2, d], [d**2/2, d, 1]])
        # for now these are hardcoded in, but find a way to remove them and still detect apogee:
        self.stdev_alt = np.sqrt(5.54)
        self.stdev_acc = np.sqrt(62.953)
        self.apogee = False
        self.simend = False
        self.max_thrust = sum([m.max_thrust for m in self.motors])
        self.state = np.array([0, 0, 0])
        self.state[2] = self.get_thrust(0)/self.get_mass(0)

    def get_mass(self, t):
        return self.dry_mass + sum([m.mass(t) for m in self.motors])

    def get_thrust(self, t):
        # to do: burnout detection.
        return sum([m.thrust(t) for m in self.motors])

    def get_rocket_drag(self):
        magnitude = self.b_drag * self.state[1]**2
        if not self.apogee:
            return -magnitude
        else:
            return magnitude

    def get_chute_drag(self, t):
        return sum([p.drag(self.state[1], t) for p in self.parachutes])

    def ext_input(self, t, mass=None):
        if mass is None:
            mass = self.get_mass(t)
        thrust = self.get_thrust(t)
        if thrust == 0 and t < 0.2:
            # to do: better way of detecting 'still on the pad'
            gravity = 0
        else:
            gravity = -9.8 * mass
        rocket_drag = self.get_rocket_drag()
        if thrust > 0.05*self.max_thrust:
            assert thrust > np.abs(rocket_drag), "Excessive drag"
        chute_drag = self.get_chute_drag(t)
        return (thrust + gravity + rocket_drag + chute_drag, str([thrust, gravity, rocket_drag, chute_drag]) + '\n')

    def evolve(self, t, dt, order=2):
        # order: of integration.
        m = self.get_mass(t)
        if order == 1:
            A = np.array([[1, dt, dt**2/2],[0, 1, dt],[0, 0, 0]])
            B = np.array([0, 0, 1/m])
        elif order == 2:
            A = np.array([[1, dt, dt**2/2], [0, 1, dt/2], [0, 0, 0]])
            B = np.array([0, dt/(2*m), 1/m])
        # Orders > 2 to be implemented based on https://math.stackexchange.com/questions/2946737/solving-a-matrix-differential-equation-using-runge-kutta-methods
        # plus some slightly jank workarounds that allow the 'a' row to still be all zero.
        # Check for apogee
        if not self.apogee:
            try:
                assert self.state[0] > -self.stdev_alt, "you will not go to space today"
            except AssertionError as e:
                e.args += ("Time", t, "State", self.state)
                raise
            if self.state[1] < -self.stdev_acc and self.state[2] < -self.stdev_acc:
                # this is kind of a jank apogee check, especially because I'm comparing velocity to a deviation in acceleration
                try:
                    assert self.get_thrust(t) == 0, "If you're at apogee, the motor can't still be burning"
                except AssertionError as e:
                    e.args += ("Time", t, "thrust", self.get_thrust(t), "state:", self.state)
                    raise
                self.apogee = True
                print("Hit apogee at altitude",str(np.round(3.28*self.state[0], 2)),"ft")
                for p in self.parachutes:
                    if p.alt is None:
                        p.deploy(t)
        # Check for altitude-based chute deployment
        # To do: nicer way of doing this than checking at every timestep.
        if self.apogee and not all([p.deployed for p in self.parachutes]):
            for p in self.parachutes:
                if p.alt > self.state[0]:
                    p.deploy(t)

        # Check for ground hit
        if self.apogee and self.state[0] < self.stdev_alt and A.dot(self.state)[0] < -2*self.stdev_alt:
            self.simend = True
        return (A, B)

    def predict(self, t, dt):
        # predicts system state at time t+dt based on system state at time t
        A, B = self.evolve(t, dt)
        u, u_status = self.ext_input(t)
        state_predicted = A.dot(self.state) + B.dot(u)
        P_predicted = A.dot(self.P.dot(A.T)) + self.Q
        return (u, u_status, state_predicted, P_predicted)

    def measure(self, state):
        return self.H.dot(state)

    def update(self, state_predicted, P_predicted, measurement):
        error = measurement - self.measure(state_predicted)
        K = P_predicted.dot(self.H.T.dot(np.linalg.inv(self.H.dot(P_predicted.dot(self.H.T)) + self.R)))
        state_updated = state_predicted + K.dot(error)
        P_updated = P_predicted - K.dot(self.H.dot(P_predicted))
        return (state_updated, P_updated)

    def sim_results(self, dt, k, states, inputs, terminate):
        self.reset()
        print("Simulation ended at t =", np.round(dt*k, -int(np.floor(np.log10(dt)))), "s due to", terminate)
        processed_states = np.zeros([self.STATE_SIZE, k])
        processed_inputs = np.zeros([self.INPUT_SIZE, k])
        states = states.T
        inputs = inputs.T
        for i, state in enumerate(states):
            processed_states[i] = state[:k]
        for i, input in enumerate(inputs):
            processed_inputs[i] = input[:k]
        return (np.linspace(0,dt*k,k+1)[:k], processed_states, processed_inputs)

    def simulate(self, dt=0.01, timeout=30, verbose=False, kalman=None):
        k = 0
        if kalman is not None:
            m = 0
            measure_times = kalman[0]
        num_timesteps = int(np.ceil(timeout/dt))+1
        states = np.zeros([num_timesteps, self.STATE_SIZE])
        inputs = np.zeros([num_timesteps, self.INPUT_SIZE])
        terminate = "error."
        try:
            while k < num_timesteps:
                if verbose and hasattr(self, "compact_status") and k % 100 == 0:
                    self.compact_status(dt*k)
                states[k] = self.state
                inputs[k], input_status, state_predicted, P_predicted = self.predict(dt*k, dt)
                if kalman is not None and m < measure_times.size and np.isclose(dt*k, measure_times[m], atol=dt/2):
                    self.state, self.P = self.update(state_predicted, P_predicted, kalman[1][m])
                    m += 1
                else:
                    self.state = state_predicted
                    self.P = P_predicted
                if verbose and input_status is not None and k % 100 == 0:
                    print(input_status)
                if self.simend:
                    terminate = "landing."
                    break
                k += 1
        except KeyboardInterrupt:
            print("\nSteps completed:", k)
            terminate = "interrupt."
        if k == num_timesteps:
            terminate = "timeout."
        return self.sim_results(dt, k, states, inputs, terminate)

    def reset(self):
        # for some reason copy.deepcopy isn't working so here
        for p in self.parachutes:
            p.deployed = False
            p.t_deployed = None
        self.apogee = False
        self.state = np.array([0,0,0])
        self.simend = False

    def status(self):
        print("Altitude:", self.state[0])
        print("Velocity:", self.state[1])
        print("Acceleration:", self.state[2])
        print("Apogee hit:", self.apogee)
        for p in self.parachutes:
            print(p.name, "deployed:", p.deployed)
        print("Ground hit:", self.simend)

    def compact_status(self, t):
        print(t, self.state, self.apogee, [p.deployed for p in self.parachutes], self.simend)
