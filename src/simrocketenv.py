import gym
from gym import spaces
import numpy as np
import random
from geodetic_toolbox import *
# pybullet:
import pybullet as p
import pybullet_data

class SimRocketEnv(gym.Env):
    def __init__(self, interactive=False):
        # vehicle config describes the vehicle specific properties:
        self.pybullet_initialized = False
        self.interactive = interactive

        self.reset_count = 0 # keep track of calls to reset() function
        self.time_sec = 0.0 # keep track of simulation time
        self.dt_sec = 1.0 / 60.0 # update rate of the simulation
        self.urdf_file = "./src/modelrocket.urdf"
        self.UMIN = -1.0
        self.UMAX =  1.0
        self.ACTUATORCOUNT = 5
        self.THRUST_UMIN = 0.0
        self.THRUST_MAX_N = 1500.0
        self.THRUST_TAU = 2.5
        self.THRUST_VECTOR_TAU = 0.3
        self.THRUST_MAX_ANGLE = np.deg2rad(10.0)
        self.ATT_MAX_THRUST = 50.0
        self.GRAVITY = 9.81
        self.mass_kg = -99999999.9 # will be loaded from URDF
        self.MIN_GROUND_DIST_M = 2.45 # shut off engine below this altitude

        self.pybullet_setup_environment()
        # initialize state of the vehicle
        # <state>
        state = self.reset() # reset will add and reset additional basic state variables
        # </state>

        # Action space is set to actuator umin/umax limits
        self.action_space = spaces.Box(low=self.UMIN, high=self.UMAX, shape=(self.ACTUATORCOUNT,))
        self.action_space.low[0] = self.THRUST_UMIN
        obs_hi = np.ones(state.shape[0]) * 2000.0
        self.observation_space = spaces.Box(low=-obs_hi, high=obs_hi, dtype=np.float32)

    def pybullet_setup_environment(self):
        # pybullet world frame is ENU EAST (X) NORTH (Y) UP (Z)
        # pybullet body frame is FORWARD (X) LEFT (Y) UP (Z)
        assert(self.pybullet_initialized == False)

        self.PYBULLET_DT_SEC = 1.0/240.0

        if self.interactive:
            print("GUI mode")
            self.CLIENT = p.connect(p.GUI)
        else:
            self.CLIENT = p.connect(p.DIRECT)

        self.pybullet_initialized == True

    def reset(self):
        """
        Gym interface. Reset the simulation.
        :return state (state vector)
        """

        self.engine_on = True
        # <state>
        self.pos_n   = np.array([np.random.uniform(-50.0, 50.0), np.random.uniform(-50.0, 50.0), np.random.uniform(30.0, 60.0)]) # ENU
        self.vel_n   = np.array([np.random.uniform(-8.0, 8.0), np.random.uniform(-8.0, 8.0), np.random.uniform(-15.0, 5.0)]) # ENU

        # Maintain the attitude as quaternion and Euler angles. The source of truth is
        # the quaternion (self.q) and roll_deg, pitch_deg and yaw_deg will be updated
        # based on the quaternion. But here for initialization the Euler angles are
        # used to initialize the orientation (Euler angles are a bit more readable)
        self.roll_deg  = np.random.uniform(-10.0, 10.0) # Random initialization with e.g. np.random.uniform(-10, 10)
        self.pitch_deg = np.random.uniform(-10.0, 10.0)
        self.yaw_deg   = 0.0
        # Careful: this quaternion is in the order: qw, qx,qy,qz (qw is the real part)
        self.q         = quat_from_rpy(np.deg2rad(self.roll_deg), np.deg2rad(self.pitch_deg), np.deg2rad(self.yaw_deg))

        roll_rate_rps  = np.deg2rad(np.random.uniform(-10.0, 10.0))
        pitch_rate_rps = np.deg2rad(np.random.uniform(-10.0, 10.0))
        yaw_rate_rps   = 0.0
        self.omega     = np.array([roll_rate_rps, pitch_rate_rps, yaw_rate_rps]) # vehicle rotation rates
        self.omega_dot = np.array([0.0, 0.0, 0.0])

        self.thrust_current_N = 0.7 * self.THRUST_MAX_N
        self.thrust_alpha = 0.0
        self.thrust_beta = 0.0
        # </state>
        self.update_state() # create/update state vector

        # <simulation>
        self.time_sec = 0.0
        self.reset_count += 1
        self.epochs = 0
        # </simulation>

        self.pybullet_reset_environment()

        return self.state

    def pybullet_reset_environment(self):
        self.pybullet_time_sec = self.time_sec
        p.resetSimulation(physicsClientId=self.CLIENT) # remove all objects and reset

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        p.setGravity( 0.0, 0.0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYBULLET_DT_SEC, physicsClientId=self.CLIENT)
        plane = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        p.changeDynamics(plane, -1, lateralFriction=1, restitution=0.5, physicsClientId=self.CLIENT)

        initial_position_enu = [self.pos_n[0], self.pos_n[1], self.pos_n[2]]
        self.pybullet_body = p.loadURDF(self.urdf_file, initial_position_enu, physicsClientId=self.CLIENT)
        self.pybullet_booster_index = -1

        self.pybullet_leg_1_joint = -1
        self.pybullet_leg_2_joint = -1
        self.pybullet_leg_3_joint = -1
        # Index of booster
        for i in range(p.getNumJoints(self.pybullet_body, physicsClientId=self.CLIENT)):
            joint_name = p.getJointInfo(self.pybullet_body, i, physicsClientId=self.CLIENT)[1].decode('UTF-8')
            if joint_name == "leg_1_joint":
                self.pybullet_leg_1_joint = i
            if joint_name == "leg_2_joint":
                self.pybullet_leg_2_joint = i
            if joint_name == "leg_3_joint":
                self.pybullet_leg_3_joint = i

        q_rosbody_to_enu = self.q
        # pybullet needs the scalar part at the end of the quaternion
        qxyzw_rosbody_to_enu = [ q_rosbody_to_enu[1], q_rosbody_to_enu[2], q_rosbody_to_enu[3], q_rosbody_to_enu[0] ]
        p.resetBasePositionAndOrientation(self.pybullet_body, initial_position_enu, qxyzw_rosbody_to_enu, physicsClientId=self.CLIENT)
        self.mass_kg = self.get_total_mass(self.pybullet_body)

        #debug lines
        self.debug_line_thrust = -1

        if self.interactive:
            print(f'\033[33mpybullet physics active.\033[0m')
            # print("Mass: %.1f kg" % self.mass_kg)


    def set_camera_follow_object(self, objectId, distance=4.5, pitch=-55, yaw=50):
        pos, orn = p.getBasePositionAndOrientation(objectId, physicsClientId=self.CLIENT)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos, physicsClientId=self.CLIENT
        )

    def pybullet_physics(self, u):

        self.set_camera_follow_object(self.pybullet_body)
        NOZZLE_OFFSET = 2.0 # OFFSET between CoG and nozzle. Dirty hack: should not of course not be hardcoded here

        pybullet_dt_sec = 0.0
        while self.pybullet_time_sec < self.time_sec:
            # simulate thrust dynamics (i.e. the thrust takes some time to react)
            self.thrust_current_N += (self.THRUST_MAX_N     * u[0] - self.thrust_current_N) * self.PYBULLET_DT_SEC / self.THRUST_TAU
            self.thrust_alpha     += (self.THRUST_MAX_ANGLE * u[1] - self.thrust_alpha) * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU
            self.thrust_beta      += (self.THRUST_MAX_ANGLE * u[2] - self.thrust_beta)  * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU

            thrust = np.array([self.thrust_alpha, self.thrust_beta, 1.0]) * self.thrust_current_N

            if self.engine_on == False:
                thrust *= 0.0

            # Add force of rocket boost to pybullet simulation
            p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                 linkIndex=self.pybullet_booster_index,
                                 forceObj=[thrust[0], thrust[1], thrust[2]], # [x forward, y left, z up]
                                 posObj=[0, 0, -NOZZLE_OFFSET],
                                 flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

            att_x_thrust = u[3] * self.ATT_MAX_THRUST
            att_y_thrust = u[4] * self.ATT_MAX_THRUST
            p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                 linkIndex=self.pybullet_booster_index,
                                 forceObj=[att_x_thrust, att_y_thrust, 0.0], # [x forward, y left, z up]
                                 posObj=[0, 0, 2.0], # offset from CoM to top
                                 flags=p.LINK_FRAME, physicsClientId=self.CLIENT)

            p.stepSimulation(physicsClientId=self.CLIENT)
            self.pybullet_time_sec += self.PYBULLET_DT_SEC
            pybullet_dt_sec += self.PYBULLET_DT_SEC

        thrust_vec_line = -np.array([self.thrust_alpha, self.thrust_beta, 1.0]) * 6.0 * self.thrust_current_N / self.THRUST_MAX_N
        thrust_start_point = [0,0,-NOZZLE_OFFSET]
        thrust_end_point = [thrust_vec_line[0],thrust_vec_line[1],thrust_vec_line[2]-2.0]
        thrust_color = [1.0, 0, 0]
        thrust_line_width = 6.0

        if self.interactive:
            if self.debug_line_thrust == -1:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point, thrust_end_point, lineColorRGB=thrust_color,
                                                            parentObjectUniqueId=self.pybullet_body,
                                                            parentLinkIndex=self.pybullet_booster_index, lineWidth=thrust_line_width)
            else:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point, thrust_end_point, lineColorRGB=thrust_color,
                                                            parentObjectUniqueId=self.pybullet_body,
                                                            parentLinkIndex=self.pybullet_booster_index,
                                                            replaceItemUniqueId=self.debug_line_thrust, lineWidth=thrust_line_width)


        # <EXTRACT CURRENT STATE FROM PYBULLET>
        position, orientation = p.getBasePositionAndOrientation(self.pybullet_body, physicsClientId=self.CLIENT)
        linear_velocity, omega_enu = p.getBaseVelocity(self.pybullet_body, physicsClientId=self.CLIENT)
        self.pos_n = np.array([position[0], position[1], position[2]])
        vel_n_prev = self.vel_n
        self.vel_n = np.array([linear_velocity[0], linear_velocity[1], linear_velocity[2]])

        # the pybullet quaternion has the scalar part (qw) of the quaternion at the end (qx,qy,qz,qw order)
        q_rosbody_to_enu = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
        self.q = q_rosbody_to_enu

        # transform the body rotation rates that are given in the ENU world
        # frame to the PyCopterCraft body frame.
        R_enu_to_rosbody = quat_to_matrix(quat_invert(q_rosbody_to_enu))
        omega_rosbody = R_enu_to_rosbody @ np.array([omega_enu[0], omega_enu[1], omega_enu[2]])
        omega_prev = self.omega # save previous rotation rate
        self.omega = omega_rosbody

        if pybullet_dt_sec <= 1e-5:
            self.omega_dot = np.array([0.0,0.0,0.0])
        else:
            self.omega_dot = (self.omega - omega_prev) / pybullet_dt_sec
        # </EXTRACT CURRENT STATE FROM PYBULLET>

    def update_state(self):
        """
        Internal helper function to update self.state vector based on attributes such as self.q, self.pos, etc.
        """
        euler          = quat_to_rpy(self.q)
        self.roll_deg  = np.rad2deg(euler[0])
        self.pitch_deg = np.rad2deg(euler[1])
        self.yaw_deg   = np.rad2deg(euler[2])

        # Produce state vector:
        self.state = np.zeros((16,))
        state_index = 0

        self.state_cfg = {}

        self.state[state_index:(state_index+4)] = self.q
        self.state_cfg["q_index"] = state_index
        state_index += 4

        self.state[state_index:(state_index+3)] = self.omega
        self.state_cfg["omega_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.pos_n
        self.state_cfg["pos_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.vel_n
        self.state_cfg["vel_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+1)] = self.thrust_current_N
        self.state_cfg["thrust_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_alpha
        self.state_cfg["thrust_alpha_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_beta
        self.state_cfg["thrust_beta_index"] = state_index
        state_index += 1

    def step(self, action):
        """
        Gym interface step function to simulate the system.
        :param action Control input to the simulation, i.e. motor/rotor setpoints between 0 and 1 (actually umin and umax to be precise)
        :return state (state vector), reward (score), done (simulation done?)
        """

        action = np.clip(action, self.UMIN, self.UMAX)
        action[0] = np.clip(action[0], self.THRUST_UMIN, self.UMAX) # thrust has a different limit

        done = False
        try:
            self.pybullet_physics(action)
        except Exception as e:
            print("pybullet exception! ", e)
            done = True

        self.time_sec = self.time_sec + self.dt_sec
        self.epochs += 1
        self.update_state()

        reward = self.calculate_reward()
        if self.engine_on == False:
            done = True

        # Stop the non-interactive simulation if the attitude is way off
        if self.interactive == False:
            if np.abs(self.pitch_deg) > 90.0 or np.abs(self.roll_deg) > 90.0:
                reward -= 100.0
                done = True

        return self.state, reward, done, {}

    def print_state(self):

        print("ENU=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) Thrust=%6.1f N alpha=%.1f beta=%.1f" %
              ( self.pos_n[0], self.pos_n[1], self.pos_n[2],
                self.vel_n[0], self.vel_n[1], self.vel_n[2],
                self.roll_deg, self.pitch_deg, self.yaw_deg,
                np.rad2deg(self.omega[0]), np.rad2deg(self.omega[1]), np.rad2deg(self.omega[2]), self.thrust_current_N, np.rad2deg(self.thrust_alpha), np.rad2deg(self.thrust_beta)), end=" ")
        print("")

    def render(self):
        """
        Gym interface. Render current simulation status.
        """
        if self.interactive:
            self.print_state()

    def calculate_reward(self):
            # Constants for reward calculation - these may need tuning
            POSITION_WEIGHT = 1.0
            VELOCITY_WEIGHT = 1.0
            ORIENTATION_WEIGHT = 1.0
            MAX_POS_REWARD = 50   # Maximum reward for position
            MAX_VEL_REWARD = 50   # Maximum reward for velocity
            MAX_ORI_REWARD = 2    # Maximum reward for orientation (cos(0) + cos(0))

            # Calculate the negative distance from the target position (0,0,0)
            target_pos = np.array([0, 0, 0])
            distance = np.linalg.norm(self.pos_n - target_pos)
            distance_reward = MAX_POS_REWARD - POSITION_WEIGHT * distance

            # Calculate the negative velocity magnitude
            velocity_magnitude = np.linalg.norm(self.vel_n)
            velocity_reward = MAX_VEL_REWARD - VELOCITY_WEIGHT * velocity_magnitude

            # Calculate orientation reward
            # Converting degrees to radians for cosine calculation
            roll_rad = np.radians(self.roll_deg)
            pitch_rad = np.radians(self.pitch_deg)
            orientation_reward = ORIENTATION_WEIGHT * (np.cos(roll_rad) + np.cos(pitch_rad))

            # Normalize rewards
            normalized_distance_reward = distance_reward / MAX_POS_REWARD
            normalized_velocity_reward = velocity_reward / MAX_VEL_REWARD
            normalized_orientation_reward = orientation_reward / MAX_ORI_REWARD

            # Total reward
            total_reward = normalized_distance_reward + normalized_velocity_reward + normalized_orientation_reward
            total_reward *= self.dt_sec

            # Shut off engine near the ground and give a huge reward bonus for landing upright and with low velocity
            if self.pos_n[2] < self.MIN_GROUND_DIST_M:
                if self.engine_on == True:
                    self.engine_on = False
                    if self.interactive:
                        print("Engine off (altitude: %.3f)" % (self.pos_n[2]))
                    if np.abs(self.roll_deg) < 5.0:
                        total_reward += 500.0
                    if np.abs(self.pitch_deg) < 5.0:
                        total_reward += 500.0
                    if velocity_magnitude < 2.0:
                        total_reward += 1000.0

            return total_reward

    def get_total_mass(self, body_id):
        # Start with the mass of the base link
        total_mass = p.getDynamicsInfo(body_id, -1, physicsClientId=self.CLIENT)[0]  # -1 refers to the base link

        # Add up the masses of all other links
        num_links = p.getNumJoints(body_id, physicsClientId=self.CLIENT)
        for i in range(num_links):
            total_mass += p.getDynamicsInfo(body_id, i, physicsClientId=self.CLIENT)[0]

        return total_mass

