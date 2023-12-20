import numpy as np
import scipy.linalg
import time
from simrocketenv import SimRocketEnv
from geodetic_toolbox import *
import threading
import copy

from acados_template import AcadosOcp, AcadosOcpSolver
from mpc.rocket_model import export_rocket_ode_model
# from casadi import SX, vertcat, cos, sin, sqrt, sumsqr

# Global messagebox to exchange data between threads as shown above
g_thread_msgbox = {
    'keymap' : { 'longitudinal_cmd': 0.0, 'lateral_cmd': 0.0, 'yaw_cmd': 0.0, 'vertical_cmd': 0.0, 'yaw_cmd': 0.0, }, # keyboard input
    'mpc_fps' : 0,                     # debug information: fps of NMPC thread
    'render_fps' : 0,                  # debug information: fps of render thread
}
g_thread_msgbox_lock = threading.Lock() # only access g_thread_msgbox with this lock
g_sim_running = True # Run application as long as this is set to True

# This thread's job is to consume the simulation state vector and emit a
# control output u (that is again consumed by the physics simulation)
def nmpc_thread_func(initial_state):
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    ocp = AcadosOcp() # create ocp object to formulate the OCP
    model = export_rocket_ode_model()
    ocp.model = model
    Tf = 3.0    # Time horizon in seconds
    nx = model.x.size()[0]  # state length
    nu = model.u.size()[0]  # control input u vector length
    ny = nx + nu
    ny_e = nx
    N_horizon = int(20*Tf)  # Epochs for MPC prediction horizon
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    Q_mat = np.diag(model.weight_diag)  # state weight
    R_mat = np.diag(np.ones(nu, )*100.0)  # weight on control input u
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(nx)

    setpoint_yref = np.zeros((ny, ))
    setpoint_yref[0] = 1.0  # set q0 (real) unit quaternion part to 1.0
    setpoint_yref[9] = 2.42 # set new setpoint altitude component
    ocp.cost.yref = setpoint_yref  # setpoint trajectory
    ocp.cost.yref_e = setpoint_yref[0:nx] # setpoint end

    # Constraints
    ocp.constraints.constr_type = 'BGH' # Comprises simple bounds, polytopic constraints, general non-linear constraints.
    ocp.constraints.lbu = np.array([ 0.20, -1.0, -1.0, -1.0, -1.0 ])
    ocp.constraints.ubu = np.array([ 1.00,  1.0,  1.0,  1.0,  1.0 ])
    ocp.constraints.x0 = initial_state
    ocp.constraints.idxbu = np.array(range(nu))

    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK' # IRK, GNSF, ERK
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP or SQP_RTI
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    # # make sure a MPC update is performed in the first epoch
    MPC_DT_SEC = 1.0 / 100.0  # run the NMPC every XX ms
    timestamp_last_mpc_update = time.time() - 2*MPC_DT_SEC
    mpc_step_counter = 0  # +1 for every mpc step, reset every 1 sec
    timestamp_last_mpc_fps_update = time.time()

    while g_sim_running:

        timestamp_current = time.time()
        if timestamp_current - timestamp_last_mpc_update < MPC_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            keymap = copy.deepcopy(g_thread_msgbox['keymap']) # read input from render thread
            state = copy.deepcopy(g_thread_msgbox['state'])
            if timestamp_current - timestamp_last_mpc_fps_update >= 1.0:
                g_thread_msgbox['mpc_fps'] = mpc_step_counter
                mpc_step_counter = 0
                timestamp_last_mpc_fps_update = timestamp_current

        # solve OCP and get next control input
        try:
            u = acados_ocp_solver.solve_for_x0(x0_bar=state)
        except Exception as e:
            print(e)

        timestamp_last_mpc_update = timestamp_current
        mpc_step_counter += 1
        with g_thread_msgbox_lock:
            g_thread_msgbox['u'] = np.copy(u) # output control vector u

def main():
    global g_thread_msgbox
    global g_thread_msgbox_lock
    global g_sim_running

    # SimRocketEnv is handling the physics simulation
    env = SimRocketEnv()
    g_thread_msgbox['state'] = env.state
    u = None
    predictedX = None

    # Spawn NMPC thread (doing the control work)
    nmpc_thread = threading.Thread(target=nmpc_thread_func, kwargs={'initial_state': g_thread_msgbox['state']})
    nmpc_thread.start()

    timestamp_lastupdate = time.time()
    MAX_DT_SEC = 0.1 # don't allow larger simulation timesteps than this
    SIM_DT_SEC = 1.0 / 240.0  # run the simulation every XX ms
    sim_step_counter = 0  # +1 for every simulation step, reset every 1 sec
    # emit a FPS stat message every second based on this timestamp:
    last_fps_update = timestamp_lastupdate

    while g_sim_running:
        timestamp_current = time.time()
        dt_sec = timestamp_current - timestamp_lastupdate
        if dt_sec < SIM_DT_SEC:
            time.sleep(0)
            continue

        with g_thread_msgbox_lock:
            if 'u' in g_thread_msgbox:
                u = copy.deepcopy(g_thread_msgbox['u']) # get control vector u from NMPC thread
            else:
                continue
        # wait with the simulation thread until we receive the first u vector
        # e.g. the MPC thread needs to compile .c files until it is ready

        timestamp_lastupdate = timestamp_current
        # make sure dt_sec is within a reasonable range
        if dt_sec > MAX_DT_SEC:
            print("Warning, high dt_sec: %.1f" % (dt_sec))
            continue
        dt_sec = np.clip(dt_sec, 0.0, MAX_DT_SEC)

        env.dt_sec = dt_sec
        state, reward, done, _ = env.step(u) # update physics simulation
        sim_step_counter += 1

        if done == True:
            g_sim_running = False

        with g_thread_msgbox_lock:
            g_thread_msgbox['state'] = state
            mpc_fps       = g_thread_msgbox['mpc_fps']
            render_fps    = g_thread_msgbox['render_fps']

        if timestamp_current - last_fps_update >= 0.1:
            print("FPS=%3i SIM=%4i MPC=%3i" % (render_fps, sim_step_counter, mpc_fps), end=' ')
            last_fps_update = timestamp_current

            sim_step_counter = 0

            env.print_state()
            for elem in u:
                print("%3.0f" % (elem*99.0), end=' ')
            print("")

    nmpc_thread.join()

if __name__ == '__main__':
    main()

