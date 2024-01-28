# (c) Jan Zwiener (jan@zwiener.org)
#
# This file sets up a non-linear model predictive control
# class to control the rocket in the simrocketenv.py simulation.
# The core functionality to control the rocket is setup in the
# mpc/rocket_model.py function.
#
# Create an instance of this class and call the step()
# method with a state vector and the return value
# is a control input (u) vector to control the rocket.

from basecontrol import BaseControl

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc.rocket_model import export_rocket_ode_model
# from casadi import SX, vertcat, cos, sin, sqrt, sumsqr

class MPCPolicy(BaseControl):
    def __init__(self, initial_state, time_horizon=3.0, epochs_per_sec=20):
        super().__init__()

        self.ocp        = AcadosOcp() # create ocp object to formulate the OCP
        self.model      = export_rocket_ode_model()
        self.ocp.model  = self.model
        self.Tf         = time_horizon # Time horizon in seconds
        self.nx         = self.model.x.size()[0]  # state length
        self.nu         = self.model.u.size()[0]  # control input u vector length
        self.ny         = self.nx + self.nu
        self.ny_e       = self.nx
        self.N_horizon  = int(epochs_per_sec*self.Tf) # prediction horizon
        self.ocp.dims.N = self.N_horizon

        # set cost module
        self.ocp.cost.cost_type   = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.Q_mat                = np.diag(self.model.weight_diag)  # state weight
        self.R_mat                = np.diag(np.ones(self.nu, )*100.0)  # weight on control input u
        self.ocp.cost.W           = scipy.linalg.block_diag(self.Q_mat, self.R_mat)
        self.ocp.cost.W_e         = self.Q_mat
        self.Vu                   = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vx_e        = np.eye(self.nx)

        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = self.Vu
        self.Vu[self.nx : self.nx + self.nu, 0:self.nu] = np.eye(self.nu)

        # Setpoint state
        setpoint_yref        = np.zeros((self.ny, ))
        setpoint_yref[0]     = 1.0  # set q0 (real) unit quaternion part to 1.0
        setpoint_yref[9]     = 2.42 # set new setpoint altitude component
        self.ocp.cost.yref   = setpoint_yref  # setpoint trajectory
        self.ocp.cost.yref_e = setpoint_yref[0:self.nx] # setpoint end

        # Constraints
        # BGH: Comprises simple bounds, polytopic constraints, general
        # non-linear constraints.
        self.ocp.constraints.constr_type = 'BGH'
        self.ocp.constraints.lbu = np.array([ 0.20, -1.0, -1.0, -1.0, -1.0 ])
        self.ocp.constraints.ubu = np.array([ 1.00,  1.0,  1.0,  1.0,  1.0 ])
        self.ocp.constraints.x0 = initial_state
        self.ocp.constraints.idxbu = np.array(range(self.nu))

        # Solver options
        self.ocp.solver_options.qp_solver        = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type  = 'ERK' # IRK, GNSF, ERK
        self.ocp.solver_options.nlp_solver_type  = 'SQP_RTI'  # SQP or SQP_RTI
        self.ocp.solver_options.qp_solver_cond_N = self.N_horizon
        self.ocp.solver_options.tf               = self.Tf

        solver_json = 'acados_ocp_' + self.model.name + '.json'
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)

    def get_name(self):
        return "MPC"

    def next(self, observation):
        # solve OCP and get next control input
        action = self.acados_ocp_solver.solve_for_x0(x0_bar=observation)

        # emit 5 state vectors from the prediction horizon
        NUM_PRED_EPOCHS = 5
        step_size = self.N_horizon // NUM_PRED_EPOCHS

        predictedX = np.ndarray((NUM_PRED_EPOCHS, self.nx))
        for i in range(NUM_PRED_EPOCHS):
            predictedX[i,:] = self.acados_ocp_solver.get(i * step_size, "x")

        return action, predictedX

