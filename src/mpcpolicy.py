from basepolicy import BasePolicy

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc.rocket_model import export_rocket_ode_model
# from casadi import SX, vertcat, cos, sin, sqrt, sumsqr

class MPCPolicy(BasePolicy):
    def __init__(self, initial_state):
        super().__init__()

        self.ocp = AcadosOcp() # create ocp object to formulate the OCP
        self.model = export_rocket_ode_model()
        self.ocp.model = self.model
        self.Tf = 3.0    # Time horizon in seconds
        self.nx = self.model.x.size()[0]  # state length
        self.nu = self.model.u.size()[0]  # control input u vector length
        self.ny = self.nx + self.nu
        self.ny_e = self.nx
        self.N_horizon = int(20*self.Tf)  # Epochs for MPC prediction horizon
        self.ocp.dims.N = self.N_horizon

        # set cost module
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.Q_mat = np.diag(self.model.weight_diag)  # state weight
        self.R_mat = np.diag(np.ones(self.nu, )*100.0)  # weight on control input u
        self.ocp.cost.W = scipy.linalg.block_diag(self.Q_mat, self.R_mat)
        self.ocp.cost.W_e = self.Q_mat
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.Vu = np.zeros((self.ny, self.nu))
        self.Vu[self.nx : self.nx + self.nu, 0:self.nu] = np.eye(self.nu)
        self.ocp.cost.Vu = self.Vu
        self.ocp.cost.Vx_e = np.eye(self.nx)

        setpoint_yref = np.zeros((self.ny, ))
        setpoint_yref[0] = 1.0  # set q0 (real) unit quaternion part to 1.0
        setpoint_yref[9] = 2.42 # set new setpoint altitude component
        self.ocp.cost.yref = setpoint_yref  # setpoint trajectory
        self.ocp.cost.yref_e = setpoint_yref[0:self.nx] # setpoint end

        # Constraints
        self.ocp.constraints.constr_type = 'BGH' # Comprises simple bounds, polytopic constraints, general non-linear constraints.
        self.ocp.constraints.lbu = np.array([ 0.20, -1.0, -1.0, -1.0, -1.0 ])
        self.ocp.constraints.ubu = np.array([ 1.00,  1.0,  1.0,  1.0,  1.0 ])
        self.ocp.constraints.x0 = initial_state
        self.ocp.constraints.idxbu = np.array(range(self.nu))

        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK' # IRK, GNSF, ERK
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP or SQP_RTI
        self.ocp.solver_options.qp_solver_cond_N = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        solver_json = 'acados_ocp_' + self.model.name + '.json'
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=solver_json)

    def get_name(self):
        return "MPC"

    def predict(self, observation):
        # solve OCP and get next control input
        action = self.acados_ocp_solver.solve_for_x0(x0_bar=observation)

        # emit 5 state vectors from the prediction horizon
        num_pred_epochs = 5
        step_size = self.N_horizon // num_pred_epochs

        predictedX = np.ndarray((num_pred_epochs, self.nx))
        for i in range(num_pred_epochs):
            predictedX[i,:] = self.acados_ocp_solver.get(i * step_size, "x")

        return action, predictedX
