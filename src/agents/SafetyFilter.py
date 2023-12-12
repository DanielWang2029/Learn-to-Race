from casadi import *
import numpy as np
from scipy.linalg import fractional_matrix_power
import math
import dill
import ipdb

class PredictiveSafetyFilter:
    def __init__(self, system=None, x_init=None, horizon=None, obj_weights = None, state_constr = None, input_constr = None):
        self.system = system
        if hasattr(system, 'sigma_w'):
            self.sigma_w = system.sigma_w
        else:
            self.sigma_w = 0.0

        self.x_init = x_init
        self.horizon = horizon
        self.nx, self.nu = system.nx, system.nu

        if obj_weights is not None:
            self.Q, self.R, self.Q_T = obj_weights['Q'], obj_weights['R'], obj_weights['Q_T']
            if self.R.dim() == 1:
                self.R = self.R.unsqueeze(0)
        else:
            self.Q, self.R, self.Q_T = torch.eye(self.nx), torch.eye(self.nu), torch.eye(self.nx)

        self.augmented_nn_input = self.system.augmented_nn_input

        self.state_constr = state_constr
        self.input_constr = input_constr

    def local_nn_linear_bounds(self, local_domain, method = 'backward'):
        T = len(local_domain)
        nn_model = self.system.nn_model
        linear_bounds = []
        for i in range(T):
            bds = local_domain[i]
            if self.augmented_nn_input:
                x_lb, x_ub, u_lb, u_ub = bds['x_lb'], bds['x_ub'], bds['u_lb'], bds['u_ub']
            else:
                x_lb, x_ub, u_lb, u_ub = bds['x_lb'], bds['x_ub'], None, None
            _, _, linear_bds = nn_linear_bounds_abstraction(nn_model, x_lb, x_ub, u_lb, u_ub, method = method)
            linear_bounds.append(linear_bds)
        return linear_bounds

    def affine_dynamics_data_transform(self, linear_bounds):
        A = self.system.A
        B = self.system.B
        nx, nu = self.nx, self.nu
        W_t, c_t, new_linear_bounds = linear_bounds_recenter(linear_bounds)

        if self.augmented_nn_input:
            A_t = [A + item[:, :nx] for item in W_t]
            B_t = [B + item[:, nx:] for item in W_t]
        else:
            A_t = [A + item for item in W_t]
            B_t = [B] * len(W_t)

        return A_t, B_t, c_t, new_linear_bounds

    def solve(self, ref_control = None, init = None, horizon = None, x_eps = 1.2, u_eps = 1.2, enlarge_multiplier = 1.1, shrink_multiplier = 1.2, num_iter = 20, options = None):
        # Solve the predictive safety filter problem with adaptive trust regions.
        # if init is None, then the nominal trajectory is obtained by simulating the NN system with the base policy starting from self.x_init
        # if init is not None and is a (nx) dim. tensor, then we choose the nominal trajectory as the rollout trajectory of the NN system with the base policy starting from init
        # if init is not None and is a dictionary containing the nominal trajectory of x_t and u_t, we set the nominal trajectory as init
        # x_eps, u_eps decide the size of the trust region, which will be changed according to the enlarge_multiplier or the shrink_multiplier in each iteration
        # in case of infeasibility of the predictive safety filter

        if horizon is None:
            horizon = self.horizon

        if init is None:
            x0 = self.x_init
            nominal_traj = self.system.simulate_nominal_dynamics(horizon, x0)
        else:
            if isinstance(init, torch.Tensor):
                x0 = init
                # reset the initial state
                self.x_init = x0
                nominal_traj = self.system.simulate_nominal_dynamics(horizon, x0)
            elif isinstance(init, dict):
                nominal_traj = init
                self.x_init = nominal_traj['x'][0]
                x0 = self.x_init
            else:
                raise ValueError('The type of the init argument is not supported.')

        if ref_control is None:
            ref_control = nominal_traj['u'][0]

        record_list = []
        freq_infeasible = 0
        is_feasible = False
        x_eps_list = []
        u_eps_list = []

        slack_value_max_list = []
        for k in tqdm(range(num_iter), desc='Safety_filter_loop'):
            x_eps_list.append(x_eps)
            u_eps_list.append(u_eps)
            # try:
            # print(f'x_eps: {x_eps}, u_eps: {u_eps}')
            record, local_domain, linear_bounds = self.solve_RMPC_with_local_bounds(nominal_traj, x_eps, u_eps, ref_control = ref_control, horizon = horizon, options = options)

            TR_slack_max = record['sol']['TR_slack_max']
            constr_slack_max = record['sol']['constr_slack_max']
            # print(f'TR slack max:{TR_slack_max}, constraint slack max: {constr_slack_max}')
            slack_value_max_list.append(max(TR_slack_max,constr_slack_max))

            if record['is_feasible']:
                is_feasible = True
                record_list.append(record)
                break
            else:
                # In case of infeasibility, keep the reference trajectory unchanged and shrink x_eps.
                x_eps = x_eps*enlarge_multiplier
                freq_infeasible = freq_infeasible + 1

                # update the nominal trajectory using RMPC
                if record['solver_status'] == 'feasible':
                    K, h_vec, v_vec = record['sol']['K'], record['sol']['h_vec'], record['sol']['v_vec']
                    device = x0.device
                    K, h_vec, v_vec = torch.from_numpy(K).to(device), torch.from_numpy(h_vec).to(device), torch.from_numpy(v_vec).to(device)

                    w_seq = torch.zeros((horizon, self.nx))
                    nominal_traj = self.system.simulate_dynamics_SLS_MPC_policy(x0, K, h_vec, v_vec, w_seq = w_seq)

                record_list.append(record)

        index = np.argmin(slack_value_max_list)
        # TR_slack_chosen = record_list[index]['sol']['TR_slack_max']
        # constr_slack_chosen = record_list[index]['sol']['constr_slack_max']
        # print(f'For the applied control inputs, TR slack var: {TR_slack_chosen}, constr slack var: {constr_slack_chosen}')
        u0 = record_list[index]['u0']
        u_ref = record_list[index]['u_ref']
        is_feasible = record_list[index]['is_feasible']

        return u0, u_ref, is_feasible, record_list[index], record_list

    def solve_RMPC_with_local_bounds(self, init_traj, x_eps, u_eps, ref_control = None, horizon = None, options = None):
        # solve the predictive filter problem using robust MPC and local linear bounds on the NN dynamics
        traj = init_traj
        local_domain = local_box_set_init(traj, x_eps, u_eps, init_box=False)
        linear_bounds = self.local_nn_linear_bounds(local_domain, method='backward')

        if ref_control is None:
            ref_control = traj['u'][0]

        if horizon is None:
            horizon = self.horizon

        A_t, B_t, c_t, new_linear_bounds = self.affine_dynamics_data_transform(linear_bounds)

        with torch.no_grad():
            sol = self.solve_SLS_MPC(local_domain, new_linear_bounds, ref_control = ref_control, horizon = horizon, A_t=A_t, B_t=B_t, c_t=c_t, options = options)

        is_feasible = (sol['slack_value_max'] < 1e-3)

        if not is_feasible:
            record =  {'is_feasible': is_feasible, 'traj': traj, 'u0': sol['u0'], 'u_dist': sol['u_dist'], 'u_ref': sol['u_ref'], 'local_domain': local_domain, 'linear_bounds': linear_bounds,
                 'sol': sol, 'solver_status': sol['solver_status'], 'x_eps': x_eps, 'u_eps': u_eps}
        else:
            record = {'is_feasible': is_feasible, 'traj': traj, 'u0': sol['u0'], 'u_dist': sol['u_dist'], 'u_ref': sol['u_ref'], 'local_domain': local_domain, 'linear_bounds': linear_bounds,
                 'sol': sol, 'solver_status': sol['solver_status'],  'x_eps': x_eps, 'u_eps': u_eps}

        return record, local_domain, linear_bounds


    def solve_SLS_MPC(self, local_domain, linear_bounds, ref_control = None, horizon = None, A_t = None, B_t = None, c_t = None, options = None):
        # local_domain: list of dict that contains lower and upper bounds on x, u
        # linear_bounds: list of dict that contains the weights and bias of linear lower and upper bounds on the nn
        # A_t, B_t, c_t: list of dynamics matrices/vectors that describes x_{t+1} = A_t x_t + B_t u_t + c_t + eta_t
        # all the above inputs are supposed to have length T = horizon

        device = torch.device('cpu')

        if horizon is None:
            T = self.horizon
        else:
            T = horizon

        if ref_control is None:
            ref_control = (local_domain[0]['u_lb'] + local_domain[0]['u_ub'])/2

        A = self.system.A.to(device).numpy()
        B = self.system.B.to(device).numpy()
        nx = self.nx
        nu = self.nu

        x0 = self.x_init.numpy()
        x0 = x0.flatten()

        if self.sigma_w is None:
            sigma_w = 1e-4
        else:
            sigma_w = self.sigma_w

        Phix = cp.Variable(((T + 1) * nx, (T + 1) * nx), symmetric=False)
        Phiu = cp.Variable(((T + 1) * nu, (T + 1) * nx), symmetric=False)
        Sigma = cp.Variable(((T + 1) * nx, (T + 1) * nx), symmetric=False)
        d = cp.Variable(T * nx)  # diagonal entries of Sigma

        h_vec = cp.Variable((T+1)*nx)
        v_vec = cp.Variable((T+1)*nu)

        # structure constraints
        constr = []

        # block lower triangular constraints
        constr += [Phix[i * nx:(i + 1) * nx, (i + 1) * nx:] == np.zeros((nx, (T - i) * nx)) for i in range(T)]
        constr += [Phiu[i * nu:(i + 1) * nu, (i + 1) * nx:] == np.zeros((nu, (T - i) * nx)) for i in range(T)]
        constr += [Sigma[i * nx:(i + 1) * nx, (i + 1) * nx:] == np.zeros((nx, (T - i) * nx)) for i in range(T)]

        # constraints on the diagonal blocks of Sigma
        constr += [Sigma[:nx, :nx] == np.eye(nx)]

        constr += [Sigma[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] == cp.diag(d[(i - 1) * nx:i * nx]) for i in range(1, T + 1)]

        # affine constraint
        eye_mat = np.eye((T + 1) * nx)
        Z_mat = np.kron(np.diag(np.ones(T), k=-1), np.eye(nx))
        if A_t is None:
            A_mat = block_diag(np.kron(np.eye(T), A), np.zeros((nx, nx)))
        else:
            A_t_numpy = [A_t[k].numpy() for k in range(T)]
            A_mat = block_diag(*A_t_numpy)
            A_mat = block_diag(A_mat, np.zeros((nx, nx)))

        if B_t is None:
            B_mat = block_diag(np.kron(np.eye(T), B), np.zeros((nx, nu)))
        else:
            B_t_numpy = [B_t[k].numpy() for k in range(T)]
            B_mat = block_diag(*B_t_numpy)
            B_mat = block_diag(B_mat, np.zeros((nx, nu)))

        if c_t is None:
            c_vec = np.zeros((T+1)*nx)
        else:
            c_t_numpy = [c_t[k].numpy() for k in range(T)]
            c_vec = np.concatenate(c_t_numpy)
            c_vec = np.concatenate((x0,  c_vec))

        # x_tilde = x - h_vec
        constr += [h_vec == np.linalg.inv(eye_mat - Z_mat@A_mat)@(Z_mat@B_mat@v_vec + c_vec)]
        constr += [(eye_mat - Z_mat @ A_mat) @ Phix - Z_mat @ B_mat @ Phiu == Sigma]

        u_ref = ref_control.detach().cpu().numpy()
        cost = cp.norm(v_vec[:nu] - u_ref, 2)**2

        # use soft constraints
        slack_penalty = 1e3

        # uncertainty over-approximation constraints
        for t in range(T):
            # extract linear lower and upper bounds
            uA, ubias, lA, lbias = linear_bounds[t]['uA'], linear_bounds[t]['ubias'], linear_bounds[t]['lA'], \
                                   linear_bounds[t]['lbias']
            uA, ubias, lA, lbias = uA.squeeze(0), ubias.squeeze(0), lA.squeeze(0), lbias.squeeze(0)
            uA, ubias, lA, lbias = uA.numpy(), ubias.numpy(), lA.numpy(), lbias.numpy()

            if self.augmented_nn_input:
                # if (x, u) is the input of the nn dynamics
                for i in range(nx):
                    e_i = np.zeros(nx)
                    e_i[i] = 1.0
                    # add e_i-associated constraints
                    term_nominal = e_i @ (uA @ cp.hstack((h_vec[t*nx:(t+1)*nx], v_vec[t*nu:(t+1)*nu])) + ubias)
                    sum_plus = cp.sum([cp.norm(e_i @ (uA @ cp.vstack((Phix[t * nx:(t + 1) * nx, j * nx:(j + 1) * nx],
                                                                           Phiu[t * nu:(t + 1) * nu, j * nx:(j + 1) * nx]))
                               - Sigma[(t + 1) * nx:(t + 2) * nx, j * nx:(j + 1) * nx]), 1) for j in range(1, t + 1)])

                    constr += [term_nominal + sum_plus + sigma_w <= d[t * nx + i]]

                    # add -e_i-associated constraints
                    term_nominal = -e_i @ (lA @ cp.hstack((h_vec[t*nx:(t+1)*nx], v_vec[t*nu:(t+1)*nu])) + lbias)
                    sum_plus = cp.sum([cp.norm(e_i @ (lA @ cp.vstack((Phix[t * nx:(t + 1) * nx, j * nx:(j + 1) * nx],
                                                                           Phiu[t * nu:(t + 1) * nu, j * nx:(j + 1) * nx]))
                               - Sigma[(t + 1) * nx:(t + 2) * nx, j * nx:(j + 1) * nx]), 1) for j in range(1, t + 1)])

                    constr += [term_nominal + sum_plus + sigma_w <= d[t * nx + i]]

            else:
                for i in range(nx):
                    e_i = np.zeros(nx)
                    e_i[i] = 1.0
                    # add e_i-associated constraints
                    term_nominal = e_i @ (uA @ h_vec[t * nx:(t + 1) * nx] + ubias)
                    sum_plus = cp.sum([cp.norm(e_i @ (uA @ Phix[t * nx:(t + 1) * nx, j * nx:(j + 1) * nx]
                                                      - Sigma[(t + 1) * nx:(t + 2) * nx, j * nx:(j + 1) * nx]), 1) for j
                                       in range(1, t + 1)])

                    constr += [term_nominal + sum_plus + sigma_w <= d[t * nx + i]]

                    # add -e_i-associated constraints
                    term_nominal = -e_i @ (lA @ h_vec[t * nx:(t + 1) * nx] + lbias)
                    sum_plus = cp.sum([cp.norm(e_i @ (lA @ Phix[t * nx:(t + 1) * nx, j * nx:(j + 1) * nx]
                                                      - Sigma[(t + 1) * nx:(t + 2) * nx, j * nx:(j + 1) * nx]), 1) for j
                                       in range(1, t + 1)])

                    constr += [term_nominal + sum_plus + sigma_w <= d[t * nx + i]]

        # locality constraints

        # create slack variables for hard constraints
        TR_slack_var = {'x': [], 'u':[]}
        constr_slack_var = {'x': [], 'u':[]}

        # trust region constraints
        u_ub, u_lb = local_domain[0]['u_ub'], local_domain[0]['u_lb']
        u_ub = u_ub.numpy().flatten()
        u_lb = u_lb.numpy().flatten()
        eye_mat = np.eye(nu)

        t = 0
        v = v_vec[t * nu:(t + 1) * nu]
        for i in range(nu):
            e = eye_mat[i]

            # create slack variables
            u_lb_slack = cp.Variable(u_lb.shape[0])
            u_ub_slack = cp.Variable(u_ub.shape[0])
            TR_slack_var['u'].append(u_lb_slack)
            TR_slack_var['u'].append(u_ub_slack)
            constr += [u_lb_slack >= 0, u_ub_slack >= 0]

            constr += [e @ v <= u_ub[i] + u_ub_slack[i] ]
            constr += [-e @ v <= -u_lb[i] + u_lb_slack[i] ]

        for t in range(1, T):
            local_bds = local_domain[t]

            # consider state constraints
            x_ub, x_lb = local_bds['x_ub'], local_bds['x_lb']

            x_ub = x_ub.numpy().flatten()
            x_lb = x_lb.numpy().flatten()

            # create slack variables
            x_lb_slack = cp.Variable(x_lb.shape[0])
            x_ub_slack = cp.Variable(x_ub.shape[0])
            TR_slack_var['x'].append(x_lb_slack)
            TR_slack_var['x'].append(x_ub_slack)

            constr += [x_lb_slack >= 0, x_ub_slack >= 0]

            eye_mat = np.eye(nx)
            for i in range(nx):
                e = eye_mat[i]
                h = h_vec[t*nx:(t+1)*nx]
                # we use the fact tilde_w_0 = 0
                constr += [e @ h + cp.norm(
                    e @ Phix[t * nx:(t + 1) * nx, nx:(t + 1) * nx], 1)  <= x_ub[i] + x_ub_slack[i] ]
                constr += [-e @ h + cp.norm(
                    -e @ Phix[t * nx:(t + 1) * nx, nx:(t + 1) * nx], 1) <= -x_lb[i] + x_lb_slack[i] ]

            # consider control input constraints
            u_ub, u_lb = local_bds['u_ub'], local_bds['u_lb']
            u_ub = u_ub.numpy().flatten()
            u_lb = u_lb.numpy().flatten()
            eye_mat = np.eye(nu)

            v = v_vec[t*nu:(t+1)*nu]

            # create slack variables
            u_lb_slack = cp.Variable(u_lb.shape[0])
            u_ub_slack = cp.Variable(u_ub.shape[0])
            TR_slack_var['u'].append(u_lb_slack)
            TR_slack_var['u'].append(u_ub_slack)

            constr += [u_lb_slack >= 0, u_ub_slack >= 0]

            for i in range(nu):
                e = eye_mat[i]
                constr += [e @ v + cp.norm(e @ Phiu[t * nu:(t + 1) * nu, nx:(t + 1) * nx], 1) <= u_ub[i] + u_ub_slack[i] ]

                constr += [-e @ v + cp.norm(-e @ Phiu[t * nu:(t + 1) * nu, nx:(t + 1) * nx], 1) <= -u_lb[i] + u_lb_slack[i] ]

        # state and input constraints
        if self.state_constr is not None:
            # state constraints
            state_constr = self.state_constr

            for t in range(1, T + 1):
                Ax, bx = state_constr[t]['A'], state_constr[t]['b']
                Ax, bx = Ax.cpu().numpy().astype(x0.dtype), bx.cpu().numpy().astype(x0.dtype)

                h = h_vec[t * nx:(t + 1) * nx]

                # create slack variables
                state_slack = cp.Variable(Ax.shape[0])
                constr_slack_var['x'].append(state_slack)

                constr += [state_slack >= 0]

                for i in range(Ax.shape[0]):
                    f, b = Ax[i], bx[i]
                    constr += [f @ h + cp.norm(
                        f @ Phix[t * nx:(t + 1) * nx, nx:(t + 1) * nx], 1) <= b + state_slack[i]]

        if self.input_constr is not None:
            # input constraints
            input_constr = self.input_constr

            for t in range(T):
                Au, bu = input_constr[t]['A'], input_constr[t]['b']
                Au, bu = Au.cpu().numpy().astype(x0.dtype), bu.cpu().numpy().astype(x0.dtype)

                v = v_vec[t * nu:(t + 1) * nu]

                # create slack variables
                input_slack = cp.Variable(Au.shape[0])
                constr_slack_var['u'].append(input_slack)

                constr += [input_slack >= 0]

                for i in range(Au.shape[0]):
                    f, b = Au[i], bu[i]
                    constr += [f @ v + cp.norm(f @ Phiu[t * nu:(t + 1) * nu, nx:(t + 1) * nx], 1) <= b + input_slack[i]]

        constr_slack_var_list = constr_slack_var['x'] + constr_slack_var['u']
        TR_slack_var_list = TR_slack_var['x'] + TR_slack_var['u']

        constr_slack = cp.hstack(constr_slack_var_list)
        TR_slack = cp.hstack(TR_slack_var_list)

        cost += 10*slack_penalty*cp.sum(constr_slack) + slack_penalty*cp.sum(TR_slack)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cvxpy_solver, verbose=False)

        cvx_status = prob.status
        solver_time = prob.solver_stats.solve_time

        if cvx_status not in ['infeasible', 'unbounded']:
            K = Phiu.value @ np.linalg.inv(Phix.value)

            # slack variable processing
            TR_slack_values = {'x': [], 'u': []}
            constr_slack_values = {'x': [], 'u':[]}

            if len(TR_slack_var['x']) == 0:
                TR_slack_values['x'] = [0.0]
            else:
                TR_slack_values['x'] = [item.value for item in TR_slack_var['x']]

            TR_slack_values['u'] = [item.value for item in TR_slack_var['u']]
            constr_slack_values['x'] = [item.value for item in constr_slack_var['x']]
            constr_slack_values['u'] = [item.value for item in constr_slack_var['u']]

            TR_slack_values_max = np.max([np.hstack(TR_slack_values['x']).max(), np.hstack(TR_slack_values['u']).max()])
            constr_slack_values_max = np.max([np.hstack(constr_slack_values['x']).max(), np.hstack(constr_slack_values['u']).max()])
            slack_value_max = max(TR_slack_values_max, constr_slack_values_max)

            v_vec_value = v_vec.value
            u0 = v_vec_value[:nu].astype('float32')
            u_dist = np.linalg.norm(u0 - u_ref)

            sol = {'Phix': Phix.value.astype(x0.dtype), 'Phiu': Phiu.value.astype(x0.dtype),
                   'Sigma': Sigma.value.astype(x0.dtype),
                   'd': d.value.astype(x0.dtype), 'K': K.astype(x0.dtype), 'h_vec': h_vec.value.astype(x0.dtype),
                   'v_vec': v_vec_value.astype(x0.dtype),
                   'u0': u0,
                   'u_dist': u_dist,
                   'u_ref': u_ref,
                   'cost': cost.value,
                   'cvx_status': cvx_status, 'solver_time': solver_time, 'solver_status': 'feasible',
                    'TR_slack_values': TR_slack_values, 'constr_slack_values': constr_slack_values,
                   'slack_value_max': slack_value_max, 'TR_slack_max': TR_slack_values_max, 'constr_slack_max': constr_slack_values_max,
                   }
        else:
            warnings.warn('SLS OCP infeasible!')
            sol = {'Phix': None, 'Phiu': None, 'Sigma': None, 'd': None, 'K': None, 'h_vec': None, 'cost': cost.value,
                   'v_vec': None, 'u0':None, 'u_dist': None, 'u_ref': None,
                   'cvx_status': cvx_status, 'solver_time': solver_time, 'solver_status': 'infeasible',
                   'slack_value_max': None, 'TR_slack_max': None,
                   'constr_slack_max': None, 'TR_slack_values': None, 'constr_slack_values': None,
                   }
        return sol

class MPSafetyFilter:
    def __init__(self,x_dim=None,u_dim=None, P=None, G_x=None, f_x=None, G_u=None, f_u=None, alpha=None, dynamics=None, N = 20):
        
        '''
            x_dim       : state dimension
            u_dim       : input dimension 
            G_x         : state constraint matrix
            f_x         : rhs of state constraints
            G_u         : inputs constraint matrix
            f_u         : rhs of input constraints
            alpha       : ellipsoidal level-set defined by matrix P
            N           : horizon length
        '''
        
        self.P     = P
        self.G_x   = G_x
        self.f_x   = f_x
        self.G_u   = G_u
        self.f_u   = f_u
        self.alpha = alpha
        self.N     = N
        self.opti   = casadi.Opti() 
        
        # state and action dimensions:
        self.x_dim = x_dim
        self.u_dim = u_dim
        
        # solver variable/parameter pointers for performance optimization
        self.u     = None
        self.x     = None
        self.u_L   = None

        # solver tolerance
        self.tol = 1e-8
 
        # define a dynamics callback 
        self.dynamics = dynamics

        # define the Safety Filter status
        self.status = {'feasible': True, 'setup': False}
    
    def setup(self):

        ''' 
             setup the optimization problem proposed in casadi opti.
        '''
        
        self.status['setup'] = True

        N       = self.N
        x_dim   = self.x_dim
        u_dim   = self.u_dim
        
        x0    = self.opti.parameter(x_dim,1)          # initial value
        u_L   = self.opti.parameter(u_dim,1)          # learning control input 
        x     = self.opti.variable(N*x_dim,1)         # N states concatenated into 1 vector
        u     = self.opti.variable((N-1)*u_dim,1)     # N inputs concatenated into 1 vector 
        
        # assign to instance variables for global access
        self.x0      = x0
        self.u       = u
        self.x       = x
        self.u_L     = u_L

        '''
            choose solver
        '''
        
        p_opts = {}
        s_opts = {'print_level' : 0,
                  'print_user_options': "no",
                  'print_options_documentation': "no",
                  'print_frequency_iter': 10000,
                  'max_iter' : 100000,
                  'tol'      : self.tol}
         
        self.opti.solver("ipopt", p_opts, s_opts)
        
        '''
            cost function 
        '''
        
        self.opti.minimize((u[0:u_dim] - u_L).T@(u[0:u_dim] - u_L))
         

        '''
            dynamics constraint
        '''
        
        self.opti.subject_to(x[0:x_dim]  == x0)
        for i in range(1,N):

            self.opti.subject_to(x[i*x_dim:(i+1)*x_dim] == self.dynamics(x[(i-1)*x_dim:i*x_dim],
                                                                         u[(i-1)*u_dim:i*u_dim]))
           
        '''
            state constraints with the slack variables
        '''
        
        for i in range(0,N-1):
            self.opti.subject_to(self.G_x@x[i*x_dim:(i+1)*x_dim] <= self.f_x) 
         
        '''
            input constraints 
        '''
        for i in range(0,N-1):
            self.opti.subject_to(self.G_u@u[i*u_dim:(i+1)*u_dim] <= self.f_u) 

        '''
            terminal constraint
        '''
        
        self.opti.subject_to(x[-x_dim:].T@self.P@x[-x_dim:] <= self.alpha) 
        pass

    def solve(self, x0=None, u_L=None):
        
        # solve
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.u_L, u_L) 
        try:
            sol = self.opti.solve()
            self.status['feasible'] = True
        except:
            print('Safety filter is still infeasible. Switching back to the stabilizing controller')
            self.status['feasible'] = False

        # return the control input along with the planned trajectory
        u0    = sol.value(self.u)[:self.u_dim].reshape((self.u_dim,1))
        
        return u0
    
    def save(self, filename):
        if self.status['setup'] == False:
            with open(filename, 'wb') as output:
                dill.dump(self, output)
        else:
            raise Exception('must save the controller before setting it up. Casadi variables cannot be pickled.')

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['opti']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.opti = casadi.Opti()

def getSafetyFilter(option=1):
  try:
    if option == 1:
      return MPSafetyFilter()
    else:
      return PredictiveSafetyFilter()
  except Exception as e:
    return None
