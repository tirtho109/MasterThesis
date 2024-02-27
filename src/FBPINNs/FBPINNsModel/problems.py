import sys
import os
from scipy.integrate import odeint
from scipy.interpolate import interp1d

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import jax.numpy as jnp
from fbpinns.problems import Problem


class SaturatedGrowthModel(Problem):
    """
    u' = u(C-u)
    I.C.
    u(0) = u_0 = 0.01
    We have to pass: "C_ture":C, "u_0":u0,"sd":sd,
            "time_limit":time_limit, "numx":numx,
    """

    @staticmethod
    def init_params(C=1, u0=0.01, sd=0.1, time_limit=[0, 24], numx=50):
        
        static_params = {
            "dims":(1,1),
            "C_ture":C,
            "u_0":u0,
            "sd":sd,
            "time_limit":time_limit,
            "numx":numx,

        }
        trainable_params = {
            "C":jnp.array(0.), # learn C from constraints
        }
        
        return static_params, trainable_params
    
    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        u0 = all_params["static"]["problem"]["u_0"]
        C = all_params["static"]["problem"]["C_ture"]

        exp = jnp.exp(-C*x_batch[:,0:1])
        u = C / (1 + ((C - u0) / u0) * exp)
        return u
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),
            (0, (0,)),
        )
        time_limit = all_params["static"]["problem"]["time_limit"]
        numx = all_params["static"]["problem"]["numx"]
        # Data Loss
        x_batch_data = jnp.linspace(time_limit[0],time_limit[1],numx).astype(float).reshape((numx,1)) #20 data observation # numx = 20, time_limit = [0,24]
        u_data = SaturatedGrowthModel.exact_solution(all_params, x_batch_data)
        required_ujs_data = (
            (0, ()),
        )
        return [[x_batch_phys, required_ujs_phys],
                [x_batch_data, u_data, required_ujs_data]]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        u0 = all_params["static"]["problem"]["u_0"]

        x, tanh = x_batch[:,0:1], jnp.tanh

        u = u0 + tanh(x/sd) * u
        
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        C = all_params["trainable"]["problem"]["C"]
        # physics loss
        _, u, ut = constraints[0]
        phys = jnp.mean((ut - u*(C-u))**2)

        # data loss
        _, uc, u = constraints[1]
        data = 1e6*jnp.mean((u-uc)**2)

        return phys + data
    
    @staticmethod
    def model(u, t, C):
        """Defines the ODE to be solved: du/dt = u * (C - u)."""
        return u * (C - u)
    
    @staticmethod
    def learned_solution(all_params, x_batch):
        """Solves the ODE for given initial conditions and  learned parameters."""
        # Extracting parameters and initial condition
        C = all_params['trainable']["problem"]["C"]
        u0 = all_params["static"]["problem"]["u_0"]
        
        # Solving the ODE
        solution = odeint(SaturatedGrowthModel.model, u0, x_batch, args=(C,))
        
        return solution
    

class CompetitionModel(Problem):

    @staticmethod
    def init_params(params=[0.5, 0.7, 0.3, 0.3, 0.6], u0=2, v0=1, sd=0.1, time_limit=[0,24], numx=50):
        
        r, a1, a2, b1, b2 = params 
        static_params = {
            "dims":(2,1),   # dims of solution and problem
            "r_true":r,
            "a1_true":a1,
            "a2_true":a2,
            "b1_true":b1,
            "b2_true":b2,
            "u0":u0,
            "v0":v0,
            "sd":sd,
            "time_limit":time_limit,
            "numx":numx,
        }
        trainable_params = {
            "r":jnp.array(0.),
            "a1":jnp.array(0.),
            "a2":jnp.array(0.),
            "b1":jnp.array(0.),
            "b2":jnp.array(0.),
        }
        return static_params, trainable_params
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),  
            (1, ()),  
            (0, (0,)), 
            (1, (0,)),  
        )

        # Data Loss
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        time_limit = all_params["static"]["problem"]["time_limit"]
        numx = all_params["static"]["problem"]["numx"]
        x_batch_data = jnp.linspace(time_limit[0], time_limit[1],numx).astype(float).reshape((-1,1))
        r_true, a1_true, a2_true, b1_true, b2_true = [all_params['static']["problem"][key] for key in ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')]
        params = (r_true, a1_true, a2_true, b1_true, b2_true)
        solution = odeint(CompetitionModel.model, [u0,v0], x_batch_data.reshape((-1,)), args=(params,))
        u_data = solution[:,0]
        v_data = solution[:,1]
        required_ujs_data = (
            (0, ()), 
            (1, ()),  
        )

        return [[x_batch_phys, required_ujs_phys],
                [x_batch_data, u_data, v_data, required_ujs_data]]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, solution):
        sd = all_params["static"]["problem"]["sd"]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]

        x, tanh = x_batch[:,0:1], jnp.tanh

        u = solution[:, 0:1] * tanh(x/sd) + u0 # Hard constraining
        v = solution[:, 1:2] * tanh(x/sd)  + v0

        return jnp.concatenate([u, v], axis=1)
    
    @staticmethod
    def loss_fn(all_params, constraints):
        
        r, a1, a2, b1, b2 = [all_params['trainable']["problem"][key] for key in ('r', 'a1', 'a2', 'b1', 'b2')]

        # Physics loss
        _, u, v, ut, vt = constraints[0]
        phys1 = jnp.mean((ut - u + a1*u**2 + a2*u*v)**2)
        phys2 = jnp.mean((vt - r*v + r*b1*u*v + r*b2*v**2)**2)
        phys = phys1 + phys2

        # Data Loss
        _, ud, vd, u, v = constraints[1]
        u = u.reshape(-1) 
        v = v.reshape(-1) 
        data = 1e6*jnp.mean((u-ud)**2) + 1e6*jnp.mean((v-vd)**2)
        
        return phys + data
    
    @staticmethod
    def model(y, t, params):
        """
        Compute the derivatives of the system at time t.
        
        :param y: Current state of the system [u, v].
        :param t: Current time.
        :param params: Parameters of the model (a1, a2, b1, b2, r).
        :return: Derivatives [du/dt, dv/dt].
        """
        u, v = y  
        r, a1, a2, b1, b2 = params  
        
        # Define the equations
        du_dt = u * (1 - a1 * u - a2 * v)
        dv_dt = r * v * (1 - b1 * u - b2 * v)
        
        return [du_dt, dv_dt]
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        r, a1, a2, b1, b2 = [all_params['static']["problem"][key] for key in ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [r, a1, a2, b1, b2]
        
        t = jnp.arange(0, 25.02, 0.02)  
        
        # Solve the system 
        solution = odeint(CompetitionModel.model, [u0, v0], t, args=(params,))
        
        # Interpolation 
        u_interp = interp1d(t, solution[:, 0], kind='cubic')
        v_interp = interp1d(t, solution[:, 1], kind='cubic')
        
        u_data = u_interp(x_batch.flatten())
        v_data = v_interp(x_batch.flatten())
        
        # Combine 
        combined_solution = jnp.vstack((u_data, v_data)).T
        if batch_shape:
            combined_solution = combined_solution.reshape(batch_shape + (2,))
        
        return combined_solution
    
    @staticmethod
    def learned_solution(all_params, x_batch):
        # r_true, a1_true, a2_true, b1_true, b2_true = [all_params['static']["problem"][key] for key in ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')]
        r, a1, a2, b1, b2 = [all_params['trainable']["problem"][key] for key in ('r', 'a1', 'a2', 'b1', 'b2')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [r, a1, a2, b1, b2]

        solution = odeint(CompetitionModel.model, [u0, v0], x_batch, args=(params,))

        return solution
    
