import sys
import os
from scipy.integrate import odeint
from scipy.interpolate import interp1d

paths_to_add = [
    os.path.abspath(os.path.join('../..')),  
    os.path.abspath(os.path.join('..'))  
]

sys.path.extend(path for path in paths_to_add if path not in sys.path)

import jax
import jax.numpy as jnp
from fbpinns.problems import Problem
import matplotlib.pyplot as plt


class PredatorPrey(Problem):
    """
    u,t = au - buv
    v,t = cuv - dv
    u--->Prey population, a>0--->growth rate, b>0--->predator damage
    v--->Predator population, c>0--->predator advantage, d>0--->mortality
    u(0)=u0, v(0)= v0
    Stationary points: (u,v)=(0,0)--->trivial steady state. Unstable
                    & (u,v)=(d/c , a/b) ---> Non-trivial steady state. Stable, but not asymptotically stable
    Initial set a = 1, b=0.1, c=0.75, d=0.25; u*=d/c=1/3; v*=a/b=10
    another set: params = [0.5, 0.1, 0.25, 1.25], with u*=5, v*=5
    """

    @staticmethod
    def init_params(params=[1.0, 0.1, 0.75, 0.25], 
                    u0=2, v0=1, sd=0.1, time_limit=[0,24], 
                    numx=50, lambda_phy=1e0, lambda_data=1e0,
                    sparse=False, noise_level=0.0, tend=50):
        
        a, b, c, d = params 
        static_params = {
            "dims":(2,1),   # dims of solution and problem
            "a_true":a,
            "b_true":b,
            "c_true":c,
            "d_true":d,
            "u0":u0,
            "v0":v0,
            "sd":sd,
            "time_limit":time_limit,
            "numx":numx,
            "lambda_phy": lambda_phy,
            "lambda_data": lambda_data,
            "sparse":sparse,
            "noise_level":noise_level,
            "tend":tend
        }
        trainable_params = {
            "a":jnp.array(0.),
            "b":jnp.array(0.),
            "c":jnp.array(0.),
            "d":jnp.array(0.),
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
        time_limit = all_params["static"]["problem"]["time_limit"]
        numx = all_params["static"]["problem"]["numx"]
        if all_params['static']['problem']['sparse']:
            x_batch_data = jnp.sort(jax.random.uniform(key=key, shape=(numx,1), minval=time_limit[0], maxval=time_limit[1]), axis=0)
        else:
            x_batch_data = jnp.linspace(time_limit[0],time_limit[1],numx).astype(float).reshape((numx,1)) 
        noise = jax.random.normal(key, shape=(numx,1))  * all_params['static']['problem']['noise_level']
        solution = PredatorPrey.exact_solution(all_params, x_batch_data)
        u_data = solution[:,0] + noise
        v_data = solution[:,1] + noise
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
        
        a, b, c, d = [all_params['trainable']["problem"][key] for key in ('a', 'b', 'c', 'd')]
        lambda_phy = all_params["static"]["problem"]["lambda_phy"]
        lambda_data = all_params["static"]["problem"]["lambda_data"]
        # Physics loss
        _, u, v, ut, vt = constraints[0]
        phys1 = jnp.mean((ut - a*u + b*v*u)**2)
        phys2 = jnp.mean((vt - c*u*v + d*v)**2)
        phys = lambda_phy*(phys1 + phys2)

        # Data Loss
        _, ud, vd, u, v = constraints[1]
        u = u.reshape(-1) 
        v = v.reshape(-1) 
        data = lambda_data*(jnp.mean((u-ud)**2) + lambda_data*jnp.mean((v-vd)**2))

        # Penalty for negative parameters
        penalty_factor = 1e6
        penalty_terms = [a,b,c,d]
        penalties = sum(jnp.where(param < 0, penalty_factor * (param ** 2), 0) for param in penalty_terms)
        
        return phys + data + penalties
    
    @staticmethod
    def model(y, t, params):
        """
        Compute the derivatives of the system at time t.
        
        :param y: Current state of the system [u, v].
        :param t: Current time.
        :param params: Parameters of the model (a, b, c, d).
        :return: Derivatives [du/dt, dv/dt].
        """
        u, v = y  
        a, b, c, d = params  
        
        # Define the equations
        du_dt = a*u - b*v*u
        dv_dt = c*u*v - d*v
        
        return [du_dt, dv_dt]
    
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        a,b,c,d = [all_params['static']["problem"][key] for key in ('a_true', 'b_true', 'c_true', 'd_true')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [a,b,c,d ]
        tend = all_params["static"]["problem"]["tend"]
        
        t = jnp.arange(0, tend+0.02, 0.02)  
        
        # Solve the system 
        solution = odeint(PredatorPrey.model, [u0, v0], t, args=(params,))
        
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
        a,b,c,d = [all_params['trainable']["problem"][key] for key in ('a', 'b', 'c', 'd')]
        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        params = [a,b,c,d ]

        solution = odeint(PredatorPrey.model, [u0, v0], x_batch, args=(params,))

        return solution