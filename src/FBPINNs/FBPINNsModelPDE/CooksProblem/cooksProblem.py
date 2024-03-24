import os 
import sys
import time

paths_to_add = [
    os.path.abspath(os.path.join('../..')),  
    os.path.abspath(os.path.join('..'))  
]

sys.path.extend(path for path in paths_to_add if path not in sys.path)

import jax
from jax import vmap
import jax.numpy as jnp
from fbpinns.problems import Problem

class CooksProblemForwardSoft(Problem):
    """ Linear Elasticity, plain strain, DBC at left, NBC at right"""
    @staticmethod
    def init_params(lambda_true= 4, mu_true = 5, nbc_points_right = 10, nbc_points_top=10, nbc_points_bottom=10, sd=0.1):
        E = lambda_true * jnp.einsum("ij, kl->ijkl", jnp.eye(2), jnp.eye(2)) + mu_true *(
                jnp.einsum("ik, jl->ijkl", jnp.eye(2), jnp.eye(2)) + jnp.einsum("il, jk->ijkl", jnp.eye(2), jnp.eye(2))
            )
        static_params = {
            "dims":(5,2), # Out: ux, uy,sigma_xx, sigma_yy, sigma_xy[symmetric], in: (x,y)
            "E":E,
            "nbc_points_right":nbc_points_right,
            "nbc_points_top":nbc_points_top,
            "nbc_points_bottom":nbc_points_bottom,
            "sd":sd
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (   # we need: ux,x; uy,y; ux,y; uy,x; sigma_xx,x; sigma_yy,y; sigma_xy,x; sigma_xy,y
            (0, (0,)),      # ux,x (out_idx, (in_idx, in_idx))
            (1, (1,)),      # uy,y
            (0, (1,)),      # ux, y
            (1, (0,)),      # uy, x
            (2, (0,)),      # sigma_xx,x
            (3, (1,)),      # sigma_yy,y;
            (4, (0,)),      # sigma_xy,x; 
            (4, (1,)),      # sigma_xy,y
            (2,()),         # stress_xx
            (3,()),         # stress_yy
            (4,()),         # stress_xy or stress_yx
        )
        nbc_points_right = all_params["static"]["problem"]["nbc_points_right"]
        nbc_points_top = all_params["static"]["problem"]["nbc_points_top"]
        nbc_points_bottom = all_params["static"]["problem"]["nbc_points_bottom"]
        batch_shapes = ((nbc_points_bottom,),(nbc_points_top,),(0,),(nbc_points_right,)) # bottom, top, left, right
        x_batch_neumann_bottom, x_batch_neumann_top, x_batch_dirichlet_left,x_batch_neumann_right = domain.sample_boundaries(all_params, key, sampler, batch_shapes)
        required_ujs_neumann_right = (
            (2, ()),      # sigma_xx,
            (3, ()),      # sigma_yy
            (4, ()),      # sigma_xy
        )

        required_ujs_neumann_top = (
            (2, ()),      # sigma_xx,
            (3, ()),      # sigma_yy
            (4, ()),      # sigma_xy
        )

        required_ujs_dirichlet_left= (
            (0, ()),        # ux
            (1, ()),        # uy
        )

        required_ujs_neumann_bottom = (
            (2, ()),      # sigma_xx,
            (3, ()),      # sigma_yy
            (4, ()),      # sigma_xy
        )

        return [[x_batch_phys, required_ujs_phys],
                [x_batch_neumann_right, required_ujs_neumann_right],
                [x_batch_neumann_top, required_ujs_neumann_top],
                [x_batch_dirichlet_left, required_ujs_dirichlet_left],
                [x_batch_neumann_bottom, required_ujs_neumann_bottom]]

    @staticmethod
    def constraining_fn(all_params, x_batch, solution):
        sd = all_params["static"]["problem"]["sd"]

        x, tanh = x_batch[:,0:1], jnp.tanh

        u = solution[:, 0:1] * tanh(x/sd)  # Hard constraining
        v = solution[:, 1:2] * tanh(x/sd) 

        return jnp.concatenate([u, v, solution[:, 2:3], solution[:, 3:4], solution[:, 4:5]], axis=1)
    
    @staticmethod
    def shape_symmetric_gradient(ux, uy, vx, vy):
        return jnp.array([[ux, 0.5*uy*vx], [0.5*uy*vx, vy]])
    
    @staticmethod
    def batched_shape_symmetric_gradient(ux_x, uy_y, ux_y, uy_x):

        # Flatten
        ux_x_flat = ux_x.ravel()
        uy_y_flat = uy_y.ravel()
        ux_y_flat = ux_y.ravel()
        uy_x_flat = uy_x.ravel()

        # Vectorize
        v_shape_symmetric_gradient = vmap(CooksProblemForwardSoft.shape_symmetric_gradient, in_axes=(0, 0, 0, 0), out_axes=0)

        # Apply the vectorized function 
        epsilon_batch = v_shape_symmetric_gradient(ux_x_flat, uy_y_flat, ux_y_flat, uy_x_flat)

        return epsilon_batch

    @staticmethod
    def loss_fn(all_params, constraints):
        #TODO : The loss setup is not correct, mean should be applied on the total equation.
        E = all_params["static"]["problem"]["E"]
        _, ux_x, uy_y, ux_y, uy_x, sigmaxx_x, sigmayy_y, sigmaxy_x, sigmaxy_y, stressxx, stressyy, stressxy= constraints[0]
        # Material Model
        epsilon_batch = CooksProblemForwardSoft.batched_shape_symmetric_gradient(ux_x, uy_y, ux_y, uy_x)
        stress_batch = jnp.einsum("ijkl, akl->aij", E, epsilon_batch)
        MM_loss = (jnp.mean((stressxx - stress_batch[:, 0, 0])**2) + # sigmaxx - sigmaxx (E \boxdot epsilon)
                        jnp.mean((stressyy - stress_batch[:, 1, 1])**2) + # sigmayy - sigmayy (E \boxdot epsilon)
                        jnp.mean((stressxy - stress_batch[:, 0, 1])**2)) # sigmaxy - sigmaxy (E \boxdot epsilon). stress_batch[:, 1, 0] == stress_batch[:, 0, 1] [symmetric stress]
        # Balance
        Balance_loss =  jnp.mean((sigmaxx_x + sigmaxy_y)**2) + jnp.mean((sigmayy_y + sigmaxy_x)**2)
        phy_loss = MM_loss + Balance_loss

        # Neumann Boundary condition [dirichlet handeled in the hard constraints] See: constraining_fn()
        # nbc right
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[1]
        unit_normal_vec_right = jnp.array([1.,0.])                      # unit normal (1,0)
        applied_traction_right = jnp.array([0.,1.])
        nbc_loss_right = (jnp.mean((sigmaxx_n*unit_normal_vec_right[0] + sigmaxy_n*unit_normal_vec_right[1] - applied_traction_right[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_right[0] + sigmayy_n*unit_normal_vec_right[1] - applied_traction_right[1])**2))
        
        # nbc top
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[2]
        unit_normal_vec_top = jnp.array([-0.3162279,  0.9486832])        # unit normal (-0.3162279,  0.9486832) [nx_top = -0.1961 ny_top = 0.9806]
        applied_traction_top = jnp.array([0.,0.])
        nbc_loss_top = (jnp.mean((sigmaxx_n*unit_normal_vec_top[0] + sigmaxy_n*unit_normal_vec_top[1] - applied_traction_top[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_top[0] + sigmayy_n*unit_normal_vec_top[1] - applied_traction_top[1])**2))
        
        # dbc left
        _, ux, uy = constraints[3]
        dbc_loss_top = jnp.mean(ux**2) + jnp.mean(uy**2)

        # nbc bottom
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[4]
        unit_normal_vec_bottom = jnp.array([ 0.67572457, -0.7371541])    # unit normal (0.67572457, -0.7371541)  [nx_bottom = 0.625 ny_bottom = -0.78125]

        applied_traction_bottom = jnp.array([0.,0.])
        nbc_loss_bottom = (jnp.mean((sigmaxx_n*unit_normal_vec_bottom[0] + sigmaxy_n*unit_normal_vec_bottom[1] - applied_traction_bottom[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_bottom[0] + sigmayy_n*unit_normal_vec_bottom[1] - applied_traction_bottom[1])**2))
        
        return 1e6*phy_loss + 1e6*nbc_loss_right + 1e6*nbc_loss_top + 1e6*dbc_loss_top + 1e6*nbc_loss_bottom
    
    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        """For now, exact solution(numerical solution) is not added, which will be added later on from the julia FEM code"""
        noise_scale = 1e-6
        key = jax.random.PRNGKey(0) 

        noise = noise_scale * jax.random.normal(key, (x_batch.shape[0],all_params["static"]["problem"]["dims"][0]))
        
        return noise
    
if __name__=="__main__":
    print(" ")