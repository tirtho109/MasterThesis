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

"""
    2D-linear elastisity - plain strain condition epsilon_zz==0
    Material Model:
    σ = Ε ⊡ ϵ
    σᵢⱼ = Εᵢⱼₖₗ ϵₖₗ                   # [2OT = 4OT ⊡ 2OT]
    with, ϵˢʸᵐ = [u,x               0.5*(u,y + v,x) ; 
                0.5*(u,y + v,x)              v,y]
    Ends up with 3 equations for the loss term, as ϵ and σ are symmetric (σ₁₂ == σ₂₁).
    (1) σ₁₁ = Ε₁₁ₖₗ ϵₖₗ 
    (2) σ₁₂ = σ₂₁ = Ε₁₂ₖₗ ϵₖₗ 
    (3) σ₂₂ = Ε₂₂ₖₗ ϵₖₗ 

    Balance:
    σ ⋅ ∇ = 0                       # [2OT ⋅ 1OT = 1OT]
    σᵢⱼ,ⱼ = 0ᵢ
    Ends up in 2 equations:
    (1) σ₁₁,₁ + σ₁₂,₂ = 0
    (2) σ₂₁,₁ + σ₂₂,₂ = 0           # Note: σ₁₂ == σ₂₁ (symmetric)

    Neumann BC: Applies on top, right & bottom boundaries of Cook's Membrane Problem
    The following applies to all different part of the Neumann Boundaries. 
    σ ⋅ n = t                       # [2OT ⋅ 1OT = 1OT]
    σᵢⱼnⱼ = tᵢ
    Ends up in 2 equations:
    (1) σ₁₁n₁ + σ₁₂n₂ = t₁
    (2) σ₁₂n₁ + σ₂₂n₂ = t₂          # Note: σ₁₂ == σ₂₁ (symmetric)

    Dirichlet BC: Applies on the left Boundary of Cook's Problem
    u = 0              # 1OT
    uᵢ= 0
    Ends up in 2 equations :
    u₁=0
    u₂=0
"""

class CooksProblemForwardSoft(Problem):
    """ Linear Elasticity, plain strain, DBC at left, NBC at right, 0 traction at the bottom and top"""
    @staticmethod
    def init_params(lambda_true= 4, mu_true = 5, 
                    nbc_points_right = 10, nbc_points_top=10, 
                    nbc_points_bottom=10, dbc_points_left=10, sd=0.1):
        
        E = lambda_true * jnp.einsum("ij, kl->ijkl", jnp.eye(2), jnp.eye(2)) + mu_true *(
                jnp.einsum("ik, jl->ijkl", jnp.eye(2), jnp.eye(2)) + jnp.einsum("il, jk->ijkl", jnp.eye(2), jnp.eye(2))
            )
        static_params = {
            "dims":(5,2), # Out: ux, uy,sigma_xx, sigma_yy, sigma_xy[symmetric], in: (x,y)
            "E":E,
            "nbc_points_right":nbc_points_right,
            "nbc_points_top":nbc_points_top,
            "nbc_points_bottom":nbc_points_bottom,
            "dbc_points_left":dbc_points_left,
            "sd":sd,
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (   # we need: ux,x; uy,y; ux,y; uy,x; sigma_xx,x; sigma_yy,y; sigma_xy,x; sigma_xy,y; sigma_xx; sigma_yy; sigma_xy
            (0, (0,)),      # ux,x (out_idx, (in_idx, in_idx))
            (1, (1,)),      # uy,y
            (0, (1,)),      # ux, y
            (1, (0,)),      # uy, x
            (2, (0,)),      # sigma_xx,x
            (3, (1,)),      # sigma_yy,y;
            (4, (0,)),      # sigma_xy,x; 
            (4, (1,)),      # sigma_xy,y
            (2,()),         # sigma_xx
            (3,()),         # sigma_yy
            (4,()),         # sigma_xy or sigma_yx
        )

        nbc_points_right = all_params["static"]["problem"]["nbc_points_right"]
        nbc_points_top = all_params["static"]["problem"]["nbc_points_top"]
        nbc_points_bottom = all_params["static"]["problem"]["nbc_points_bottom"]
        dbc_points_left = all_params["static"]["problem"]["dbc_points_left"]
        batch_shapes = ((nbc_points_bottom,),(nbc_points_top,),(dbc_points_left,),(nbc_points_right,)) # bottom, top, left, right
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
    def shape_symmetric_gradient(ux, uy, vx, vy):
        return jnp.array([[ux, 0.5*(uy + vx)], [0.5*(uy + vx), vy]])

    
    @staticmethod
    def batched_shape_symmetric_gradient(ux_x, ux_y, uy_x, uy_y):

        # Flatten
        ux_x_flat = ux_x.ravel()
        ux_y_flat = ux_y.ravel()
        uy_y_flat = uy_y.ravel()
        uy_x_flat = uy_x.ravel()

        # Vectorize
        v_shape_symmetric_gradient = vmap(CooksProblemForwardSoft.shape_symmetric_gradient, in_axes=(0, 0, 0, 0), out_axes=0)

        # Apply the vectorized function 
        epsilon_batch = v_shape_symmetric_gradient(ux_x_flat, ux_y_flat, uy_x_flat, uy_y_flat)

        return epsilon_batch

    @staticmethod
    def loss_fn(all_params, constraints):
        #TODO : Recheck the loss terms 
        E = all_params["static"]["problem"]["E"]
        _, ux_x, uy_y, ux_y, uy_x, sigmaxx_x, sigmayy_y, sigmaxy_x, sigmaxy_y, stressxx, stressyy, stressxy= constraints[0]
        # Material Model
        """
        MM Loss:
        (1) σ₁₁ - Ε₁₁ₖₗ ϵₖₗ = 0
        (2) σ₁₂ - σ₂₁ = Ε₁₂ₖₗ ϵₖₗ = 0
        (3) σ₂₂ - Ε₂₂ₖₗ ϵₖₗ  = 0
        """
        epsilon_batch = CooksProblemForwardSoft.batched_shape_symmetric_gradient(ux_x, ux_y, uy_x, uy_y)
        stress_batch = jnp.einsum("ijkl, akl->aij", E, epsilon_batch)
        MM_loss = (jnp.mean((stressxx - stress_batch[:, 0, 0])**2) + # sigmaxx - sigmaxx (E \boxdot epsilon)
                        jnp.mean((stressyy - stress_batch[:, 1, 1])**2) + # sigmayy - sigmayy (E \boxdot epsilon)
                        jnp.mean((stressxy - stress_batch[:, 0, 1])**2)) # sigmaxy - sigmaxy (E \boxdot epsilon). stress_batch[:, 1, 0] == stress_batch[:, 0, 1] [symmetric stress]
        
        # Balance
        """
        Balance Loss
        (1) σ₁₁,₁ + σ₁₂,₂ = 0
        (2) σ₂₁,₁ + σ₂₂,₂ = 0           # Note: σ₁₂ == σ₂₁ (symmetric)
        """
        Balance_loss =  jnp.mean((sigmaxx_x + sigmaxy_y)**2) + jnp.mean((sigmayy_y + sigmaxy_x)**2)
        phy_loss = MM_loss + Balance_loss

        # Neumann Boundary condition [dirichlet handeled in the hard constraints] See: constraining_fn()
        """
        NBC Loss
        (1) σ₁₁n₁ + σ₁₂n₂ - t₁ = 0
        (2) σ₁₂n₁ + σ₂₂n₂ - t₂ = 0      # Note: σ₁₂ == σ₂₁ (symmetric)
        """
        # nbc right
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[1]
        unit_normal_vec_right = jnp.array([1.,0.])                      # unit normal (1,0)
        applied_traction_right = jnp.array([0.,1.])
        nbc_loss_right = (jnp.mean((sigmaxx_n*unit_normal_vec_right[0] + sigmaxy_n*unit_normal_vec_right[1] - applied_traction_right[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_right[0] + sigmayy_n*unit_normal_vec_right[1] - applied_traction_right[1])**2))
        
        # nbc top
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[2]
        unit_normal_vec_top = jnp.array([-0.3162279,  0.9486832])        # unit normal (-0.3162279,  0.9486832) 
        applied_traction_top = jnp.array([0.,0.])
        nbc_loss_top = (jnp.mean((sigmaxx_n*unit_normal_vec_top[0] + sigmaxy_n*unit_normal_vec_top[1] - applied_traction_top[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_top[0] + sigmayy_n*unit_normal_vec_top[1] - applied_traction_top[1])**2))
        
        # dbc left
        _, ux, uy = constraints[3]
        dbc_loss_top = jnp.mean(ux**2) + jnp.mean(uy**2) 

        # nbc bottom
        _, sigmaxx_n, sigmayy_n, sigmaxy_n = constraints[4]
        unit_normal_vec_bottom = jnp.array([ 0.67572457, -0.7371541])    # unit normal (0.67572457, -0.7371541) 

        applied_traction_bottom = jnp.array([0.,0.])
        nbc_loss_bottom = (jnp.mean((sigmaxx_n*unit_normal_vec_bottom[0] + sigmaxy_n*unit_normal_vec_bottom[1] - applied_traction_bottom[0])**2) +
                    jnp.mean((sigmaxy_n*unit_normal_vec_bottom[0] + sigmayy_n*unit_normal_vec_bottom[1] - applied_traction_bottom[1])**2))
        
        return phy_loss + nbc_loss_right + nbc_loss_top + dbc_loss_top + nbc_loss_bottom
    
    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        """For now, exact solution(numerical solution) is not added, which will be added later on from the julia FEM code""" #TODO
        noise_scale = 1e-6
        key = jax.random.PRNGKey(0) 

        noise = noise_scale * jax.random.normal(key, (x_batch.shape[0],all_params["static"]["problem"]["dims"][0]))
        
        return noise
    



#################################################################################
############################ Hard Boundary Condition ############################
#################################################################################




    
# We will apply Hard Boundary condition:
class CooksProblemForwardHard(Problem):
    """ Linear Elasticity, plain strain, DBC at left, NBC at right"""
    @staticmethod
    def init_params(lambda_true= 4, mu_true = 5, 
                    nbc_points_right = 10, nbc_points_top=10, 
                    nbc_points_bottom=10, dbc_points_left=10, sd=0.1):
        
        E = mu_true*(3*lambda_true+2*mu_true) /(lambda_true+mu_true)    # Young's modulus
        nu = lambda_true /(2*(lambda_true + mu_true))                   # Poisson's ratio
        static_params = {
            "dims":(5,2), # Out: ux, uy,sigma_xx, sigma_yy, sigma_xy[symmetric], in: (x,y)
            "E":E,
            "nu":nu,
            "nbc_points_right":nbc_points_right,
            "nbc_points_top":nbc_points_top,
            "nbc_points_bottom":nbc_points_bottom,
            "dbc_points_left":dbc_points_left,
            "sd":sd,
            "lambda_true":lambda_true,
            "mu_true":mu_true,
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # Physics Loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])

        nbc_points_right = all_params["static"]["problem"]["nbc_points_right"]
        nbc_points_top = all_params["static"]["problem"]["nbc_points_top"]
        nbc_points_bottom = all_params["static"]["problem"]["nbc_points_bottom"]
        dbc_points_left = all_params["static"]["problem"]["dbc_points_left"]
        batch_shapes = ((nbc_points_bottom,),(nbc_points_top,),(dbc_points_left,),(nbc_points_right,)) # bottom, top, left, right

        x_boundaries = domain.sample_boundaries(all_params, key, sampler, batch_shapes)
        x_batch = jnp.vstack([x_batch_phys] + x_boundaries)
        assert x_batch.shape[0] == x_batch_phys.shape[0]+nbc_points_bottom+nbc_points_right+nbc_points_top+dbc_points_left
        assert x_batch.shape[1] == 2
        required_ujs_phys = (   # we need: ux,x; uy,y; ux,y; uy,x; sigma_xx,x; sigma_yy,y; sigma_xy,x; sigma_xy,y; sigma_xx; sigma_yy; sigma_xy
            (0, (0,)),      # ux,x (out_idx, (in_idx, in_idx))
            (1, (1,)),      # uy,y
            (0, (1,)),      # ux, y
            (1, (0,)),      # uy, x
            (2, (0,)),      # sigma_xx,x
            (3, (1,)),      # sigma_yy,y;
            (4, (0,)),      # sigma_xy,x; 
            (4, (1,)),      # sigma_xy,y
            (2,()),         # sigma_xx
            (3,()),         # sigma_yy
            (4,()),         # sigma_xy or sigma_yx
        )

        return [[x_batch, required_ujs_phys]]
    
    @staticmethod
    def constraining_fn(all_params, x_batch, solution):
        """
            u_pinn^tilda(x) = G(x) + D(x) * u_pinn(x)
            sigma_pinn^tilda(x) = G_sigma + D_sigma(x) * sigma_pinn(x)
        """
        sd = all_params["static"]["problem"]["sd"]
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jnp.tanh

        sigmaxx, sigmayy, sigmaxy = solution[:,2:3], solution[:, 3:4], solution[:,4:5]

        # Hard BC on displacement
        u = solution[:, 0:1] * x + 0.0 # Hard constraining (DBC, ux=0, at x=0) 
        v = solution[:, 1:2] * x + 0.0 # DBC, uy=0, at x=0

        # Hard BC on stress
        tolerence = 1e-8
        G_sigmaxx = 0.0         # Boundary Extension
        D_sigmaxx = 0.048 - x   # Distance Function
        sigmaxx = G_sigmaxx +  D_sigmaxx * sigmaxx
        G_sigmaxy = 1   # [traction at the right boundary = [0,1]]
        D_sigmaxy = 0.048 - x 
        sigmaxy = G_sigmaxy + D_sigmaxy * sigmaxy


        # top NBC: Traction=[0,0]
        n_top = jnp.array([-0.3162279,  0.9486832])  
        m_top = (0.06 -0.044) / (0.048 -0)
        b_top = 0.044                       # Create the line
        y_on_the_top = (jnp.abs(y-(m_top * x + b_top)) < tolerence) & (x>0) & (x<0.048)
        updated_sigmaxx = -sigmaxy * (n_top[1]/n_top[0])
        updated_sigmaxy = -sigmayy * (n_top[1]/n_top[0])
        sigmaxx = jnp.where(y_on_the_top, updated_sigmaxx, sigmaxx)
        sigmaxy = jnp.where(y_on_the_top, updated_sigmaxy, sigmaxy)

        # bottom NBC: Traction=[0,0]
        n_bottom = jnp.array([ 0.67572457, -0.7371541]) 
        m_bottom = (0.044 - 0) / (0.048 -0)
        b_bottom = 0.0
        y_on_the_bottom = (jnp.abs(y-(m_bottom * x + b_bottom)) < tolerence) & (x>0) & (x<0.048)
        updated_sigmaxx = -sigmaxy * (n_bottom[1]/n_bottom[0])
        updated_sigmaxy = -sigmayy * (n_bottom[1]/n_bottom[0])
        sigmaxx = jnp.where(y_on_the_bottom, updated_sigmaxx, sigmaxx)
        sigmaxy = jnp.where(y_on_the_bottom, updated_sigmaxy, sigmaxy)

        return jnp.concatenate([u, v, sigmaxx, sigmayy, sigmaxy], axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        #TODO : Recheck the loss terms 
        E = all_params["static"]["problem"]["E"]
        nu = all_params["static"]["problem"]["nu"]
        
        _, ux_x, uy_y, ux_y, uy_x, sigmaxx_x, sigmayy_y, sigmaxy_x, sigmaxy_y, sigmaxx, sigmayy, sigmaxy= constraints[0]
        # Material Model
        """
        MM Loss:
        (1) σ₁₁ - Ε₁₁ₖₗ ϵₖₗ = 0
        (2) σ₁₂ - σ₂₁ = Ε₁₂ₖₗ ϵₖₗ = 0
        (3) σ₂₂ - Ε₂₂ₖₗ ϵₖₗ  = 0
        Alternative:
        σ_xx = (E /((1+ν)(1-2ν)) * [(1-ν)ϵ_xx + νϵ_yy]
        σ_yy = (E /((1+ν)(1-2ν)) * [νϵ_xx + (1-ν)ϵ_yy]
        σ_xy = (E / 2(1+ν)) * ϵ_xy  #TODO check 2*(1+ν) or (1+ν)???
        """
        epsilon_x = ux_x
        epsilon_y = uy_y
        epsilon_xy = 0.5*(uy_x + ux_y)
        MMx = (E / ( (1+nu) * (1-2*nu) )) * ( (1-nu) * epsilon_x + nu * epsilon_y )
        MMy = (E / ( (1+nu) * (1-2*nu) )) * ( nu*epsilon_x + (1-nu) * epsilon_y )
        MMxy = (E / (2*(1+nu)) ) *  epsilon_xy    #TODO check multiplicity of 2*(1-nu)
        MM_loss = (jnp.mean((sigmaxx - MMx)**2) + jnp.mean((sigmayy - MMy)**2) + jnp.mean((sigmaxy - MMxy)**2))
        # Balance
        """
        Balance Loss
        (1) σ₁₁,₁ + σ₁₂,₂ = 0
        (2) σ₂₁,₁ + σ₂₂,₂ = 0           # Note: σ₁₂ == σ₂₁ (symmetric)
        """
        Balance_loss =  jnp.mean((sigmaxx_x + sigmaxy_y)**2) + jnp.mean((sigmayy_y + sigmaxy_x)**2)
        phy_loss = MM_loss + Balance_loss

        return phy_loss

    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        """For now, exact solution(numerical solution) is not added, which will be added later on from the julia FEM code""" #TODO
        noise_scale = 1e-6
        key = jax.random.PRNGKey(0) 

        noise = noise_scale * jax.random.normal(key, (x_batch.shape[0],all_params["static"]["problem"]["dims"][0]))
        
        return noise

    
if __name__=="__main__":
    print(" ")