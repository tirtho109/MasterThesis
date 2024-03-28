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


class InteractionProblemForward3DSoft(Problem): # (2+1)-D problem, t:0, x:1, y:2
    """
        Interaction between virus and immune system: Ω ⊂ ℜ² 
        uₜ = ug(u) - γuv + αΔu          for  x ∈ Ω, t>0
        vₜ = j[u] - η(1-u)v + βΔv       for  x ∈ Ω, t>0
        u(0,x) = u₀(x), v(0,x) = v₀(x)  for x ∈ Ω
        0 = u ⋅ n = v ⋅ n                for x ∈ ∂Ω, t>0

        Where,
        Virus(u): 
        • ug(u) → logistic growth with Alle effect for small population.
        • - γuv → Predator-prey-reaction describing a decrease of vires in dependency 
        • αΔu  → diffusive spreading of the virus

        T-cells(v):
        • j[u] → inflow of T cells through the portal field Θ
        •   [
                - ηv → decay term
                ηuv → predator-prey frowth term for predator v
            ] → the maximal capacity of the virus u as 1. Combined term is non-positive for all 
            u and v, and describes the decay of T-cells in the absence of virus. (the T-cells 
            decal less, if virus is present)
        • βΔv → diffusive spread of T-cells.

        with, 
        g(u) = (1-u) * [(u -uₘᵢₙ) / (u + κ)]
        j[u](x) = δχΘ(x) ∫Ω u(t,x) dx
        ∫Ω χΘ(x)dx = 1, with χΘ(x) ⋝ 0 on Θ ⊂ Ω
                            χΘ(x) = 0 on Ω\Θ       [Θ is a portal field]
        
        Cases:
        • Healing infection course → uₘᵢₙ=0.05, κ=0.01, γ=0.4, α=0.6, δ=1, η=0.2, β=3
                with, IC: u₀(x)=1 & v₀(x)=0
                lim(t→∞) u(t,x) =  lim(t→∞) v(t,x) = 0
        • Chronice infection courses → uₘᵢₙ=0.05, κ=0.01, γ=0.4, α=0.6, δ=0.8, η=0.2, β=0.65
                with, IC: u₀(x)=1 & v₀(x)=0 tends to a spatially inhomogeneous stationary state
        Ref: https://www.tandfonline.com/doi/full/10.1080/13873954.2021.2020296
    """

    @staticmethod
    def init_params(u_min=0.05, kappa = 0.01, gamma=0.4,
                    alpha=0.6, delta=1, eta=0.2, beta=3,
                    u0=1, v0=0, time_limit=[0,2], sd=0.1,
                    lambda_phy=1e0, lambda_neumann=1e4,
                    lambda_ic = 1e4, t_begin=0, t_end=2,
                    nbc_points_left=(5,5),  nbc_points_right=(5,5),
                    nbc_points_top=(5,5),  nbc_points_bottom=(5,5),
                    grid=(10,10,10),):
        ic_points = (grid[1], grid[2]) # The 2d mesh we are creating for each time step
        static_params = {
            "dims":(2,3),   # Out: u, v; In: t, x, y
            "u_min":u_min,
            "kappa":kappa,
            "gamma":gamma,
            "alpha":alpha,
            "delta":delta,
            "eta":eta,
            "beta":beta,
            "time_limit":time_limit,
            "sd":sd,
            "u0":u0,
            "v0":v0,
            "lambda_phy":lambda_phy,
            "lambda_neumann":lambda_neumann,
            "lambda_ic":lambda_ic,
            "t_begin":t_begin,
            "t_end":t_end,
            "ic_points":ic_points,
            "nbc_points_left":nbc_points_left,
            "nbc_points_right":nbc_points_right,
            "nbc_points_top":nbc_points_top,
            "nbc_points_bottom":nbc_points_bottom,
            "time_keys":jnp.linspace(t_begin, t_end, grid[0]),
            "U0":jnp.array([u0*grid[1]*grid[2]]),
            "grid":grid
        }
        trainable_params={
            "usums_rest":jnp.array([200] * (grid[0] - 1), dtype=jnp.float32) # we are gonna train the usums as well for j[u]
        }

        return static_params, trainable_params
    
    @staticmethod 
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        # out: u = 0, v = 1
        # in: t = 0, x = 1, y = 2
        x_batch_phy = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (   # we need u; v; u_t; v_t; u_xx; v_xx; u_yy; v_yy
            (0, ()),            # u    #    (out_idx, (in_idx, in_idx)) 
            (1, ()),            # v
            (0, (0,)),          # u_t
            (1, (0,)),          # v_t
            (0, (1,1)),         # u_xx
            (1, (1,1)),         # v_xx
            (0, (2,2)),         # u_yy
            (1, (2,2)),         # v_yy
        )

        # Neumann Part
        nbc_point_left = all_params["static"]["problem"]["nbc_points_left"]
        nbc_points_right = all_params["static"]["problem"]["nbc_points_right"]
        nbc_points_top = all_params["static"]["problem"]["nbc_points_top"]
        nbc_points_bottom = all_params["static"]["problem"]["nbc_points_bottom"]
        ic_points = all_params["static"]["problem"]["ic_points"]
        ##################################################################################################################################################
        # Follow the sequence for domain.sample_boundaries()                                                                                             #
        # (t=0, x,y), (t=end, x, y), (t, 0, y), (t, x=end, y), (t, x, y=0), (t, x, y=end)                                                                #
        # (time_bottom(initial_conditions), time_top, [Consider 2D plane x-y])--> (nbc_points_left, nbc_points_right, nbc_points_bottom, nbc_points_top) #
        # we need the follwing derivatives for neumann bc: u_x, v_x, u_y, v_y                                                                            #
        #          t                                                                                                                                     #
        #          |                                                                                                                                     #
        #          |                                                                                                                                     #
        #          |   / y                                                                                                                               #
        #          |  /                                                                                                                                  #
        #          | /                                                                                                                                   #
        #          |/_________ x                                                                                                                         #
        #         (0,0,0)                                                                                                                                #
        ##################################################################################################################################################
        batch_shapes = (ic_points, (0,0), nbc_point_left, nbc_points_right, nbc_points_bottom, nbc_points_top)
        x_batch_ic, _, x_batch_neumann_left, x_batch_neumann_right, x_batch_neumann_bottom, x_batch_neumann_top = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

        ################# IC Bottom ################
        required_ujs_initial_condition = (
            (0, ()),            # u 
            (1, ()),            # v
        )

        ################# NBC right ################
        required_ujs_neumann_right = (
            (0, (1,)),          # u_x
            (1, (1,)),          # v_x
            (0, (2,)),          # u_y
            (1, (2,)),          # v_y
        )

        ################# NBC left ################
        required_ujs_neumann_left = (
            (0, (1,)),          # u_x
            (1, (1,)),          # v_x
            (0, (2,)),          # u_y
            (1, (2,)),          # v_y
        )

        ################# NBC bottom ################
        required_ujs_neumann_bottom = (
            (0, (1,)),          # u_x
            (1, (1,)),          # v_x
            (0, (2,)),          # u_y
            (1, (2,)),          # v_y
        )

        ################# NBC top ################
        required_ujs_neumann_top = (
            (0, (1,)),          # u_x
            (1, (1,)),          # v_x
            (0, (2,)),          # u_y
            (1, (2,)),          # v_y
        )

        ############## usums loss #################
        grid_batch = all_params["static"]["problem"]["grid"]
        x_grid_batch = domain.sample_interior(all_params, key, sampler, grid_batch)
        required_ujs_gird = (
            (0, ()),            # u 
        )
        
        return [[x_batch_phy, required_ujs_phys],
                [x_batch_ic, required_ujs_initial_condition],
                [x_batch_neumann_right, required_ujs_neumann_right],
                [x_batch_neumann_left, required_ujs_neumann_left],
                [x_batch_neumann_bottom, required_ujs_neumann_bottom],
                [x_batch_neumann_top, required_ujs_neumann_top],
                [x_grid_batch, required_ujs_gird]]
    
    @staticmethod
    def loss_fn(all_params, constraints):

        u0 = all_params["static"]["problem"]["u0"]
        v0 = all_params["static"]["problem"]["v0"]
        lambda_phy = all_params["static"]["problem"]["lambda_phy"]
        lambda_neumann = all_params["static"]["problem"]["lambda_neumann"]
        lambda_ic = all_params["static"]["problem"]["lambda_ic"]

        # Model Params: kappa
        u_min = all_params["static"]["problem"]["u_min"]
        kappa = all_params["static"]["problem"]["kappa"]
        gamma = all_params["static"]["problem"]["gamma"] 
        alpha = all_params["static"]["problem"]["alpha"] 
        delta = all_params["static"]["problem"]["delta"] 
        eta = all_params["static"]["problem"]["eta"] 
        beta = all_params["static"]["problem"]["beta"]  

        lambda_phy = all_params["static"]["problem"]["lambda_phy"] 
        lambda_ic = all_params["static"]["problem"]["lambda_ic"] 
        lambda_neumann = all_params["static"]["problem"]["lambda_neumann"] 

        time_keys = all_params["static"]["problem"]["time_keys"]
        U0 = all_params["static"]["problem"]["U0"]  # sum of virus at t=0
        usums_rest = all_params["trainable"]["problem"]["usums_rest"]
        usums = jnp.concatenate([U0, usums_rest])
        # PDE loss
        """
            uₜ - ug(u) - γuv + αΔu = 0         for  x ∈ Ω, t>0
            vₜ - j[u] - η(1-u)v + βΔv = 0      for  x ∈ Ω, t>0
                                    &
            g(u) = (1-u) * [(u -uₘᵢₙ) / (u + κ)]
            j[u](x) = δχΘ(x) ∫Ω u(t,x) dx
            ∫Ω χΘ(x)dx = 1, with χΘ(x) ⋝ 0 on Θ ⊂ Ω
                                χΘ(x) = 0 on Ω\Θ       [Θ is a portal field]
            The portal Θ is modelled as a rectangular with the size of 0.14 * 0.14
            in the corner around (x1, x2) = (1, 1)
            #TODO: Check the funciton j[u] with Cordula.
        """
        x_batch, u, v, u_t, v_t, u_xx, v_xx, u_yy, v_yy = constraints[0]

        # Set up ju & gu
        print(u.shape)
        print(x_batch.shape)
        # unique_times = jnp.unique(x_batch[:, 0])                    # unique time in batch
        # indices = jnp.isin(time_keys, unique_times).nonzero()[0]    # common time indices of batch in time keys
        # print(indices)
        # ju = jnp.zeros(x_batch.shape[0])    # initialize ju
        # for idx in indices:
        #     current_time = time_keys[idx]
        #     usum = usums[idx] * delta
        #     condition_time = x_batch[:,0] == current_time
        #     condition_x1 = (0.86 < x_batch[:,1]) & (x_batch[:,1] <= 1)
        #     condition_x2 = (0.86 < x_batch[:,2]) & (x_batch[:,2] <= 1)
        #     combined_condition = condition_time & condition_x1 & condition_x2
        #     locations = jnp.where(combined_condition)[0]
        #     ju = ju.at[locations].set(usum)
        ju = jnp.zeros(x_batch.shape[0]) 
        for time,usum in zip(time_keys, usums):
            condition_time = x_batch[:,0] == time
            condition_x1 = (0.86 < x_batch[:,1]) & (x_batch[:,1] <= 1)
            condition_x2 = (0.86 < x_batch[:,2]) & (x_batch[:,2] <= 1)
            combined_condition = condition_time & condition_x1 & condition_x2
            updates = combined_condition * usum * delta
            ju += updates

        gu = (1-u) * ((u - u_min) / (u + kappa))

        pde_u_loss = jnp.mean((u_t - u*gu + gamma*u*v - alpha*(u_xx+u_yy))**2)
        pde_v_loss = jnp.mean((v_t - ju + eta*(1-u)*v - beta*(v_xx+v_yy))**2)

        pde_loss = lambda_phy*(pde_u_loss + pde_v_loss)

        # IC loss
        _, u, v = constraints[1]
        ic_loss = lambda_ic*(jnp.mean((u-1)**2) + jnp.mean((v**2)))
        # NBC right
        _, u_x, v_x, u_y, v_y = constraints[2]
        n_right = jnp.array([1., 0.])
        nbc_right_loss = lambda_neumann * (jnp.mean((u_x*n_right[0] + u_y*n_right[1])**2) + 
                                           jnp.mean((v_x*n_right[0] + v_y*n_right[1])**2))
        # NBC left
        _, u_x, v_x, u_y, v_y = constraints[3]
        n_left = jnp.array([-1., 0.])
        nbc_left_loss = lambda_neumann * (jnp.mean((u_x*n_left[0] + u_y*n_left[1])**2) + 
                                           jnp.mean((v_x*n_left[0] + v_y*n_left[1])**2))
        # NBC bottom
        _, u_x, v_x, u_y, v_y = constraints[4]
        n_bottom = jnp.array([0., -1.])
        nbc_bottom_loss = lambda_neumann * (jnp.mean((u_x*n_bottom[0] + u_y*n_bottom[1])**2) + 
                                           jnp.mean((v_x*n_bottom[0] + v_y*n_bottom[1])**2))
        # NBC top
        _, u_x, v_x, u_y, v_y = constraints[5]
        n_top = jnp.array([0., 1.])
        nbc_top_loss = lambda_neumann * (jnp.mean((u_x*n_top[0] + u_y*n_top[1])**2) + 
                                           jnp.mean((v_x*n_top[0] + v_y*n_top[1])**2))
        
        nbc_total_loss = nbc_right_loss + nbc_left_loss + nbc_bottom_loss + nbc_top_loss

        # usum loss in eq(2) for v'-> j[u]
        x_batch_usum, u = constraints[6]
        losses = []
        for time,usum in zip(time_keys[1:], usums_rest):
            condition_time = x_batch_usum[:,0] == time
            usum_out = jnp.sum(condition_time * u)
            squared_diff = (usum_out - usum) ** 2
            losses.append(squared_diff)

        usum_loss = 1e4*jnp.mean(jnp.array(losses))
        return pde_loss + ic_loss + nbc_total_loss + usum_loss
    
    @staticmethod 
    def exact_solution(all_params, x_batch, batch_shape=None):
        """For now, exact solution(numerical solution) is not added, which will be added later on from the julia FEM code""" #TODO
        noise_scale = 1e-6
        key = jax.random.PRNGKey(0) 

        noise = noise_scale * jax.random.normal(key, (x_batch.shape[0],all_params["static"]["problem"]["dims"][0]))
        
        return noise
        