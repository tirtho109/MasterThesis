import os 
import sys
import time

paths_to_add = [
    os.path.abspath(os.path.join('../..')),  
    os.path.abspath(os.path.join('..'))  
]

sys.path.extend(path for path in paths_to_add if path not in sys.path)

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import scipy.stats

from fbpinns import networks
from fbpinns.domains import Domain

class CooksDomainND(Domain):
    """
                            -*(0.048, 0.060)
                        --  |  
                    --      |   
                --          |
    (0,0.044) *-            -* (0.048, 0.044)
        |                 --             
        |              --         
        |           --        
        |         --      
        |      --      
        |   --           
    (0,0) *-

    """
    @staticmethod
    def init_params(corners):
        xd = corners.shape[1]
        static_params = {
            "xd" : xd,
            "corners" : jnp.array(corners)
        }
        return static_params, {}
    @staticmethod
    def phi(xi, eta):
        phiA = 0.25 * (1 - xi) * (1 - eta)
        phiB = 0.25 * (1 + xi) * (1 - eta)
        phiC = 0.25 * (1 + xi) * (1 + eta)
        phiD = 0.25 * (1 - xi) * (1 + eta)
        return jnp.array([phiA, phiB, phiC, phiD])
    @staticmethod
    def transformation_matrix(xi, eta, x_hat):
        phi_values = CooksDomainND.phi(xi, eta)
        
        t_mat = jnp.zeros((2, 8))
        t_mat = t_mat.at[0, :4].set(phi_values)
        t_mat = t_mat.at[1, 4:].set(phi_values)
        transformed = jnp.dot(t_mat, x_hat)
        return transformed
    
    @staticmethod
    def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin,xmax,b in zip(xmin, xmax, batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)# (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1,-1)), xmax.reshape((1,-1))
            x_batch = xmin + (xmax - xmin)*s

        return jnp.array(x_batch)
    
    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):
        # xd = all_params["static"]["domain"]["xd"]
        corners = all_params["static"]["domain"]["corners"]
        x_hat = jnp.concatenate([corners[:, 0], corners[:, 1]])
        # isoparametric element
        xmin, xmax = jnp.array([-1,-1]), jnp.array([1,1])
        x_batch_iso =  CooksDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)
        return vmap(CooksDomainND.transformation_matrix, in_axes=(0, 0, None))(x_batch_iso[:, 0], x_batch_iso[:, 1], x_hat)
    
    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        xmin, xmax = jnp.array([-1,-1]), jnp.array([1,1]) #isoparametric_elememnt

        xd = all_params["static"]["domain"]["xd"]
        corners = all_params["static"]["domain"]["corners"]
        x_hat = jnp.concatenate([corners[:, 0], corners[:, 1]])

        assert len(batch_shapes) == 2*xd# total number of boundaries

        x_batches = []
        for i in range(xd):
            ic = jnp.array(list(range(i))+list(range(i+1,xd)), dtype=int)
            for j,v in enumerate([xmin[i], xmax[i]]):
                batch_shape = batch_shapes[2*i+j]
                if len(ic):
                    xmin_, xmax_ = xmin[ic], xmax[ic]
                    key, subkey = jax.random.split(key)
                    x_batch_ = CooksDomainND._rectangle_samplerND(subkey, sampler, xmin_, xmax_, batch_shape)# (n, xd-1)
                    x_batch = v*jnp.ones((jnp.prod(jnp.array(batch_shape)),xd), dtype=float)
                    x_batch = x_batch.at[:,ic].set(x_batch_)
                else:
                    assert len(batch_shape) == 1
                    x_batch = v*jnp.ones(batch_shape+(1,), dtype=float)
                x_batch = vmap(CooksDomainND.transformation_matrix, in_axes=(0, 0, None))(x_batch[:, 0], x_batch[:, 1], x_hat)
                x_batches.append(x_batch)
        return x_batches
    
    #TODO check norm_fn with Alexander
    @staticmethod 
    def norm_fn(all_params, x):
        corners = all_params["static"]["domain"]["corners"]
        xmin, xmax = jnp.array([-1.,-1.]), jnp.array([1.,1,])
        #xmin, xmax = jnp.array([0.,0.]), jnp.array([0.048,0.06]) #isoparametric_elememnt
        # xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        x_hat = jnp.concatenate([corners[:, 0], corners[:, 1]])
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        x = networks.norm(mu, sd, x)
        return x
    

if __name__=="__main__":
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)

    corners = jnp.array([(0., 0.), (0, 0.044), (0.048, 0.060), (0.048, 0.044)]) 

    domain = CooksDomainND
    sampler = "halton"

    # Example 2D Plot
    key = jax.random.PRNGKey(0)
    all_params = {
        "static": {
            "domain": {
                "xd": 2,
                "corners": corners,
            }
        }
    }

    batch_shape = (10,20)  # for example
    x_batch = domain.sample_interior(all_params, key, 'uniform', batch_shape)

    batch_shapes = ((20,),(15,),(10,),(10,)) # bottom, top, left, right
    x_boundaries = domain.sample_boundaries(all_params, key, 'grid', batch_shapes)

    plt.scatter(x_batch[:, 0:1], x_batch[:, 1:2])
    for x_batch_b in x_boundaries:
            plt.scatter(x_batch_b[:,0], x_batch_b[:,1], )

    plt.show()