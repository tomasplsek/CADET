import numpy as np
import jax.numpy as jnp
from tensorflow import convert_to_tensor

from beta_model import run_vect

shape_image = (128, 128, 1)

################## DATA READING FUNCTION ##################
 
def read_data(x0, batch, suf, params):
    p = params.loc[x0:x0+batch-1]

    # 50% of the data has no cavities
    if "50" in suf:
        true = np.array([False for i in range(batch)])
        true[::2] = True
        p["R1"][true] = 0
        p["R2"][true] = 0
        p["rim_height"][true] = 0

    # 90% of the data has no cavities
    if "90" in suf:
        true = np.array([False for i in range(batch)])
        true[::10] = True
        p["R1"][true] = 0
        p["R2"][true] = 0
        p["rim_height"][true] = 0

    # turn-off cavity rims
    if "norims" in suf:
        p["rim_height"] = 0

    true = np.array([True for i in range(batch)])
    true[::5] = False
    p["rim_height"][true] = 0

    # turn-off sloshing
    if "nosloshing" in suf:
        p["s_depth"] = 0

    # turn-off changing angle theta
    if "notheta" in suf:
        p["theta1"] = 0
        p["theta2"] = 0

    parnames = ["dx", "dy", "dx_2", "dy_2", "phi", "phi_2",
                "A", "r0", "beta", "ellip", "A_2", "r0_2", "beta_2", "bkg", 
                "s_depth", "s_period", "s_dir", "s_angle",
                "r1", "r2", "varphi1", "varphi2", "theta1", "theta2", 
                "R1", "R2", "e1", "e2", "phi1", "phi2",
                "rim_size", "rim_height", "rim_type"]

    # transform arrays into JAX stacks
    for par in parnames:
        globals()[par] = jnp.stack(list(p[par]))
    nums = jnp.stack([i for i in range(x0,x0+batch)])

    X, y, v = run_vect(nums,
                       dx, dy, dx_2, dy_2, phi, phi_2,
                       A, r0, beta, ellip, A_2, r0_2, beta_2, bkg, 
                       s_depth, s_period, s_dir, s_angle,
                       r1, r2, varphi1, varphi2, theta1, theta2, 
                       R1, R2, e1, e2, phi1, phi2,
                       rim_size, rim_height, rim_type)

    # choose weighting
    if "logweights" in suf:
        w = 1 / jnp.log10(jnp.sum(X, axis=(1,2)))
    elif "sqrtweights" in suf:
        w = 1 / jnp.sqrt(jnp.sum(X, axis=(1,2)))
    elif "alphaweights" in suf:
        w = jnp.array(p["beta"])
    else:
        w = jnp.ones(batch)
    
    # normalize weights
    if "weights" in suf:
        w = batch / sum(w) * w

    # scale images by logarithm
    X = jnp.log10(X+1) #/ jnp.max(jnp.log10(X+1), axis=(1,2)).reshape((batch,1,1))

    X = convert_to_tensor(X.reshape(batch, *shape_image))
    y = convert_to_tensor(y.reshape(batch, *shape_image))
    w = convert_to_tensor(w.reshape(batch, 1))

    return X, y, w

# image generator
def img_generator(batch = 8, suf="", params=[]):
    x0 = 0

    while True:
        X, y, w = read_data(x0, batch, suf, params=params)
        x0 += batch
        if x0 >= params.shape[0]- 13000: x0 = 0
        yield (X, y, w)