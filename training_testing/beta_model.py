from jax import device_put
from jax import random
from jax import jit, vmap
from jax.numpy import asarray, stack, arctan, array, arange, cos, deg2rad, dot, exp, float32, int16, floor, linspace, log10, min, max, meshgrid, ones, pi, save, sqrt, sin, sum, where, zeros

t = float32
shape = (128,128,128)

R = arange(128) - 128 / 2 + 0.5
points = meshgrid(R, R, R)
points = array(points, dtype=t).reshape(3, -1)[::-1]
points = device_put(points)


@jit
def create_rotation_matrix(a, b, c):
    rot_x = array([[1, 0, 0],
                  [0, cos(a), -sin(a)],
                  [0, sin(a), cos(a)]], dtype=t)

    rot_y = array([[cos(b), 0, sin(b)],
                  [0, 1, 0],
                  [-sin(b), 0, cos(b)]], dtype=t)

    rot_z = array([[cos(c), -sin(c), 0],
                  [sin(c), cos(c), 0],
                  [0, 0, 1]], dtype=t)

    return dot(rot_z, dot(rot_y, rot_x))

@jit
def create_ellipsoidal_grid(center, radii, angle):
    rot = create_rotation_matrix(*angle)

    rotated = dot(rot, points).T

    center = array(center) - 0.5 * array(shape[::-1], dtype=t)
    center = dot(rot, center)

    dR = (rotated - center)**2 / radii**2
    nR = sum(dR, axis=1).reshape(shape)

    return sqrt(nR)

@jit
def create_ellipsoid(center, radii, angle):
    r = create_ellipsoidal_grid(center, radii, angle)
    return where(r <= 1, 1, 0)

@jit
def create_ellipsoidal_shell(center, radii1, radii2, angle):
    rot = create_rotation_matrix(*angle)

    rotated = dot(rot, points).T

    center = array(center) - 0.5 * array(shape[::-1], dtype=t)
    center = dot(rot, center)

    dR1 = (rotated - center)**2 / radii1**2
    dR2 = (rotated - center)**2 / radii2**2
    
    nR1 = sum(dR1, axis=1).reshape(shape)
    nR2 = sum(dR2, axis=1).reshape(shape)

    return where(sqrt(nR2) <= 1, 1, 0) - where(sqrt(nR1) <= 1, 1, 0)

@jit
def create_beta(center, A, r0, beta, ellip, phi):
    radii = array([1, 1-ellip, 1], dtype=t)
    angles = deg2rad(array([0, 0, phi], dtype=t))
    r = create_ellipsoidal_grid(center, radii, angles)
    return (A * (1 + (r/r0)**2)**(-3/2*beta))**2

@jit
def create_model(center, A, r0, beta, ellip, phi, center_2, A_2, r0_2, beta_2, phi_2):
    model3D = create_beta(center, A, r0, beta, ellip, phi)
    model3D = where(A_2 != 0, model3D + create_beta(center_2, A_2, r0_2, beta_2, ellip, phi_2), model3D)
    return model3D

@jit
def create_rims(img_center, center, radii1, radii2, angle, typ, angle_to_center):
    radius = sum((img_center - center)**2)**0.5
    ell = create_ellipsoidal_grid(img_center, array([radius,radius,radius]), array([0,0,0]))

    # angle = where(typ == 1, array([0, 0, angle_to_center]), array([0, 0, 0]))
    # angle = where(typ == 1, array([0, 0, angle_to_center]), array([0, 0, pi / 2 - angle_to_center]))
    # fac = where(typ == 1, 1, 1) #0.75)
    radii2 = where(typ == 1, radii2 * array([1,2,1]), radii2)

    r = create_ellipsoidal_grid(center, radii2, angle)

    r1, r2 = min(radii1, axis=0), min(radii2, axis=0)
    alpha, beta = -1 / (r1/r2 - 1), 1 / (r1/r2 - 1)
    model = alpha + beta * sqrt(r)

    shell = create_ellipsoidal_shell(center, radii1, radii2, angle) * model

    shell = where(typ == 2, shell * ell**3, shell)
    shell = where(shell < 0, 0, shell)
    shell = where(shell > 1, 1, shell)

    return shell

@jit
def create_cavity_pair(center, distances, varphi, theta, radii, ellip, phi, rim_size, rim_height, rim_type):
    varphi1 = deg2rad(varphi[0])
    varphi2 = deg2rad(varphi[1])
    theta1 = deg2rad(theta[0])
    theta2 = deg2rad(theta[1])
    phi1 = deg2rad(phi[0] - varphi[0])
    phi2 = deg2rad(phi[1] - varphi[1])

    x1 = center[0] + distances[0] * cos(varphi1) * cos(theta1)
    y1 = center[1] + distances[0] * sin(varphi1) * cos(theta1)
    z1 = center[2] + distances[0] * sin(theta1)
    x2 = center[0] + distances[1] * cos(varphi2) * cos(theta2)
    y2 = center[1] + distances[1] * sin(varphi2) * cos(theta2)
    z2 = center[2] + distances[1] * sin(theta2)

    rx1 = radii[0]
    ry1 = radii[0] * (1 - ellip[0])
    rz1 = max(array([rx1, ry1]))

    rx2 = radii[1]
    ry2 = radii[1] * (1 - ellip[1])
    rz2 = max(array([rx2, ry2]))

    cav3D = create_ellipsoid(array([x1, y1, z1]), array([rx1, ry1, rz1]), array([0,0,phi1]))
    cav3D += create_ellipsoid(array([x2, y2, z2]), array([rx2, ry2, rz2]), array([0,0,phi2]))

    f = 1 + rim_size
    rim1 = where(rim_height != 0, rim_height * create_rims(center, array([x1, y1, z1]), array([rx1, ry1, rz1]), array([f*rx1, f*ry1, f*rz1]), array([0,0,phi1]), rim_type, phi1), zeros(shape))
    rim2 = where(rim_height != 0, rim_height * create_rims(center, array([x2, y2, z2]), array([rx2, ry2, rz2]), array([f*rx2, f*ry2, f*rz2]), array([0,0,phi2]), rim_type, phi2), zeros(shape))
    rim3D = rim1 + rim2

    return cav3D, rim3D

@jit
def apply_sloshing(image, period, angle, depth, direction):
    R = linspace(-1, 1, 128)
    x, y = meshgrid(R, R)
    r = sqrt(x**2 + y**2) * pi * period
    angle = where(angle == 90, angle + 1e-5, angle)
    rotation = deg2rad(angle)
    x, y = x * cos(rotation) - y * sin(rotation), x * sin(rotation) + y * cos(rotation)
    phi = arctan(y / x)
    angle = where(x > 0, phi + r**0.5*2, phi + r**0.5*2 + pi)
    val = cos(angle) * depth #+ ones(x.shape)
    # slosh = where(val < 0, 0, val)
    slosh = 10**val
    return image * where(direction, slosh, slosh.T)

@jit
def apply_mask(model3D, cav3D, rim3D, bkg):
    masked3D = where(cav3D > 0, 0, model3D)
    masked3D = masked3D + rim3D
    masked3D = masked3D + bkg / 128
    return masked3D

@jit
def beta_model(n=0, dx=0, dy=0, dx_2=0, dy_2=0, phi=0, phi_2=0, 
            A=10, r0=10, beta=1, ellip=0, A_2=0, r0_2=0, beta_2=0, bkg=0, 
            s_depth=0, s_period=0, s_dir=0, s_angle=0,
            r1=20, r2=20, varphi1=0, varphi2=180, theta1=0, theta2=0, 
            R1=10, R2=10, e1=0, e2=0, phi1=0, phi2=0,
            rim_size=0, rim_height=0, rim_type=0):

    center = (64-0.5) * ones(3) - array([dx, dy, 0])
    center_2 = (64-0.5) * ones(3) - array([dx_2, dy_2, 0])

    model3D = create_model(center, A, r0, beta, ellip, phi, center_2, A_2, r0_2, beta_2, phi_2)

    rim_A = rim_height * A * r0 / (beta+0.6)**5
    cav3D, rim3D = create_cavity_pair(center, array([r1, r2]), array([varphi1, varphi2]), array([theta1, theta2]), array([R1, R2]), array([e1, e2]), array([phi1, phi2]), rim_size, rim_A, rim_type)

    masked3D = apply_mask(model3D, cav3D, rim3D, bkg)

    image = sum(masked3D, axis=1)
    rim = sum(rim3D, axis=1)
    mask = sum(cav3D, axis=1)
    volume = sum(cav3D)
    binary_mask = where(mask > 0, 1, 0)

    image = where(s_depth, apply_sloshing(image, s_period, s_angle, s_depth, s_dir), image)

    key = random.PRNGKey(n)
    noisy = random.poisson(key, image, shape=(128,128))

    return noisy, binary_mask, volume

@vmap
def run_vect(n, dx, dy, dx_2, dy_2, phi, phi_2,
             A, r0, beta, ellip, A_2, r0_2, beta_2, bkg, 
             s_depth, s_period, s_dir, s_angle,
             r1, r2, varphi1, varphi2, theta1, theta2, 
             R1, R2, e1, e2, phi1, phi2,
             rim_size, rim_height, rim_type):

    return beta_model(n, dx, dy, dx_2, dy_2, phi, phi_2,
                    A, r0, beta, ellip, A_2, r0_2, beta_2, bkg,
                    s_depth, s_period, s_dir, s_angle,
                    r1, r2, varphi1, varphi2, theta1, theta2, 
                    R1, R2, e1, e2, phi1, phi2,
                    rim_size, rim_height, rim_type)

def get_batch(i1, i2, params):
    p = params.loc[i1:i2]

    parnames = ["dx", "dy", "dx_2", "dy_2", "phi", "phi_2",
                "A", "r0", "beta", "ellip", "A_2", "r0_2", "beta_2", "bkg", 
                "s_depth", "s_period", "s_dir", "s_angle",
                "r1", "r2", "varphi1", "varphi2", "theta1", "theta2", 
                "R1", "R2", "e1", "e2", "phi1", "phi2",
                "rim_size", "rim_height", "rim_type"]

    for par in parnames:
        globals()[par] = stack(list(p[par]))

    nums = stack([i for i in range(i1,i2+1)])

    imgs, mask, vol = run_vect(nums,
                            dx, dy, dx_2, dy_2, phi, phi_2,
                            A, r0, beta, ellip, A_2, r0_2, beta_2, bkg, 
                            s_depth, s_period, s_dir, s_angle,
                            r1, r2, varphi1, varphi2, theta1, theta2, 
                            R1, R2, e1, e2, phi1, phi2,
                            rim_size, rim_height, rim_type)

    return imgs, mask, vol
