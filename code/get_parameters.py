import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

# Define the function of the dynamics
def cosine_dynamics(X, t, constant, alpha, sigma, B):
    x,y = X # x = angle = theta, y = dxdt = angular velocity = omega, dydt = angular acceleration

    conditions = [(x > (B - sigma)) & (x < (B + sigma)),
              (x <= (B - sigma)) | (x >= (B + sigma))]

    functions = [lambda x: 1 + np.cos(2 * np.pi * (x - B) / (2 * sigma)), lambda x: 0]

    dxdt = y
    dydt = constant * (x - np.sign(B) * alpha * np.piecewise(x, conditions, functions)) # minuses the cosine
    return [dxdt, dydt]

#function to get max velocity for a specific alpha
def get_max_velocity(alpha, cos_start, cos_acc0, constant):
    # get sigma and B
    sigma = sp.Symbol('sigma')
    # X = cos_acc0
    B = sigma + cos_start
    equation = constant * (cos_acc0 - alpha * (1 + sp.cos((2 * sp.pi * (cos_acc0 - B)) / (2 * sigma)))) 
    solutions = sp.solve(equation, sigma)
    try: 
        sigma = float(solutions[1]) ## stability width
    except TypeError:
        return None,None,None
    B = sigma + cos_start
    # print(B, sigma)
    max_velocity = None
    for i in np.arange(cos_start, cos_acc0, 0.0001):
        # print(i)
        X = [i,0]
        t = np.linspace(0, 5, 100000)
        sol = odeint(cosine_dynamics, X, t, args=(constant, alpha, sigma, B)) #integrates dxdt and dydt to get theta (angle) and omega (velocity)
        theta = np.rad2deg(sol[:,0])
        omega = np.rad2deg(sol[:,1])
        # print(theta)
        # print(np.rad2deg(cos_acc0))
        if (any(x < 0 for x in omega)):
            max_velocity = max(omega)
            break
    if isinstance(max_velocity, (int, float)):
        return max_velocity, B, sigma
    else: return None,None,None

def get_parameters(mass, height, setpoint_baseline, sd_baseline, maxV):
    #Constants
    scaling = 0.971
    L = 1.09
    g = 9.81
    m = mass #change
    H = height #change
    I = 0.35*m*H**2     # Distributed Inertia
    b = 0
    maxAngle = 65
    constant = scaling * m * g * L / I

    ## baselines
    setpoint_baseline = setpoint_baseline #change
    sd_baseline = sd_baseline #change
    cos_start = np.deg2rad(setpoint_baseline + 1 * sd_baseline) 
    cos_acc0 = np.deg2rad(setpoint_baseline + 3 * sd_baseline) 
    # print(cos_start, cos_acc0, constant)

    alpha_out = 0
    velocity_out = 0
    for alpha in np.arange(0.01, 0.04, 0.0001):
        max_velocity, B, sigma = get_max_velocity(alpha, cos_start, cos_acc0, constant)
        if isinstance(max_velocity, (int, float)):
            if max_velocity < maxV:
                alpha_out = alpha
                B_out = B
                sigma_out = sigma
                velocity_out = max_velocity
            else:
                print("stability location: ", np.rad2deg(B_out))
                print("stability width: ", np.rad2deg(sigma_out))
                print("stability strength: ", alpha_out)
                # print(velocity_out)
                break
        else:
            continue

get_parameters(mass, height, setpoint_baseline, sd_baseline, maxV) # sd_everything + personal maxV