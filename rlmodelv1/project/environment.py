import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

def cosine_dynamics(th, constant, alpha, sigma, B):
    # X = [theta, theta_dot]
    x = th
    conditions = [
        (x > (B - sigma)) & (x < (B + sigma)),
        (x <= (B - sigma)) | (x >= (B + sigma))
    ]
    functions = [
        lambda x: 1 + np.cos(2 * np.pi * (x - B) / (2 * sigma)), 
        lambda x: 0
    ]

    dydt = constant * (x - np.sign(B) * alpha * np.piecewise(x, conditions, functions))
    return dydt

# Register the custom environment
gym.register(
    id='StandingBalance-v0',
    entry_point='environment:StandingBalanceEnv',
    max_episode_steps=15000,  # adjust as needed
)

class StandingBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 dt=0.004, 
                 max_episode_steps=15000,
                 seed=0,
                 render_mode=None):

        super().__init__()
        
        # Physics parameters
        self.g = 9.81
        self.m = 74
        self.l = 1.70*0.52
        self.I = self.m * (self.l**2)
        self.b = 0.001
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.th_range_rad = [-3*np.pi/180, 6*np.pi/180]
        self.thdot_range_rad = [-15*np.pi/180, 15*np.pi/180]
        self.act_range=[-100, 50]


        # State: (theta_dot, theta)
        # We'll allow for large ranges, but ideally the controller keeps it near 0
        # high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-15*np.pi/180, -3*np.pi/180], dtype=np.float32), np.array([15*np.pi/180, 6*np.pi/180], dtype=np.float32), dtype=np.float32)

        # Action: single torque value
        self.action_space = spaces.Box(low=-100.0, high=50.0, shape=(1,), dtype=np.float32)

        # # Observation and Action spaces
        # high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # self.action_space = spaces.Box(low=-100.0, high=50.0, shape=(1,), dtype=np.float32)

        self.current_step = 0
        self.state = None
        self.seed(seed)

        # For rendering
        self.fig = None
        self.ax = None

        # Parameters for altered dynamics
        self.constant = 0.971 * self.m * self.g * self.l / self.I
        # self.alpha = 0.0283
        self.alpha = 0.1
        self.sigma = 0.030887 
        self.B = 0.05139

        # Will we use altered dynamics this episode?
        self.use_altered = False

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        # theta = 0.05 * np.random.randn()
        # theta_dot = 0.05 * np.random.randn()
        theta = 0
        theta_dot = 0
        self.state = np.array([theta_dot, theta], dtype=np.float32)

        # 50% chance to use altered dynamics
        # self.use_altered = (np.random.rand() < 0.5)
        # self.use_altered = True
        radnum = np.random.rand()
        if radnum > 0.5:
            self.use_altered = True
        else:
            self.use_altered = False

        return self.state.copy(), {}

    def step(self, action):
        action = float(action[0])
        theta_dot, theta = self.state

        # Compute A matrix depending on use_altered
        if self.use_altered:
            # Use cosine_dynamics instead of constant 0.971*m*g*l/I
            # This value replaces A[1,0]
            dyn_value = cosine_dynamics(theta, 0.971*self.m*self.g*self.l/self.I, self.alpha, self.sigma, self.B)
            A = np.array([[0, 1],
                          [dyn_value, -self.b/self.I]])
            # if (theta<self.B + 0.001) and (theta>self.B - 0.001-self.sigma):
            #     print(dyn_value, np.rad2deg(theta), np.rad2deg(theta_dot), action)
            #     if np.rad2deg(theta_dot)>1.5:
            #         action = -30
            #         print(action)
            #     else: action = 0

        else:
            A = np.array([[0, 1],
                          [0.971*self.m*self.g*self.l/self.I, -self.b/self.I]])
            # print(0.971*self.m*self.g*self.l/self.I)

        B = np.array([0, 1/self.I])

        state_vec = np.array([theta, theta_dot])
        # print(action)
        diff_state = A @ state_vec + B * action
        
        theta_dot_new = theta_dot + diff_state[1]*self.dt
        theta_new = theta + theta_dot_new*self.dt


        self.state = np.array([theta_new, theta_dot_new], dtype=np.float32)
        # dyn_indicator = 1.0 if self.use_altered else 0.0
        obs = np.array([theta_dot_new, theta_new], dtype=np.float32)

        # Reward: Encourage longer episodes and staying upright
        th = theta_new
        thdot = theta_dot_new
        reward = - (th**2 + thdot**2)*0.1
        reward = -0.01 * abs(action)
        reward += 1  # Encourage survival each step

        done = (th > 6*np.pi/180) or (th < -3*np.pi/180)
        truncated = (self.current_step >= self.max_episode_steps)

        self.current_step += 1
        return self.state.copy(), reward, done, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_rgb_array(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(4,4), dpi=100)
            self.ax.set_aspect('equal')
            self.ax.axis('off')

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        theta = self.state[1]
        x = self.l * np.sin(theta)
        y = -self.l * np.cos(theta)

        self.ax.set_xlim(-1.2*self.l, 1.2*self.l)
        self.ax.set_ylim(-1.2*self.l, 1.2*self.l)

        self.ax.plot([0],[0],'ko')
        self.ax.plot([0,x],[0,y],'k-', lw=2)
        self.ax.plot([x],[y],'ro', markersize=10)

        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((h, w, 3))
        return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

##############

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import matplotlib.pyplot as plt



# class StandingBalanceEnv(gym.Env):
#     # Register the custom environment under a unique ID
#     gym.register(
#         id='StandingBalance-v0',
#         entry_point='environment:StandingBalanceEnv',
#         max_episode_steps=15000,  # adjust as needed
#     )
#     metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
#     def __init__(self, 
#                  dt=0.004, 
#                  max_episode_steps=15000,
#                  seed=0,
#                  render_mode=None):

#         super().__init__()
        
#         # Physics parameters for a simple inverted pendulum
#         # We'll treat it as a pendulum of length l with mass m at the end.
#         self.g = 9.81
#         self.m = 74
#         self.l = 1.70*0.52
#         self.I = self.m * (self.l**2)  # moment of inertia about pivot
#         self.b = 0.001  # damping term
#         self.dt = dt
#         self.max_episode_steps = max_episode_steps
#         self.render_mode = render_mode
#         self.th_range_rad = [-3*np.pi/180, 6*np.pi/180]
#         self.thdot_range_rad = [-15*np.pi/180, 15*np.pi/180]
#         self.act_range=[-100, 50]


#         # State: (theta_dot, theta)
#         # We'll allow for large ranges, but ideally the controller keeps it near 0
#         # high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
#         # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
#         self.observation_space = spaces.Box(np.array([-15*np.pi/180, -3*np.pi/180], dtype=np.float32), np.array([15*np.pi/180, 6*np.pi/180], dtype=np.float32), dtype=np.float32)

#         # Action: single torque value
#         self.action_space = spaces.Box(low=-100.0, high=50.0, shape=(1,), dtype=np.float32)

#         self.current_step = 0
#         self.state = None
#         self.seed(seed)

#         # For rendering
#         self.fig = None
#         self.ax = None

#     def seed(self, seed=None):
#         np.random.seed(seed)
#         return [seed]

#     def reset(self, *, seed=None, options=None):
#         if seed is not None:
#             self.seed(seed)

#         self.current_step = 0
#         # Start near upright with a small random angle and angular velocity
#         theta = 0.05 * np.random.randn()
#         theta_dot = 0.05 * np.random.randn()
#         self.state = np.array([theta_dot, theta], dtype=np.float32)

#         return self.state.copy(), {}
    
#     def step(self, action):
#         action = float(action[0])
#         theta_dot, theta = self.state

#         # Compute derivatives using the given formula
#         # theta_ddot = (0.971*m*g*l/I)*theta + (-b/I)*theta_dot + (1/I)*T
#         # But note that in the original matrix form:
#         # [theta_dot_new; theta_ddot] = A * [theta; theta_dot] + B * torque
        
        
#         A = np.array([[0, 1],
#                       [0.971*self.m*self.g*self.l/self.I, -self.b/self.I]])
#         B = np.array([0, 1/self.I])

#         state_vec = np.array([theta, theta_dot])
#         diff_state = A @ state_vec + B * action
        
#         # diff_state = [theta_dot, theta_ddot]
#         # Update using Euler integration
#         theta_dot_new = theta_dot + diff_state[1]*self.dt
#         theta_new = theta + theta_dot_new*self.dt

#         self.state = np.array([theta_new, theta_dot_new], dtype=np.float32)

#         # Reward: Encourage staying near upright (theta=0, theta_dot=0)
#         # Add a small positive reward per step to encourage longer balancing
#         th = theta_new
#         thdot = theta_dot_new
#         reward = - (th**2 + 0.1*thdot**2) + 0.1  # small positive offset
#         reward += 1 #encourage longer episodes

#         # Done if angle too large
#         done = (th > self.th_range_rad[1]) or (th < self.th_range_rad[0])
#         truncated = (self.current_step >= self.max_episode_steps)

#         self.current_step += 1
#         return self.state.copy(), reward, done, truncated, {}

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_rgb_array()
#         return None

#     def _render_rgb_array(self):
#         if self.fig is None or self.ax is None:
#             self.fig, self.ax = plt.subplots(figsize=(4,4), dpi=100)
#             self.ax.set_aspect('equal')
#             self.ax.axis('off')

#         self.ax.clear()
#         self.ax.set_aspect('equal')
#         self.ax.axis('off')

#         theta = self.state[1]
#         x = self.l * np.sin(theta)
#         y = -self.l * np.cos(theta)

#         self.ax.set_xlim(-1.2*self.l, 1.2*self.l)
#         self.ax.set_ylim(-1.2*self.l, 1.2*self.l)

#         # Draw pivot
#         self.ax.plot([0],[0],'ko')
#         # Draw rod
#         self.ax.plot([0,x],[0,y],'k-', lw=2)
#         # Draw bob
#         self.ax.plot([x],[y],'ro', markersize=10)

#         self.fig.canvas.draw()
#         w, h = self.fig.canvas.get_width_height()
#         img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img = img.reshape((h, w, 3))

#         return img

#     def close(self):
#         if self.fig is not None:
#             plt.close(self.fig)
#             self.fig = None
#             self.ax = None


###########################

# # import gymnasium as gym
# # from gymnasium import spaces
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from io import BytesIO
# # from PIL import Image

# # # Register the custom environment under a unique ID
# # gym.register(
# #     id='StandingBalance-v0',
# #     entry_point='environment:StandingBalanceEnv',
# #     max_episode_steps=15000,  # adjust as needed
# # )

# # class StandingBalanceEnv(gym.Env):
# #     metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
# #     def __init__(self, 
# #                  dt=0.02, 
# #                  max_episode_steps=20000, # 15000*0.02 = 300s = 5min
# #                  seed=0,
# #                  render_mode=None,
# #                  ThRange=[-3, 6],  # degrees
# #                  ThDotRange=[-15, 15], # deg/s
# #                  ActRange=[-100, 50]):
# #         super().__init__()


# # #     def __init__(self,
# # #                  dt=0.02,
# # #                  max_episode_steps=500,
# # #                  g=9.81,
# # #                  m=1.0,
# # #                  l=1.0,
# # #                  c=0.1,
# # #                  torque_max=10.0,
# # #                  render_mode=None,
# # #                  seed=None):
# # #         super().__init__()

# #         # Set environment parameters
# #         self.render_mode = render_mode
# #         self.m = 74.0
# #         self.h = 1.70*0.52
# #         self.I = self.m*(self.h**2)
# #         self.g = 9.81
# #         self.c = 5.73
        
# #         self.th_range = ThRange
# #         self.thdot_range = ThDotRange
# #         self.act_range = ActRange

# #         # Convert angle ranges to radians
# #         self.th_range_rad = [ThRange[0]*np.pi/180, ThRange[1]*np.pi/180]
# #         self.thdot_range_rad = [ThDotRange[0]*np.pi/180, ThDotRange[1]*np.pi/180]

# #         self.mn_coeff = 0.02
# #         self.thn_coeff = 0.003
# #         self.thdotn_coeff = 0.001

# #         # Load noise data
# #         self.motor_noiseTest = np.loadtxt("noise/Noise1Test.txt", delimiter=',')
# #         self.th_noiseTest = np.loadtxt("noise/Noise2Test.txt", delimiter=',')
# #         self.thdot_noiseTest = np.loadtxt("noise/Noise3Test.txt", delimiter=',')

# #         self.motor_noise = np.loadtxt("noise/Noise1Train.txt", delimiter=',')
# #         self.th_noise = np.loadtxt("noise/Noise2Train.txt", delimiter=',')
# #         self.thdot_noise = np.loadtxt("noise/Noise3Train.txt", delimiter=',')

# #         # Episode control
# #         self.dt = dt
# #         self.max_episode_steps = max_episode_steps
# #         self.current_step = 0
# #         self.num_episode = 0

# #         # Gym action/observation spaces
# #         # Observation: [theta_dot (rad/s), theta (rad)]
# #         self.observation_space = spaces.Box(
# #             low=np.array([self.thdot_range_rad[0], self.th_range_rad[0]], dtype=np.float32),
# #             high=np.array([self.thdot_range_rad[1], self.th_range_rad[1]], dtype=np.float32)
# #         )

# #         # Action: torque (Nm)
# #         self.action_space = spaces.Box(
# #             low=np.array([self.act_range[0]], dtype=np.float32),
# #             high=np.array([self.act_range[1]], dtype=np.float32)
# #         )

# #         self.seed(seed)

# #         # Passive stiffness state variables
# #         self.prevVelocitySign = 1
# #         self.prevPStorque = 0
# #         self.prevReversal = 0
# #         self.prevPStorqueHistory = 0
# #         self.prev_act = 0

# #         self.state = None

# #         # Set up a figure for rendering
# #         self.fig = None
# #         self.ax = None

# #     def seed(self, seed=None):
# #         np.random.seed(seed)
# #         return [seed]

# #     def reset_pt(self, now_state):
# #         self.prevVelocitySign = np.sign(now_state[0])
# #         self.prevPStorque = 0
# #         self.prevReversal = 0
# #         self.prevPStorqueHistory = 0
# #         self.prev_act = 0

# #     def passive_torque(self, now_state):
# #         th = now_state[1]
# #         thdot = now_state[0]

# #         rotationSign = np.sign(thdot)
# #         if rotationSign == 0:
# #             rotationSign = 1

# #         if rotationSign != self.prevVelocitySign:
# #             self.prevReversal = th
# #             self.prevVelocitySign = rotationSign
# #             self.prevPStorque = self.prevPStorqueHistory

# #         if abs(th - self.prevReversal) <= 0.03/180*np.pi:
# #             rotation = rotationSign * 0.03/180*np.pi
# #         else:
# #             rotation = th - self.prevReversal

# #         coeffT = (0.467 * (abs(rotation)**(-0.334)) * 2 * self.m * 180/np.pi *
# #                   self.g * self.h / (11*(180/np.pi)))

# #         psTorque = coeffT*rotation + self.prevPStorque
# #         self.prevPStorqueHistory = psTorque
# #         return psTorque

# #     def muscle_dyn(self, true_torque, MVC, dt):
# #         activation = true_torque/MVC
# #         if np.sign(true_torque) != np.sign(self.prev_act):
# #             prevAct = 0
# #         else:
# #             prevAct = abs(self.prev_act)

# #         if abs(activation) >= prevAct:
# #             tau = 0.01*(0.5+1.5*prevAct)
# #         else:
# #             tau = 0.04/(0.5+1.5*prevAct)

# #         output = (abs(activation)-prevAct)/tau*dt + prevAct
# #         torque_final = np.sign(true_torque)*output*MVC
# #         self.prev_act = activation
# #         return torque_final

# #     def dynamics(self, true_state, true_torque, ps_torque, t, num_episode, dt):
# #         idx = num_episode % 1000

# #         if true_torque < 0:
# #             mn_added = self.mn_coeff * abs(true_torque) * 0.77 * self.motor_noise[t, idx]
# #         else:
# #             mn_added = self.mn_coeff * abs(true_torque) * self.motor_noise[t, idx]

# #         exerted_torque = true_torque + mn_added

# #         A = np.array([[-self.c/self.I, self.m*self.g*self.h/self.I],
# #                       [1, 0]])
# #         # B = np.array([[1/self.I],[0]])
# #         # C = np.array([[-1/self.I],[0]])
# #         B = np.array([1/self.I, 0])   # shape (2,)
# #         C = np.array([-1/self.I, 0])  # shape (2,)

# #         diff_state = A@true_state + B*exerted_torque + C*ps_torque
# #         next_state = diff_state*dt + true_state

# #         tdn_added = self.thdotn_coeff * self.thdot_noise[t, idx]
# #         perceived_thdot = next_state[0] + tdn_added

# #         tn_added = self.thn_coeff * self.th_noise[t, idx]
# #         perceived_th = next_state[1] + tn_added

# #         perceived_state = np.array([perceived_thdot, perceived_th])
# #         return perceived_state, next_state

# #     def failure_check(self, now_state):
# #         # return 1 if failed
# #         # print("ns:", now_state)
# #         if (now_state[1] > self.th_range_rad[1]) or (now_state[1] < self.th_range_rad[0]):
# #             return 1
# #         return 0

# #     def initiate_test(self, totalDelay):
# #         # Compute offsets in degrees
# #         gTmax = self.m*self.g*self.h*self.th_range[1]*np.pi/180
# #         galpha = gTmax/self.I
# #         Vmax = galpha*totalDelay/1000
# #         deltaThDelay = (Vmax**2)/(2*galpha)
# #         TBackward = abs(self.act_range[0])-gTmax
# #         balpha = TBackward/self.I
# #         deltaThTorque = (Vmax**2)/(2*balpha)
# #         dThTot1 = (deltaThTorque+deltaThDelay)*180/np.pi
# #         thetaRange = [self.th_range[0]+dThTot1, self.th_range[1]-dThTot1]

# #         gTmin = self.m*self.g*self.h*(self.th_range[1]-1)*np.pi/180
# #         TBackwardMax = abs(self.act_range[0])-gTmin
# #         balphaMax = TBackwardMax/self.I
# #         dThMaxDot = ((self.thdot_range[1]*np.pi/180)**2)/(2*balphaMax)
# #         dThTot2 = (deltaThDelay+dThMaxDot)*180/np.pi
# #         thInnerLim = [self.th_range[0]+dThTot2, self.th_range[1]-dThTot2]

# #         th_rand = np.mean(thetaRange) + (np.ptp(thetaRange)/8)*np.random.randn()
        
# #         # Compute velocity range
# #         if th_rand < thInnerLim[1]:
# #             vel_upper = self.thdot_range[1]
# #         else:
# #             mLine = self.thdot_range[1]/(thInnerLim[1]-thetaRange[1])
# #             bLine = -mLine*thetaRange[1]
# #             vel_upper = mLine*th_rand+bLine

# #         if th_rand > thInnerLim[0]:
# #             vel_lower = self.thdot_range[0]
# #         else:
# #             mLine = self.thdot_range[0]/(thInnerLim[0]-thetaRange[0])
# #             bLine = -mLine*thetaRange[0]
# #             vel_lower = mLine*th_rand+bLine

# #         vel_range = [vel_lower, vel_upper]
# #         thDot_rand = np.mean(vel_range)+(np.ptp(vel_range)/6)*np.random.randn()

# #         state_deg = np.array([thDot_rand, th_rand])
# #         return state_deg

# #     def reset(self, *, seed=None, options=None):
# #         if seed is not None:
# #             self.seed(seed)

# #         self.num_episode += 1
# #         self.current_step = 0

# #         state_deg = self.initiate_test(totalDelay=10)
# #         # Convert state to radians
# #         # state = [thDot (deg/s), th (deg)] -> [rad/s, rad]
# #         state_rad = np.array([state_deg[0]*np.pi/180, state_deg[1]*np.pi/180])

# #         self.state = state_rad
# #         self.reset_pt(self.state)

# #         # Return initial observation (perceived_state can be just the state at reset)
# #         return np.array(self.state, dtype=np.float32), {}

# #     def step(self, action):
# #         action = float(action[0])  # single torque value
# #         ps_torque = self.passive_torque(self.state)

# #         MVC = 100.0  # Example: maximum voluntary contraction torque
# #         torque_applied = self.muscle_dyn(action, MVC, self.dt)

# #         perceived_state, next_state = self.dynamics(
# #             true_state=self.state,
# #             true_torque=torque_applied,
# #             ps_torque=ps_torque,
# #             t=self.current_step,
# #             num_episode=self.num_episode,
# #             dt=self.dt
# #         )

# #         self.state = next_state

# #         done = (self.failure_check(self.state) == 1)
# #         truncated = (self.current_step >= self.max_episode_steps)

# #         # Simple reward: negative of squared angle and a small penalty for angular velocity
# #         # print("ps:", perceived_state)
# #         thdot = perceived_state[0]
# #         th = perceived_state[1]
# #         reward = - (th**2 + 0.1*(thdot**2))

# #         #####

# #         # Add time-based reward for surviving longer
# #         time_bonus = 0.1  # Reward for each step the agent survives
# #         reward += time_bonus

# #         # Add metabolic cost
# #         torque_cost = 0.01 * abs(action)
# #         reward -= torque_cost

# #         # # Add smooth control penalty
# #         # delta_action = abs(action - self.prev_act)
# #         # smoothness_cost = 0.005 * delta_action

# #         # # Update reward
# #         # reward -= (torque_cost + smoothness_cost)

# #         # # Reward for being close to upright
# #         # upright_bonus = 1.0 if abs(th) < 0.1 and abs(thdot) < 0.5 else 0
# #         # reward += upright_bonus

# #         #####

# #         self.current_step += 1
# #         info = {}
# #         return np.array(perceived_state, dtype=np.float32), reward, done, truncated, info
    
# #     def render(self):
# #         if self.render_mode == "rgb_array":
# #             return self._render_rgb_array()
# #         # If no rendering is required, pass
# #         return None

# #     def _render_rgb_array(self):
# #         # If no figure/axes exist, create them
# #         if self.fig is None or self.ax is None:
# #             self.fig, self.ax = plt.subplots(figsize=(4,4), dpi=100)
# #             self.ax.set_aspect('equal')
# #             self.ax.axis('off')

# #         self.ax.clear()
# #         self.ax.set_aspect('equal')
# #         self.ax.axis('off')

# #         # Pendulum length for drawing
# #         l = self.h  # Using self.h as the pendulum length

# #         # Set plot limits so the pendulum is always visible
# #         self.ax.set_xlim(-1.2*l, 1.2*l)
# #         self.ax.set_ylim(-1.2*l, 1.2*l)

# #         # Current angle (theta) from state
# #         theta = self.state[1]
# #         # Compute bob position
# #         x = l * np.sin(theta)
# #         y = -l * np.cos(theta)

# #         # Draw the pivot
# #         self.ax.plot([0],[0],'ko')
# #         # Draw the pendulum rod
# #         self.ax.plot([0, x], [0, y], 'k-', lw=2)
# #         # Draw the bob
# #         self.ax.plot([x],[y],'ro', markersize=10)

# #         # Render the figure to an RGB array
# #         self.fig.canvas.draw()
# #         w, h = self.fig.canvas.get_width_height()
# #         img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
# #         img = img.reshape((h*2, w*2, 3))

# #         return img

# #     def close(self):
# #         if self.fig is not None:
# #             plt.close(self.fig)
# #             self.fig = None
# #             self.ax = None

# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from io import BytesIO
# # # from PIL import Image

# # # # Register the custom environment under a unique ID
# # # gym.register(
# # #     id='StandingBalance-v0',
# # #     entry_point='environment:StandingBalanceEnv',
# # #     max_episode_steps=1000,  # adjust as needed
# # # )

# # # class StandingBalanceEnv(gym.Env):
# # #     metadata = {"render_modes": ["rgb_array"]}

# # #     def __init__(self,
# # #                  dt=0.02,
# # #                  max_episode_steps=500,
# # #                  g=9.81,
# # #                  m=1.0,
# # #                  l=1.0,
# # #                  c=0.1,
# # #                  torque_max=10.0,
# # #                  render_mode=None,
# # #                  seed=None):
# # #         super().__init__()
        
# # #         self.render_mode = render_mode
# # #         self.g = g
# # #         self.m = m
# # #         self.l = l
# # #         self.c = c
# # #         self.I = self.m*(self.l**2)
# # #         self.dt = dt
# # #         self.max_episode_steps = max_episode_steps
        
# # #         obs_high = np.array([np.pi, 10.0], dtype=np.float32)
# # #         self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
# # #         self.action_space = spaces.Box(low=-torque_max, high=torque_max, shape=(1,), dtype=np.float32)
        
# # #         self.seed(seed)
        
# # #         self.current_step = 0
# # #         self.state = None

# # #         # Set up a figure for rendering if needed
# # #         self.fig = None
# # #         self.ax = None

# # #     def seed(self, seed=None):
# # #         np.random.seed(seed)
# # #         return [seed]

# # #     def reset(self, *, seed=None, options=None):
# # #         if seed is not None:
# # #             self.seed(seed)
        
# # #         self.current_step = 0
# # #         theta = 0.05 * np.random.randn()
# # #         theta_dot = 0.05 * np.random.randn()
# # #         self.state = np.array([theta, theta_dot], dtype=np.float32)
        
# # #         if self.render_mode == "rgb_array":
# # #             return self.state, {}
# # #         else:
# # #             return self.state, {}

# # #     def step(self, action):
# # #         torque = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
# # #         theta, theta_dot = self.state

# # #         theta_acc = (self.g/self.l)*np.sin(theta) + (torque/self.I) - (self.c*theta_dot/self.I)
# # #         theta_dot_new = theta_dot + theta_acc * self.dt
# # #         theta_new = theta + theta_dot_new * self.dt
# # #         self.state = np.array([theta_new, theta_dot_new], dtype=np.float32)

# # #         reward = -(theta_new**2 + 0.1*(theta_dot_new**2))

# # #         done = False
# # #         truncated = False
# # #         if abs(theta_new) > np.pi/2:
# # #             done = True

# # #         self.current_step += 1
# # #         if self.current_step >= self.max_episode_steps:
# # #             truncated = True

# # #         return self.state, reward, done, truncated, {}

# # #     def render(self):
# # #         if self.render_mode == "rgb_array":
# # #             return self._render_rgb_array()
# # #         # If no rendering is required, pass
# # #         return None

# # #     def _render_rgb_array(self):
# # #         # If no figure exists, create one
# # #         if self.fig is None:
# # #             self.fig, self.ax = plt.subplots(figsize=(4,4))
# # #             self.ax.set_xlim(-1.2*self.l, 1.2*self.l)
# # #             self.ax.set_ylim(-1.2*self.l, 1.2*self.l)
# # #             self.ax.set_aspect('equal')
# # #             self.ax.axis('off')

# # #         self.ax.clear()
# # #         self.ax.set_xlim(-1.2*self.l, 1.2*self.l)
# # #         self.ax.set_ylim(-1.2*self.l, 1.2*self.l)
# # #         self.ax.set_aspect('equal')
# # #         self.ax.axis('off')

# # #         # Draw the pivot
# # #         self.ax.plot([0,0],[0,0],'ko')
        
# # #         # The pendulum angle (theta) is measured from the vertical (downwards)
# # #         # If you consider 0 at upright, the coordinates of the pendulum end are:
# # #         # x = l*sin(theta), y = -l*cos(theta)
# # #         theta = self.state[0]
# # #         x = self.l * np.sin(theta)
# # #         y = -self.l * np.cos(theta)

# # #         # Draw the pendulum rod
# # #         self.ax.plot([0, x], [0, y], 'k-', lw=2)

# # #         # Draw a bob at the end
# # #         self.ax.plot([x],[y],'ro', markersize=10)

# # #         # Render the figure to an array
# # #         # self.fig, self.ax = plt.subplots(figsize=(8,8), dpi=100)
# # #         self.fig.canvas.draw()
# # #         w, h = self.fig.canvas.get_width_height()
# # #         img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
# # #         # print("Width:", w, "Height:", h)
# # #         img = img.reshape((h*2, w*2, 3))

# # #         return img

# # #     def close(self):
# # #         if self.fig is not None:
# # #             plt.close(self.fig)
# # #             self.fig = None
# # #             self.ax = None