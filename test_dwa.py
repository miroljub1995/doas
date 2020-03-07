from api.dwa.dwa import DWA, simulate_dwa, rad_to_deg
import math
from time import time


dwa = DWA()

obs = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5)]
goal = (2, -4)
curr = (2, 10)
curr_angle = -math.pi / 2.0
# curr_angle = 0.0

start = time()
angle = dwa.calculate_angle(goal, obs, curr)
end = time()
print("Angle: {} (from [-1, 1]), time: {}".format(angle, end - start))

simulate_dwa(goal, curr_angle, 0.9, curr=curr, obs=obs)
