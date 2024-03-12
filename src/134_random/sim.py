import cv2
import numpy as np
import time
from physics import *
from player import ai

frame = cv2.imread("carrom.jpg", cv2.IMREAD_COLOR)
WIDTH, HEIGHT,_ = frame.shape
PUCK_MASS = 1
PUCK_RADIUS = 7
CENTER = np.array([WIDTH // 2, HEIGHT // 2]) 
COLOR = {Puck.TEN : (0, 0, 0),
         Puck.TWENTY : (42, 140, 245),
         Puck.QUEEN : (0, 100, 255),
         Puck.STRIKER : (255, 0, 255)}
CORNERS = [[33,33],
           [WIDTH - 33, 33],
           [33, HEIGHT - 33],
           [WIDTH - 33, HEIGHT - 33]]
circ = lambda r, i, n, b: r * np.array([np.cos(i / n * 2 * np.pi + b), np.sin(i / n * 2 * np.pi + b)])

tens = []
twenties = []
boundary = []
queen = Puck(x0=np.array([WIDTH // 2 + 0 * 100 , HEIGHT // 2 +0*140]), type=Puck.QUEEN)
for i in range(3): tens.append(Puck(x0=CENTER + circ(17, i, 3, 0),type=Puck.TEN))
    
for i in range(3): twenties.append(Puck(x0=CENTER + circ(17, i, 3, np.pi / 3),type=Puck.TWENTY))
    
for i in range(6): tens.append(Puck(x0=CENTER + circ(34, i, 6, 0),type=Puck.TEN))
    
for i in range(6): twenties.append(Puck(x0=CENTER + circ(34, i, 6, np.pi / 6),type=Puck.TWENTY))

for i in range(4):
    boundary.append(Puck(x0=CENTER + circ(10000 + HEIGHT // 2 - 27, i, 4, 0),
                    m=1000,
                    r=10000,
                    type=Puck.BOUNDARY)) 
    
striker = Puck(x0=[WIDTH // 2 - 100, HEIGHT // 2 + 123],
               m=1.25,
               r=10,
               v0=np.zeros(2),
               type=Puck.STRIKER)

bodies = [queen] + tens + twenties + [striker] + boundary
comp = ai(corners=CORNERS, striker=striker)
# x0, v0 = comp.shot_configs(bodies, [WIDTH // 2 - 100, HEIGHT // 2 + 123], [WIDTH // 2 + 100, HEIGHT // 2 + 123])

# striker.x = x0
# striker.v = 3 * v0
eng = Engine(pucks=bodies, corners=CORNERS)
t0 = time.time()
begin = True

while True:
    # setup the board
    frame = cv2.imread("carrom.jpg", cv2.IMREAD_COLOR)
    
    # iterate the physics engine
    eng.dt = 0.01
    # t0 = time.time()
    eng.step()
    
    # visualize
    for puck in eng.pucks:
        if puck.type != Puck.BOUNDARY:
            cv2.circle(frame, tuple(puck.x.astype(int)), puck.r, COLOR[puck.type], cv2.FILLED)
    cv2.imshow("Image", frame)
    
    if begin:
        key = cv2.waitKey(0)
        # t0 = time.time()
        if key == ord('s'):
            begin = False
    else:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            x0, v0 = comp.shot_configs(eng.pucks, [WIDTH // 2 - 100, HEIGHT // 2 + 123], [WIDTH // 2 + 100, HEIGHT // 2 + 123])
            # comp.shot_config_leaned(eng.pucks, [WIDTH // 2 - 100, HEIGHT // 2 + 123], [WIDTH // 2 + 100, HEIGHT // 2 + 123])
            striker.x = x0
            striker.v = 3 * v0
            # t0 = time.time()
            # begin = True