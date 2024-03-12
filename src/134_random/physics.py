import numpy as np

class Engine:
    
    def __init__(self, pucks, corners, dt=0.001, mu=2.5):
        self.pucks = pucks
        self.dt = dt
        self.mu = mu
        self.corners = np.array(corners)
        
    def step(self):
        self.goal()
        self.friction()
        self.check_collisions()
        
        for puck in self.pucks:
            puck.step(self.dt)
            
    def friction(self):
        for p in self.pucks:
            p.v = p.v - self.mu * p.v * self.dt
            
    def goal(self):
        i = 0
        while i < len(self.pucks):
            for corner in self.corners:
                if np.linalg.norm(self.pucks[i].x - corner) < 10:
                    self.pucks.remove(self.pucks[i])
                    print('remove')
                    i -= 1
                    continue
            i += 1
                
    
    def check_collisions(self):
        for i, p in enumerate(self.pucks):
            for j, q in enumerate(self.pucks):
                if j <= i: continue
                elif p.type == Puck.BOUNDARY and q.type == Puck.BOUNDARY: continue
                elif p.collided(q):
                    pv = p.v.copy()
                    qv = q.v.copy()
                    p.v = -p.v
                    q.v = -q.v
                    while p.collided(q):
                        p.step(self.dt / 100)
                        q.step(self.dt / 100)
                    
                    p.v = -p.v
                    q.v = -q.v
                    
                    n = p.x - q.x
                    un = n / np.linalg.norm(n)
                    ut = np.array([un[1], -un[0]])
                    
                    v1n = un @ p.v
                    v1t = ut @ p.v
                    v2n = un @ q.v
                    v2t = ut @ q.v
                    
                    v1n_ = (v1n * (p.m - q.m) + 2 * q.m * v2n) / (q.m + p.m)
                    v2n_ = 0 * un @ ((q.v * (q.m - p.m) + 2 * p.m * p.v) / (p.m + q.m)) + 1*(v2n * (q.m - p.m) + 2 * p.m * v1n) / (p.m + q.m)
                    p.v = v1n_ * un + v1t * ut
                    q.v = v2n_ * un + v2t * ut
                    
                    
                    # M = p.m + q.m
                    # p.v, q.v = (p.m - q.m) / M * p.v + 2 * q.m / M * q.v, \
                    #             2 * p.m / M * p.v + (q.m - p.m) / M * q.v
                    
                    # p.v = p.v - 2 * q.m / (p.m + q.m) * ((p.v - q.v) @ (p.x - q.x) / np.linalg.norm(p.x - q.x)) * (p.x - q.x)
                    # q.v = q.v - 2 * p.m / (p.m + q.m) * ((q.v - p.v) @ (q.x - p.x) / np.linalg.norm(q.x - p.x)) * (q.x - p.x)
                p.update_collision(q)


    
class Puck:
    
    TEN = 1
    TWENTY = 2
    QUEEN = 3
    STRIKER = 4
    BOUNDARY = 5
    
    def __init__(self, x0, v0=[0,0], m=1, r=7, type=None):
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.m = m 
        self.r = r
        self.type = type
        self.colliding = set()
        
    def step(self, dt):
        # foward integrate
        self.x = dt * self.v + self.x
    
    def collided(self, other):
        # check collision
        return np.linalg.norm(self.x - other.x) < self.r + other.r #and (other not in self.colliding or self not in other.colliding)
        
    def update_collision(self, other):
        if np.linalg.norm(self.x - other.x) < self.r + other.r:
            
            self.colliding.add(other)
            other.colliding.add(self)

            return True
        else:
            if other in self.colliding:
                self.colliding.remove(other)
            
            if self in other.colliding:
                other.colliding.remove(self)
            return False