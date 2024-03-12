from physics import Puck
import numpy as np
from tqdm import tqdm

class ai:
    
    def __init__(self, corners, striker: Puck, l1=None, l2=None):
        self.corners = corners
        self.striker = striker
        self.l1 = l1
        self.l2 = l2
        
    
    def unitize(vec):
        return vec / np.linalg.norm(vec)
    
    
    def check_extra_collisions(self, pucks, puck, pos, sv, relaxation=5):
        skip = False
        for secondary in pucks:
            if secondary.type not in [Puck.TEN, Puck.TWENTY, Puck.QUEEN]:
                continue
            if secondary == puck:
                continue
            del_x = secondary.x - np.array([pos, self.l1[1]])
            phi = np.arccos((del_x @ sv) / (np.linalg.norm(del_x) * np.linalg.norm(sv)))
            if np.linalg.norm(del_x) * np.sin(phi) < secondary.r + self.striker.r + relaxation:
                skip = True
                break
        
        return skip
        
        
    def optimize_shot(self, pucks):
        sol = np.inf            # solution variable to track optimization
        chosen_corner = None    # chosen corner for shot
        chosen_puck = None      # chosen puck to sink
        x_ = None               # position for striker
        v_ = None               # velocity for striker
        prob_sink = 0           # probability shot will work
        
        for corner in tqdm(self.corners):
            for puck in pucks:
                puck.v = np.zeros(2)    # ensure that puck's are standstill
                if puck.type in [Puck.TEN, Puck.TWENTY, Puck.QUEEN]:
                    # skip puck if it's not scorable
                    continue
                    
                v_des = ai.unitize(corner - puck.x)     # velocity that results in a score
                
                for pos in np.linspace(self.l1[0], self.l2[0], num=10):
                    # search over the available striker positions
                    for theta in np.linspace(0, 2 * np.pi, num=100):
                        # search over the available angles to hit
                        
                        # striker position the instant before collision
                        sx = (self.striker.r + puck.r) * np.array([np.cos(theta), np.sin(theta)]) + puck.x
                        sv = sx - np.array([pos, self.l1[1]]) # striker velocity to hit the puck
                        
                        # see if extra collisions will occur before desired collision
                        self.check_extra_collisions(pucks, puck, pos, sv)

                        
        
    def shot_configs(self, pucks, l1, l2):
        sol = np.inf
        test_v = None
        chosen_corner = None
        chosen_puck = None
        x_ = None
        v_ = None
        prob_sink = 0
        for corner in self.corners:
            for puck in tqdm(pucks):
                puck.v = np.zeros(2)
                if puck.type not in [Puck.TEN, Puck.TWENTY, Puck.QUEEN]:
                    continue
                v_des = corner - puck.x
                v_des = v_des / np.linalg.norm(v_des)
                s = self.striker
                for pos in np.linspace(l1[0], l2[0], num=10):
                    for theta in np.linspace(0, 2 * np.pi, num=100):
                        sx = (s.r + puck.r) * np.array([np.cos(theta), np.sin(theta)]) + puck.x
                        sv = sx - np.array([pos, l1[1]])
                        skip = False
                        for secondary in pucks:
                            if secondary.type not in [Puck.TEN, Puck.TWENTY, Puck.QUEEN]:
                                continue
                            if secondary == puck:
                                continue
                            del_x = secondary.x - np.array([pos, l1[1]])
                            phi = np.arccos((del_x @ sv) / (np.linalg.norm(del_x) * np.linalg.norm(sv)))
                            if np.linalg.norm(del_x) * np.sin(phi) < secondary.r + self.striker.r + 5:
                                skip = True
                                break
                        
                        if skip: break
                        
                        svn = sv / np.linalg.norm(sv)
                        n = sx - puck.x
                        un = n / np.linalg.norm(n)
                        ut = np.array([un[1], -un[0]])
                        
                        v1n = un @ sv
                        v1t = ut @ sv
                        v2n = un @ puck.v
                        v2t = ut @ puck.v
                        
                        v1n_ = (v1n * (s.m - puck.m) + 2 * puck.m * v2n) / (puck.m + s.m)
                        v2n_ = (v2n * (puck.m - s.m) + 2 * s.m * v1n) / (s.m + puck.m)
                        sv = v1n_ * un + v1t * ut
                        pv = v2n_ * un + v2t * ut
                        v_opt = pv / np.linalg.norm(pv)
                        skip = False
                        for c in self.corners:
                            if np.all(np.isclose(sv / np.linalg.norm(sv), (c - sx) / np.linalg.norm(c - sx), atol=0.05)):
                                skip = True
                                break
                        if skip: continue 
                        
                        if un @ svn > -0.55:
                            continue
                        
                        
                        for secondary in pucks:
                            if secondary.type not in [Puck.TEN, Puck.TWENTY, Puck.QUEEN]:
                                continue
                            if secondary == puck:
                                continue
                            
                            hyp = secondary.x - puck.x
                            phi = np.arccos((hyp @ v_opt) / (np.linalg.norm(hyp) * np.linalg.norm(v_opt)))
                            skip = False
                            if np.linalg.norm(hyp) * np.sin(phi) < secondary.r + self.striker.r + 5:
                                skip = True
                                break
                        if skip: continue
                        
                        if np.linalg.norm(v_opt - v_des) + 0.05 * un @ svn < sol:
                            sol = np.linalg.norm(v_opt - v_des) + 0.05 * un @ svn
                            prob_sink = 1 - np.linalg.norm(v_opt - v_des)
                            x_ = np.array([pos, l1[1]])
                            v_ = sx - np.array([pos, l1[1]])
                            test_v = un @ svn
                            chosen_corner = corner
                            chosen_puck = puck
                             
        print(test_v)
        print('Prob =', prob_sink)
        puck.v = np.zeros(2)
        if chosen_corner == None:
            while True:
                chosen_puck = np.random.choice(pucks)
                if chosen_puck.type not in [Puck.BOUNDARY, Puck.STRIKER]:
                    break
            x_ = np.array([np.random.random() * (l2[0] - l1[0]) + l1[0], l1[1]])
            v_ = 2 * (chosen_puck.x - x_)
        else:
            v_ = (np.linalg.norm(chosen_puck.x - chosen_corner) + np.linalg.norm(chosen_puck.x - x_)) * v_ / np.linalg.norm(v_)
        return x_, v_
    
    