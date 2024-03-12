import enum

# make callback that checks if armed/ready are good to go
# put the mode switching in this callback

def splinetime(p0, pf, v0, vf):
        pass
    
def spline(Tmove, p0, pf, v0, vf, a0, af):
    pass

class Move:
    
    class Mode(enum):
        pass
        
    def __init__(self, node):
        self.mode = None
        self.done = False
        self.ready = None # cb func to read ready topic
        self.armed = None # cb func to ready armed topic
        
    def cb_ready(self, msg):
        self.ready = msg.data
    
    def cb_armed(self, msg):
        self.armed = msg.data
    
    def step(self, t):
        raise NotImplementedError()
    
    
class Grab(Move):
    class Mode(enum):
        TO_OBJ = 0
        GRAB = 1
        
    def __init__(self, node):
        super().__init__(node)
        self.mode = Grab.Mode.TO_OBJ
    
    def step(self, t):
        if self.mode == Grab.Mode.TO_OBJ:
            pass
        elif self.mode == Grab.Mode.GRAB:
            pass
        else:
            raise Exception('Grab: unknown mode found')
        
        if self.mode == Grab.Mode.TO_OBJ and self.armed:
            self.mode = Grab.Mode.GRAB


class Drop(Move):
    class Mode(enum):
        TO_PT = 0
        RELEASE = 1
        
    def __init__(self, node):
        super().__init__(node)
        self.mode = Drop.Mode.TO_PT
    
    def step(self, t):
        if self.mode == Drop.Mode.TO_PT:
            pass
        elif self.mode == Drop.Mode.RELEASE:
            pass
        else:
            raise Exception('Grab: unknown mode found')
        
        if self.mode == Drop.Mode.TO_PT and self.armed:
            self.mode = Drop.Mode.RELEASE
    

class Strike(Move):
    class Mode(enum):
        TO_POSE = 0
        STRIKE = 1
        
    def __init__(self, node):
        super().__init__(node)
        
    def step(self, t):
        pass
    

class Go(Move):
    class Mode(enum):
        MOVE = 0
        
    def __init__(self, node):
        super().__init__(node)
        
    def step(self, t):
        pass
    
    