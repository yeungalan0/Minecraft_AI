import random


def getRandomAction():
    a = Action()
    a.leftright_rotation = 1 - 2 * random.random()
    a.updown_rotation = 1 - 2 * random.random()
    a.forwardbackward_walk = random.choice([1, 0, -1])
    a.leftright_walk = random.choice([1, 0, -1])
    a.break_block = random.choice([True, False])
    
    return a

class Action:

    def __init__(self, break_block=False, updown_rot=0.0, leftright_rot=0.0, forwardback=0, leftright=0):
        
        # Pressing left button to break block
        self.break_block = break_block
        
        # Looking up or down
        self.updown_rotation = updown_rot
        
        # Looking right or left
        self.leftright_rotation = leftright_rot
        
        # Walking forward or backward (-1 is backward, 1 is forward, 0 is neither)
        self.forwardbackward_walk = forwardback
        
        # Strafing left or right
        self.leftright_walk = leftright
 

    def __str__(self):
        return "%.4f %.4f %d %d" % (self.updown_rotation, self.leftright_rotation, self.forwardbackward_walk, self.leftright_walk)







