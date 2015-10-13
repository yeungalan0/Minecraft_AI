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

    def __init__(self):
        
        # Pressing left button to break block
        self.break_block = False
        
        # Pressing right button to place a new block
        self.place_block = False
        
        # Jumping
        self.jumping = False
        
        # Looking up or down
        self.updown_rotation = 0.0
        
        # Looking right or left
        self.leftright_rotation = 0.0
        
        # Walking forward or backward
        self.forwardbackward_walk = 0  # -1 is backward, 1 is forward, 0 is neither
        
        # Strafing left or right
        self.leftright_walk = 0  # -1 is backward, 1 is forward, 0 is neither
 

    def __str__(self):
        return "%.4f %.4f %d %d" % (self.updown_rotation, self.leftright_rotation, self.forwardbackward_walk, self.leftright_walk)







