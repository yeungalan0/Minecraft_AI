import copy

# A Sequence is an alternating list of Frames and Actions
# The first element should always be a Frame
# A Sequence is full with 4 frames and 3 actions  (F A F A F A F)
# New Sequences can be created from old ones by adding new actions and frames
# Creating a new Sequence from a full Sequence first removes the oldest Frame and Action

class Sequence:

    def __init__(self, first_frame=None):
        
        # A list of frames and actions in the sequence
        self.frames_actions = []
        
        if first_frame:
            self.addElement(first_frame)
        
    def addElement(self, element):
        self.frames_actions.append(element)
        
    def setElements(self, elements):
        self.frames_actions = elements
    
    
    def createNewSequence(self, new_element):
        
        new_frames_actions = []
        
        # check if there's still room in the sequence
        if self.isFull():  # Sequence is full
            # Copy all but the first two (copy is needed because we're making
            # a new Sequence object)
            new_frames_actions = copy.deepcopy(self.frames_actions[2:])

        # Now add the new action and frame
        new_frames_actions.append(new_element)
        #new_frames_actions.append(new_frame)
        
        # Make a new Sequence with the new_frames_actions list
        new_seq = Sequence()
        new_seq.setElements(new_frames_actions)
        
        return new_seq
        
    def isEmpty(self):
        return len(self) == 0
    
    def isFull(self):
        return len(self) == 7 # 4 Frames and 3 Actions
    
    def __len__(self):
        return len(self.frames_actions)

    def __str__(self):
        return "sequence"







