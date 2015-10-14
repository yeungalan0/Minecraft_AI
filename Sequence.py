import copy

# A Sequence is an alternating list of Frames and Actions
# The first element should always be a Frame
# A Sequence is full with 4 frames and 4 actions  (F A F A F A F A)
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
        # DEEP COPY VERSION INCASE FRAMES OR ACTIONS CHANGE
        if self.isFull():  # Sequence is full
            # Copy all but the first two (copy is needed because we're making
            # a new Sequence object)
            new_frames_actions = copy.deepcopy(self.frames_actions[2:])
        else:
            new_frames_actions = copy.deepcopy(self.frames_actions)

        # THINK ABOUT THIS IF IT TAKES TOO LONG
        # if self.isFull():  # Sequence is full
        #     # Copy all but the first two (copy is needed because we're making
        #     # a new Sequence object)
        #     new_frames_actions = self.frames_actions[2:]
        # else:
        #     new_frames_actions = self.frames_actions[:]
        
        # Now add the new action and frame
        new_frames_actions.append(new_element)
        #new_frames_actions.append(new_frame)
        
        # Make a new Sequence with the new_frames_actions list
        new_seq = Sequence()
        new_seq.setElements(new_frames_actions)
        
        return new_seq

    def toCNNInput(self):
        frames = []
        for x in range(0, len(self), 2):
            frames += self.frames_actions[x].toCNNInput()
        return frames
            
    def isEmpty(self):
        return len(self) == 0
    
    def isFull(self):
        return len(self) == 8 # 4 Frames and 4 Actions
    
    def __len__(self):
        return len(self.frames_actions)

    def __str__(self):
        if self.isFull():
            return "SEQUENCE:(Frame1, {0}, Frame2, {1}, Frame3, {2}, Frame4, {3})".format(self.frames_actions[1], self.frames_actions[3], self.frames_actions[5], self.frames_actions[7])
            # return_string = ""
            # for element in self.frames_actions:
            #     return_string += str(element)
            # return return_string
        else:
            sequence_str = ""
            for x in range(len(self)):
                # TODO: CHANGE THIS TO A ZERO
                if x % 2 == 0:
                    sequence_str += "Frame{0}, ".format(x)
                else:
                    sequence_str += str(self.frames_actions[x])
            return "This sequence is not full! LENGTH: {0}, SEQUENCE: ({1})".format(len(self), sequence_str)







