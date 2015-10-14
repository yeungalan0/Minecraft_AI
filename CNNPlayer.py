import time
import pyglet
import math
import random

from game_globals import *
import Action
from Player import Player
from Sequence import Sequence
from Experience import Experience
from ReplayMemory import ReplayMemory
from caffe_minecraft import MinecraftNet

MEMORY_SIZE = 1000000
CNN_PARAMS = {"input_size":25000, "output_size":64}
STARTING_FRAMES = 100
EPSILON = 0.1
GAMMA = 0.99
FRAMES_PER_SEQ = 4

class CNNPlayer(Player):

    def __init__(self, agent_filepath=""):
        Player.__init__(self)

        # Create the experience memory database
        self.replay_memory = ReplayMemory()
        
        # Initialize the convolutional neural network
        self.network = MinecraftNet()   #initCNN(CNN_PARAMS)

        if agent_filepath!= '':
            self.network.load_model(agent_filepath)
            
        self.cnn_action_map = self.initActionMap()
        
        # The current and previous sequences of game frames and actions
        self.current_seq = None
        self.previous_seq = None
        self.previous_action = None
        
        
    # Create a map of all the CNN's legal actions
    # We will be able to pick the best move from this list based on the CNN's output
    def initActionMap(self):
        actions = []
        
        # for now do a hack to populate this map with all the same actions
        for i in range(CNN_PARAMS["output_size"]):
            actions.append(Action(False, 0.0, 0.25, 1, 0))
        
        return actions
    
    
    def sequenceForward(self, seq):
        cnn_input = seq.toCNNInput()
        return self.network.forward(cnn_input)
    
    def pickBestAction(self, seq):
        cnn_outputs = self.sequenceForward(seq)
        
        max_output_index = 0
        max_output = cnn_outputs[0]
        for i in range(len(cnn_outputs)):
            if cnn_outputs[i] > max_output:
                max_output = cnn_outputs[i]
                max_output_index = i
                
        return self.cnn_action_map[max_output_index]
        
        
    # Train the agent's CNN on a minibatch of Experiences    
    def trainMinibatch(self):
        experiences = replay_memory.get_random(LEARNING_SAMPLE_SIZE=32)
        dataset = []
        for experience in experiences:
             target = experience.reward + GAMMA * self.sequenceForward(experience.seq2)
             dataset.append((experience.seq1, target))
            
        #Do gradient descent to minimize   (target - network.runInput(experience.seq1)) ^ 2
        self.network.set_input_data(dataset)
        self.network.train(1) # train for a single iteration

    
    # Receive the agent's reward from its previous Action along with
    # a Frame screenshot of the current game state
    def getDecision(self, current_frame):
        #if self.previous_reward != 0:
        #    print "GOT POINTS:", self.previous_reward
        self.total_score += self.previous_reward
        
        # Luminance/scale/frame limiting/etc. preprocessing
        processed_frame = current_frame.preprocess()
                
        # Should I make a random move?
        r = random.random()
                    
        # First frame of game
        # print("PREVIOUS SEQUENCE: " + str(self.previous_seq) + "\n")
        if self.previous_seq == None:
            self.previous_seq = Sequence(processed_frame)
            # print("FRAME SEQUENCE: {0}".format(self.previous_seq))
            curr_action = Action.getRandomAction()
            self.previous_seq = self.previous_seq.createNewSequence(curr_action)
            # print("FIRST SEQUENCE: {0}".format(self.previous_seq))
            return
            
        # Add on the current frame to the current sequence
        self.current_seq = self.previous_seq.createNewSequence(processed_frame)

        if r < EPSILON:
            curr_action = Action.getRandomAction()
        else:
            # Run the CNN and pick the max output action
            # This will be self.network.pickBestAction(self.current_seq)
            # but it is random for now
            curr_action = self.pickBestAction(self.current_seq)
            
        # Finally, add the chosen action to the current sequence
        self.current_seq = self.current_seq.createNewSequence(curr_action)
            
        # Actually perform the action in the game
        self.performAction(curr_action)
            
        # The first time through there is no previous Sequence, so just add the first Frame
        new_experience = Experience(self.previous_seq, self.previous_action, self.previous_reward, self.current_seq)
        self.replay_memory.store(new_experience)
        # print(self.replay_memory.print_storage(0, 100))
        self.previous_seq = self.current_seq

        if self.game.world_counter > STARTING_FRAMES:
            self.trainMinibatch()
                
        # Remember the chosen Action since it will be required for the next iteration
        self.previous_action = curr_action



