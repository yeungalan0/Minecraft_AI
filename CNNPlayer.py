import time
import pyglet
import math
import random

from game_globals import *

import Action
from Player import Player
from Sequence import Sequence
from Experience import Experience


MEMORY_SIZE = 1000000
CNN_PARAMS = {"input_size":25000, "output_size":64}
STARTING_FRAMES = 100
EPSILON = 0.1
GAMMA = 0.99
FRAMES_PER_SEQ = 4

class CNNPlayer(Player):

    def __init__(self, agent_filename=""):
        Player.__init__(self)

        # Create the experience memory database
        #self.replay_memory = initReplayMemory(MEMORY_SIZE)
        self.replay_memory = []
        
        # Initialize the convolutional neural network
        if agent_filename == '':
            pass
            #self.network = initCNN(CNN_PARAMS)
        else:
            pass
            #self.network = self.loadNetwork(agent_filename)
        
        # The current and previous sequences of game frames and actions
        self.current_seq = None
        self.previous_seq = None
        
        self.previous_action = None
        
    # Load an existing CNN from a given file
    def loadNetwork(self, filename):
        pass
    
    
    # Train the agent's CNN on a minibatch of Experiences    
    def trainMinibatch(self):
        pass
        # Now spend time learning from past experiences
        # experiences = replay_memory.getRandomExperiences(LEARNING_SAMPLE_SIZE=32)
        # 
        # for experience in experiences:
        #     target = experience.reward + GAMMA * network.pickBestAction(experience.seq2)
            
            #Do gradient descent to minimize   (target - network.runInput(experience.seq1)) ^ 2
        
    
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
        if self.previous_seq == None:
            self.previous_seq = Sequence(processed_frame)
            curr_action = Action.getRandomAction()
            self.previous_seq = self.previous_seq.createNewSequence(curr_action)
            return
            
        # Add on the current frame to the current sequence
        self.current_seq = self.previous_seq.createNewSequence(processed_frame)

        if r < EPSILON:
            curr_action = Action.getRandomAction()
        else:
            # Run the CNN and pick the max output action
            # This will be self.network.pickBestAction(self.current_seq)
            # but it is random for now
            curr_action = Action.getRandomAction()
            
        # Finally, add the chosen action to the current sequence
        self.current_seq = self.current_seq.createNewSequence(curr_action)
            
        # Actually perform the action in the game
        self.performAction(curr_action)
            
        # The first time through there is no previous Sequence, so just add the first Frame
        new_experience = Experience(self.previous_seq, self.previous_action, self.previous_reward, self.current_seq)
        self.replay_memory.append(new_experience)
        self.previous_seq = self.current_seq

        if self.game.world_counter > STARTING_FRAMES:
            self.trainMinibatch()
                
        # Remember the chosen Action since it will be required for the next iteration
        self.previous_action = curr_action



