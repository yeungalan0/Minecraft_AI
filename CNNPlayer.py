import time
import sys
import pyglet
import math
import random

from game_globals import *
import Action
from Player import Player
from Sequence import Sequence
from Experience import Experience
from ReplayMemory import ReplayMemory
from caffe_minecraft_hdf5 import MinecraftNet
from FeatureNet import FeatureNet

MEMORY_SIZE = 1000000
CNN_PARAMS = {"input_size":50, "output_size":18}
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
        self.ae_network = FeatureNet()

        if agent_filepath != "":
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
        for i in range(CNN_PARAMS["output_size"]/2):
            actions.append(Action.Action(False, 0.0, 0.25, 0, 0))
        for i in range(CNN_PARAMS["output_size"]/2, CNN_PARAMS["output_size"]):
            actions.append(Action.Action(True, 0.0, 0.25, 1, 0))
        
        return actions
    
    def getActionMapIndex(self, action):
        for i in range(len(self.cnn_action_map)):
            if action == self.cnn_action_map[i]:
                return i
        print("ACTION %s NOT FOUND IN ACTION MAP" % str(action))
        sys.exit(1)
    
    
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
    
    def pickRandomAction(self):
        return random.choice(self.cnn_action_map)
        
        
    # Train the agent's CNN on a minibatch of Experiences    
    def trainMinibatch(self):
        experiences = self.replay_memory.get_random(TRAINING_BATCH_SIZE)
        inputs = []
        labels = []
        for experience in experiences:
            cnn_outputs = self.sequenceForward(experience.curr_seq)
            #best_action = self.pickBestAction(experience.curr_seq)
            target_vector = []
            for act in cnn_outputs:
                #act = cnn_outputs[act_id]
                act_target = experience.curr_reward + GAMMA * act
                target_vector.append(act_target)
            #target = experience.curr_reward + GAMMA * best_action_output
            inputs.append(experience.prev_seq)
            labels.append(target_vector)
            #dataset.append((experience.prev_seq, target))
            
        #Do gradient descent to minimize   (target - network.forward(experience.prev_seq)) ^ 2
        # print("INPUTS:", inputs)
        # print("LABELS:", labels)
        #self.network.set_input_data(inputs, labels)
        self.network.set_train_input_data(inputs, labels)
        self.network.train(1) # train for a single iteration

    
    # Receive the agent's reward from its previous Action along with
    # a Frame screenshot of the current game state
    def getDecision(self, current_frame):
        
        features = self.ae_network.encodeNumpyArray(current_frame.pixels)
        
        if self.game.world_counter % 10 == 0:
            print "*" * 40, self.game.world_counter, "%" * 40
        #if self.previous_reward != 0:
        #    print "GOT POINTS:", self.previous_reward
        self.total_score += self.previous_reward
        
        # Luminance/scale/frame limiting/etc. preprocessing
        #processed_frame = current_frame.preprocess()
                
        # Should I make a random move?
        r = random.random()
                    
        # First frame of game
        # print("PREVIOUS SEQUENCE: " + str(self.previous_seq) + "\n")
        if self.previous_seq == None:
            self.previous_seq = Sequence(features)
            # print("FRAME SEQUENCE: {0}".format(self.previous_seq))
            curr_action = self.pickRandomAction()
            self.previous_seq = self.previous_seq.createNewSequence(curr_action)
            self.previous_action = curr_action
            # print("FIRST SEQUENCE: {0}".format(self.previous_seq))
            return
            
        # Add on the current frame to the current sequence
        self.current_seq = self.previous_seq.createNewSequence(features)

        if r < EPSILON or not self.current_seq.isFull():
            curr_action = self.pickRandomAction()
        else:
            # Run the CNN and pick the max output action
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

        if self.game.world_counter > STARTING_FRAMES and self.game.world_counter % 40 == 0:
            print("TRAINING MINIBATCH")
            self.trainMinibatch()
                
        # Remember the chosen Action since it will be required for the next iteration
        self.previous_action = curr_action


