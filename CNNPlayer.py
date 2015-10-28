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
from LogFile import LogFile
import os.path
import cPickle

MEMORY_SIZE = 1000000
STARTING_FRAMES = 100
GAMMA = 0.99
FRAMES_PER_SEQ = 4
STARTING_EPSILON = 0.1
EPSILON_UPDATE = 1.001
MAX_EPSILON = 0.9

class CNNPlayer(Player):

    def __init__(self, agent_filepath=""):
        Player.__init__(self)

        # Create the experience memory database
        if not os.path.exists(REPLAY_MEMORY_FILENAME):
            self.replay_memory = ReplayMemory()
        else:
            self.replay_memory = cPickle.load(open(REPLAY_MEMORY_FILENAME, 'r'))
        
        # Initialize the convolutional neural network
        self.network = MinecraftNet(agent_filepath)   
        self.ae_network = FeatureNet()
        
        # Probability of selecting non-random action
        self.epsilon = STARTING_EPSILON
        
        # The total number of frames this agent has been trained on
        # through all the minibatch training
        self.frames_trained = 0

        # Load old epsilon and frames learned values
        self.load()
            
        self.cnn_action_map = self.initActionMap()
        
        # The current and previous sequences of game frames and actions
        self.current_seq = None
        self.previous_seq = None
        self.previous_action = None
        
        # Event logging
        self.log = LogFile("run.log", True)
        #self.log.logMessage("INITIAL NETWORK PARAMS: %s" % str(self.network.solver.net.params['ip1'][0].data[...]))

        
        
    # Create a map of all the CNN's legal actions
    # We will be able to pick the best move from this list based on the CNN's output
    def initActionMap(self):
        actions = []
        
        # Populate with all 18 legal actions
        # (break_block, updown_rot, leftright_rot, forwardback, leftright)
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=0.0, forwardback=0, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=0.0, forwardback=1, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=0.0, forwardback=-1, leftright=0))  
        
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=0, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=1, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=-1, leftright=0))

        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=0, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=1, leftright=0))
        actions.append(Action.Action(False, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=-1, leftright=0))    

        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=0.0, forwardback=0, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=0.0, forwardback=1, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=0.0, forwardback=-1, leftright=0))  
        
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=0, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=1, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=AGENT_ROTATION_SPEED, forwardback=-1, leftright=0))

        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=0, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=1, leftright=0))
        actions.append(Action.Action(True, updown_rot=0.0, leftright_rot=-AGENT_ROTATION_SPEED, forwardback=-1, leftright=0))  
        
        return actions
    
    def getActionMapIndex(self, action):
        for i in range(len(self.cnn_action_map)):
            if action == self.cnn_action_map[i]:
                return i
        self.log.logError("ACTION %s NOT FOUND IN ACTION MAP" % str(action))
        sys.exit(1)
    
    
    def sequenceForward(self, seq):
        cnn_input = seq.toCNNInput()
        output = self.network.forward(cnn_input)
        return output
    
    def pickBestAction(self, seq):
        cnn_outputs = self.sequenceForward(seq)
        self.log.logMessage("REINFORCEMENT NET OUTPUT: " + str(cnn_outputs))
        
        max_output_index = 0
        max_output = cnn_outputs[0]
        for i in range(len(cnn_outputs)):
            if cnn_outputs[i] > max_output:
                max_output = cnn_outputs[i]
                max_output_index = i
                
        self.log.logMessage("BEST ACTION CHOSEN: %s" % str(self.cnn_action_map[max_output_index]))
        return self.cnn_action_map[max_output_index]
    
    def pickRandomAction(self):
        return random.choice(self.cnn_action_map)
    
    def load(self):
        if os.path.exists(CNNPLAYER_SAVE_FILENAME):
            f = open(CNNPLAYER_SAVE_FILENAME, 'r')
            tokens = f.read().split()
            self.epsilon, self.frames_trained = float(tokens[0]), int(tokens[1])
            f.close()
    
    
    def save(self):
        # Save the replay memory as a pickled file
        o = open(REPLAY_MEMORY_FILENAME, 'w')
        cPickle.dump(self.replay_memory, o)
        o.close()
        
        o = open(CNNPLAYER_SAVE_FILENAME, 'w')
        o.write("%.8f %d" % (self.epsilon, self.frames_trained))
        o.close()

        # Log the last network weights        
        #self.log.logMessage("FINAL NETWORK PARAMS: %s" % str(self.network.solver.net.params['ip1'][0].data[...]))
        
        
        
    # Train the agent's CNN on a minibatch of Experiences    
    def trainMinibatch(self):
        self.log.logMessage("TRAINING MINIBATCH")
        self.frames_trained += TRAINING_BATCH_SIZE
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
        self.network.train(BATCH_TRAINING_ITERATIONS) # train for a single iteration

    
    # Receive the agent's reward from its previous Action along with
    # a Frame screenshot of the current game state
    def getDecision(self, current_frame):
        self.log.logMessage("DECISION #%d in GAME FRAME #%d" % (self.actions_performed, self.game.world_counter))
        self.log.logMessage("TRAINED ON %d FRAMES" % (self.frames_trained))
       
        features = self.ae_network.encodeNumpyArray(current_frame.pixels)
        #self.log.logMessage("Current frame yields features: %s" % str(features))

        if self.previous_reward != 0:
            self.log.logMessage("GOT REWARD: %d" % self.previous_reward)
        self.total_score += self.previous_reward
                        
        # First frame of game
        if self.actions_performed == 0:
            self.actions_performed += 1
            self.previous_seq = Sequence(features)
            # print("FRAME SEQUENCE: {0}".format(self.previous_seq))
            curr_action = self.pickRandomAction()
            self.previous_seq = self.previous_seq.createNewSequence(curr_action)
            self.previous_action = curr_action
            # print("FIRST SEQUENCE: {0}".format(self.previous_seq))
            return
        
        
        # Should I make a random move?
        r = random.random()
            
        # Add on the current frame to the current sequence
        self.current_seq = self.previous_seq.createNewSequence(features)

        if r > self.epsilon or self.actions_performed < 4: #not self.current_seq.isFull():
            curr_action = self.pickRandomAction()
        else:
            # Run the CNN and pick the max output action
            curr_action = self.pickBestAction(self.current_seq)
            
        # Finally, add the chosen action to the current sequence
        self.current_seq = self.current_seq.createNewSequence(curr_action)
            
        # Actually perform the action in the game
        self.performAction(curr_action)
            
        new_experience = Experience(self.previous_seq, self.previous_action, self.previous_reward, self.current_seq)
        self.replay_memory.store(new_experience)
        self.previous_seq = self.current_seq

        if self.game.world_counter > STARTING_FRAMES and self.game.world_counter % BATCH_TRAINING_FREQUENCY == 0:
            self.trainMinibatch()
                
        # Remember the chosen Action since it will be required for the next iteration
        self.previous_action = curr_action
        
        if self.epsilon < MAX_EPSILON:
            self.epsilon *= EPSILON_UPDATE
            self.log.logMessage("UPDATED EPSILON: %.5f" % self.epsilon)


