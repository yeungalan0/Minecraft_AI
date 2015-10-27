import time
import sys
import pyglet
import math
import random

from game_globals import *
import Action
from Player import Player
import h5py

class DataGatheringPlayer(Player):

    def __init__(self, dataset_filepath=""):
        Player.__init__(self)
        self.dataset_filepath = dataset_filepath
        self.h5 = h5py.File(dataset_filepath, 'w')
        self.dataset = self.h5.create_dataset("data", (16, WINDOW_SIZE, WINDOW_SIZE), maxshape=(None, WINDOW_SIZE, WINDOW_SIZE))
        self.dataset_index = 0
    
    def saveDataset(self):
        print("Saving dataset to ", self.dataset_filepath)
        self.dataset.resize((self.dataset_index, WINDOW_SIZE, WINDOW_SIZE))
        self.h5.close()
    
    def getDecision(self, current_frame):
        if self.game.world_counter % 100 == 0:
            print "*" * 40, self.game.world_counter, "%" * 40

        curr_action = Action.getRandomAction()

        # Actually perform the action in the game
        self.performAction(curr_action)
        
        data = current_frame.toCNNInput()
        #data = data.reshape((84,84))
        data *= 1/255.
        
        self.dataset[self.dataset_index] = data
        self.dataset_index += 1
        
        if self.dataset_index == len(self.dataset):
            print("RESIZING TO ", len(self.dataset)*2)
            self.dataset.resize((len(self.dataset)*2, WINDOW_SIZE, WINDOW_SIZE))


        


