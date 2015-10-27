from DataGatheringPlayer import DataGatheringPlayer
from main import *

if __name__ == '__main__':
    world = "maps/test.txt"
    total_game_frames = 200
    dataset_file = "datasets/gathered.hdf5"  # warning...an old dataset with the same name will be overwritten!
    main(DataGatheringPlayer(dataset_file), world, total_game_frames)
