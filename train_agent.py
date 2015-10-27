
from train_agent_single_episode import *
from game_globals import *
from subprocess import call

if __name__ == "__main__":
    import hdf5tools.make_dataset
    
    agent_filename = REINFORCEMENT_MODEL
    number_episodes = int(sys.argv[1])
    
    for episode in range(1, number_episodes+1):
        args = ["python", "train_agent_single_episode.py", agent_filename, str(episode)]
        call(args)

