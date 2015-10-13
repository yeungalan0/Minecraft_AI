
from train_agent_single_episode import *
from subprocess import call

if __name__ == "__main__":
    agent_filename = sys.argv[1]
    number_episodes = int(sys.argv[2])
    
    for episode in range(1, number_episodes+1):
        args = ["python", "train_agent_single_episode.py", agent_filename, str(episode)]
        call(args)

