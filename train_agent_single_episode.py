
import sys
from game_globals import *
from CNNPlayer import CNNPlayer
from generateWorld import *
from main import *

NUMBER_GAMES = 3
MAX_FRAMES_PER_GAME = 10000

# Train an agent for a single episode
def trainEpisode(agent_filename="myagent.net", episode_number=1):
    print "RUNNING EPISODE #%d" % episode_number

    # Load the existing agent as a CNNPlayer
    agent = CNNPlayer(agent_filename)

    # create a new map
    mapname = "world%d.txt" % episode_number
    generateGameWorld(mapname)

    # start the game using the agent
    window = Window(width=WINDOW_SIZE, height=WINDOW_SIZE, caption='MindCraft', resizable=True, vsync=True)

    agent.setGame(window)  
    window.set_player(agent)
    window.set_game_frame_limit(MAX_FRAMES_PER_GAME)
    window.model.loadMap("maps" + os.sep + mapname)

    opengl_setup()
    pyglet.app.run()


if __name__ == "__main__":
    trainEpisode(sys.argv[1], int(sys.argv[2]))

        






