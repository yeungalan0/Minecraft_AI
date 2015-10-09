# Minecraft Reinforcement Learning 

## Replay Memory
  * DeepMind uses 1 million memories
  * Each "memory" is a SARS' sequence
    
    
## Deep Convolutional Network
  * Input
    * Input will be pixel grayscale luminance values from game screenshots
    * Screenshots will be captured every k game ticks (to reduce computational overhead)
    * Screenshots are scaled down to some managable size (DeepMind uses 84x84)
    * Several (DeepMind uses 4) frames are combined to produce a network input of 84x84x4
    
  * Structure
    * The deep convolutional network has several layers for extracting features from the input
    * Layer #1 has filters for 8x8 sliding windows with step size of 4, followed by rectifier
    * Layer #2 has filters for 4x4 sliding windows with step size of 2, followed by rectifier
    * Layer #3 has 256, fully-connected rectifier nodes
    * The output layer has one node for each possible agent action (action perfomed is taken as max)

## Algorithm

```
replay_memory = initReplayMemory(MEMORY_SIZE)
network = initCNN(params)

Run N episodes (games run until game over or fixed time limit (50million frames total over all episodes)):
    # start the game and get the initial state (screenshot)
    curr_state = initGame()
    
    # Luminance/scale/frame limiting/etc. preprocessing
    curr_seq = preprocess(curr_state)
    
    Run game for T timesteps:
        Generate a random number, r, from 0 to 1.0
        if r < epsilon:
            curr_action = selectRandomAction()
        else:
            # Run the CNN and pick the max output action
            curr_action = network.pickBestAction(curr_seq)  
            
        # Actually perform the action in the game
        curr_reward, next_state = executeAction(curr_action)
        
        # Store a copy for adding a sequence to the replay memory
        prev_seq = seqCopy(curr_seq)
        
        curr_seq = buildNewSequence(curr_seq, curr_action, next_state)
        curr_seq = preprocess(curr_seq)
        
        # Remember this event in the big memory store
        replay_memory.addMemory((prev_seq, curr_action, curr_reward, curr_seq))
        
        # Now spend time learning from past experiences
        experiences = replay_memory.getRandomExperiences(LEARNING_SAMPLE_SIZE=32)
        
        for experience in experiences:
            seq1, action, reward, seq2 = experience   # Unpack experience tuple
            target = reward + GAMMA * network.pickBestAction(seq2)
            
            Do gradient descent to minimize   (target - network.runInput(seq1)) ^ 2
```
 

## Installing CAFFE
  The official instructions are [here](http://caffe.berkeleyvision.org/installation.html).
  
  Some suggestions:
  * If you have a GPU make sure you install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) first. 
  * Install [anaconda](https://www.continuum.io/downloads#_unix) before running `make pycaffe`. Set your anaconda path and change the PYTHON_LIB variable in the Makefile.config.
  * The instructions say to use python 2.7 or 3.3. I've used 2.7. 
  * The instructions aren't clear but you need the dev versions of `protobuf`, `glog`, `gflags`, `hdf5`, `leveldb`, `snappy`, and `lmdb`. Caffe needs to access the header files. 
  * I used OpenBLAS since they said it may offer a speedup. In the Makefile.config make sure you change `BLAS=atlas` to `BLAS=open`.
  * Build Caffe with `make all -j4`, `make test -j4`, `make runtest`. (Using `make [target] -j4` will compile with 4 threads to speed things up a bit).
  * To build the python bindings you'll have to run `make pycaffe`. If you get an error that says `Python.h` not found update the PYTHON_INCLUDE variable to also have `/usr/include/python2.7`.
  * Set your PYTHON_PATH to point to the caffe python direction `PYTHONPATH=/path/to/caffe/python:$PYTHONPATH`. You could do this in the .bashrc file and then source it. 
  * Try the examples and let me know if you have questions.





