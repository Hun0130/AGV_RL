from ENV import *
from GUI import *
from AGT import *
from Trainer import *

import sys

def main():
    
    # ================ Training parameters ===================
    # episode : the number of step of 1 episode
    episode = 1500

    # epoch : whole number of epoch with training
    epoch = 1000

    # learning rate
    learning_rate = 0.001

    # Gamma
    gamma = 0.9

    # batch_size
    batch_size = 30

    # training interval
    training_interval = 30

    # model name to be saved
    model_name = ""
# ========================================================
    
    if (len(sys.argv) > 5):
        episode = int(sys.argv[1])
        epoch = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
        gamma = float(sys.argv[4])
        batch_size = int(sys.argv[5])
        training_interval = int(sys.argv[6])
        
    environtment = env(episode)
    agent = agt(environtment, episode, epoch, learning_rate, gamma, batch_size, training_interval, model_name)
    trainer_ = trainer()
    gui = GUI(environtment , agent, trainer_)
    return

main()

