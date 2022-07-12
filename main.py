from ENV import *
from GUI import *
from AGT import *
from Trainer import *

import time

def main():
    environtment = env()
    agent = agt(environtment)
    trainer_ = trainer()
    gui = GUI(environtment , agent, trainer_)
    return

main()

