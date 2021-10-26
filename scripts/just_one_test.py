import math
import pdb

import ai2thor
import copy
import time
import random
import ai2thor.controller
import os

ENV_ARGS = dict(gridSize=0.25,
                width=224, height=224, agentMode='arm', fieldOfView=100,
                agentControllerType='mid-level',
                server_class=ai2thor.fifo_server.FifoServer,
                useMassThreshold = True, massThreshold = 10,
                autoSimulation=False, autoSyncTransforms=True,
                )
screen_size=224

ENV_ARGS['width'] = screen_size
ENV_ARGS['height'] = screen_size
ENV_ARGS['agentMode']='stretch'
# ENV_ARGS['commit_id']='8917d82118a663af77e76e0e5701fce186dcaec7'
ENV_ARGS['commit_id']='?'
ENV_ARGS['renderDepthImage'] = True

controller = ai2thor.controller.Controller(**ENV_ARGS)#, renderInstanceSegmentation=True)

controller.reset('FloorPlan15')
controller.step('MoveArm', position=dict(x=0,y=1,z=0))
