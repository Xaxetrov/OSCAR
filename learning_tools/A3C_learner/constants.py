# constants
ENV = 'pysc2-mineralshard-v1'

RUN_TIME = 60
THREADS = 3
OPTIMIZERS = 2
THREAD_DELAY = 0.0001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

# constants that will be set in main
# brain = None  # brain is global in A3C
# NUM_STATE = 0
# NUM_ACTIONS = 0
# NONE_STATE = 0