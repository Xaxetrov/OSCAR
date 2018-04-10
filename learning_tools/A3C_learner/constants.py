# constants
ENV = 'general-learning-v0'

RUN_TIME = 60 * 5
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.9
EPS_STOP = 0.1
EPS_STEPS = 175000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

LOG_FILE = "learning_tools/learning_nn/general-learning-v0.csv"

# constants that will be set in main
# brain = None  # brain is global in A3C
# NUM_STATE = 0
# NUM_ACTIONS = 0
# NONE_STATE = 0
