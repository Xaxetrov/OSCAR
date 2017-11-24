from pysc2.lib import actions
from pysc2.lib import features

SCREEN_RESOLUTION = 84

# Functions
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
NO_OP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id

# Features
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
PLAYER_MINERAL_QUANTITY = 1
IDLE_WORKER_COUNT = 7
MINERAL_FIELD = 341
MINERAL_FIELD_750 = 483
RICH_MINERAL_FIELD = 146
RICH_MINERAL_FIELD_750 = 147
ALL_MINERAL_FIELD = (MINERAL_FIELD, MINERAL_FIELD_750, RICH_MINERAL_FIELD, RICH_MINERAL_FIELD_750)

# Units ID
TERRAN_COMMAND_CENTER = 18
TERRAN_SCV = 45
MINERAL_FIELD_LIST = (341, 483, 146, 147)
VESPENE_GEYSER_LIST = (344, 342)

# Parameters
PLAYER_SELF = 1
NOT_QUEUED = [False]
QUEUED = [True]
NEW_SELECTION = [0]

# Others
MAX_COLLECTING_DISTANCE = SCREEN_RESOLUTION / 3
TILES_SIZE_IN_CELL = SCREEN_RESOLUTION / 21
