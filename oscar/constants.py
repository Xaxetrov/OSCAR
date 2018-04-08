from pysc2.lib import actions
from pysc2.lib import features

# Functions
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
SELECT_RECT = actions.FUNCTIONS.select_rect.id
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
SELECT_ARMY = actions.FUNCTIONS.select_army.id
ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
TRAIN_SCV_QUICK = actions.FUNCTIONS.Train_SCV_quick.id
TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
NO_OP = actions.FUNCTIONS.no_op.id

# Features
SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
SCREEN_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
PLAYER_SELF = 1
PLAYER_NEUTRAL = 3  # beacon/minerals
PLAYER_HOSTILE = 4
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index

# Layers
MINI_HEIGHT_MAP = features.MINIMAP_FEATURES.height_map.index
MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
MINI_CREEP = features.MINIMAP_FEATURES.creep.index
MINI_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
MINI_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
MINI_SELECTED = features.MINIMAP_FEATURES.selected.index
MINI_CAMERA = features.MINIMAP_FEATURES.camera.index

# Units ID
TERRAN_COMMAND_CENTER = 18
TERRAN_BARRACKS_ID = 21
TERRAN_SUPPLYDEPOT = 19
TERRAN_SUPPLYDEPOTLOWERED = 47
TERRAN_BUIDINGS = (TERRAN_COMMAND_CENTER, TERRAN_BARRACKS_ID, TERRAN_SUPPLYDEPOT, TERRAN_SUPPLYDEPOTLOWERED)

ALL_TERRAN_SUPPLYDEPOT = (TERRAN_SUPPLYDEPOT, TERRAN_SUPPLYDEPOTLOWERED)
TERRAN_SCV = 45
TERRAN_MARINE = 48
TERRAN_UNITS = (TERRAN_SCV, TERRAN_MARINE)

MINERAL_FIELD = 341
MINERAL_FIELD_750 = 483
RICH_MINERAL_FIELD = 146
RICH_MINERAL_FIELD_750 = 147
ALL_MINERAL_FIELD = (MINERAL_FIELD, MINERAL_FIELD_750, RICH_MINERAL_FIELD, RICH_MINERAL_FIELD_750)
VESPENE_GEYSER = 342
RICH_VESPENE_GEYSER = 344
ALL_VESPENE_GEYSER = (VESPENE_GEYSER, RICH_VESPENE_GEYSER)

# Exploration
UNEXPLORED_CELL = 0
EXPLORED_CELL = 1
VISIBLE_CELL = 2

# Parameters
NOT_QUEUED = [False]
QUEUED = [True]
NEW_SELECTION = [0]
SELECT_ALL = [0]

# Display
SCREEN_RESOLUTION = 84
MINIMAP_RESOLUTION = SCREEN_RESOLUTION
TILES_SIZE_IN_CELL = SCREEN_RESOLUTION / 21
TERRAN_COMMAND_CENTER_TILE_SIZE = 4.5
TERRAN_SCV_TILE_SIZE = 0.9
MINERAL_FIELD_TILE_SIZE = 1.5

# Others
MAX_COLLECTING_DISTANCE = SCREEN_RESOLUTION / 3

# Observation index
# 'player' indexes
PLAYER_ID = 0
MINERALS = 1
VESPENE = 2
FOOD_USED = 3
FOOD_CAP = 4
FOOD_USED_BY_ARMY = 5
FOOD_USED_BY_WORKERS = 6
IDLE_WORKER_COUNT = 7
ARMY_COUNT = 8
WARP_GATE_COUNT = 9
LARVA_COUNT = 10

# 'score_cumulative' indexes
KILLED_UNITS = 5
KILLED_BUILDINGS = 6
