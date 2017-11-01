import numpy
import sys
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from random import randint

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_SCREEN = [0]


class CollectMineralShards(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""
    """Marines are controlled independently and chose their next target on the go."""

    targets = [None, None]
    positions = [None, None]
    selected = None
    neutral_x, neutral_y, player_x, player_y = None, None, None, None

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)

        # Find our units and targets
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        self.neutral_y, self.neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        self.player_y, self.player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

        # If there is no target or no unit, do nothing
        if not self.neutral_y.any() or not self.player_y.any():
            return actions.FunctionCall(_NO_OP, [])

        # If there is a unit selected
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            # Update units positions
            oldPositions = self.positions
            self.updateMarinePositions()

            # If selected unit has no target, provide it a new one.
            # Otherwise, select the other marine.
            if self.isSamePosition(oldPositions[self.selected], self.positions[self.selected]) or not self.targets[
                self.selected] or not self.isMineralShard(self.targets[self.selected]):
                # update position using previous target position
                if self.targets[self.selected]:
                    self.positions[self.selected] = self.targets[self.selected]

                self.getNewTarget(self.selected)

                # Order selected unit to move to it
                print("target (" + str(self.selected) + "): " + str(self.targets[self.selected]))
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, self.targets[self.selected]])
            else:
                # Select other marine
                self.selected = 1 - self.selected
                print("selecting: " + str(self.selected))
                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, self.positions[self.selected]])

        # If no entity is selected, select one
        else:
            self.updateMarinePositions()
            self.selected = 0
            return actions.FunctionCall(_SELECT_POINT, [_SCREEN, self.positions[self.selected]])

    # Returns whether there is a mineral shard at this position
    def isMineralShard(self, target):
        if not target:
            return False
        for nx, ny in zip(self.neutral_x, self.neutral_y):
            if nx == target[0] and ny == target[1]:
                return True
        return False

    def isSamePosition(self, p1, p2):
        return (p1[0] == p2[0] and p1[1] == p2[1])

    # Provides the best possible target, given the heuristic
    def getNewTarget(self, marine):
        best, max_score = None, None
        for p in zip(self.neutral_x, self.neutral_y):
            score = self.scoreTarget(p)
            if not max_score or score > max_score:
                best, max_score = p, score
        self.targets[marine] = best

    # Heuristic to estimate how good a target is for the selected marine.
    def scoreTarget(self, target):
        shardDist = numpy.linalg.norm(numpy.array(self.positions[self.selected]) - numpy.array(target))
        targetsDist = 0.
        if None not in self.targets:
            targetsDist = numpy.linalg.norm(
                numpy.array(self.targets[self.selected]) - numpy.array(self.targets[1 - self.selected]))
        return (targetsDist - 2 * shardDist)

    # Estimates new marines positions, using previous positions and points
    def updateMarinePositions(self):
        marinesPoints = zip(self.player_x, self.player_y);
        extremes = self.getMaxDistancePoints(marinesPoints, marinesPoints)

        centroid0 = []
        centroid1 = []

        for p in zip(self.player_x, self.player_y):
            d0 = numpy.linalg.norm(numpy.array(p) - numpy.array(extremes[0]))
            d1 = numpy.linalg.norm(numpy.array(p) - numpy.array(extremes[1]))
            if d0 < d1:
                centroid0.append(p)
            else:
                centroid1.append(p)

        centroidCenter0 = numpy.mean(centroid0, 0)
        centroidCenter1 = numpy.mean(centroid1, 0)

        if None in self.positions:
            self.positions = [centroidCenter0, centroidCenter1]
        else:
            d0 = numpy.linalg.norm(numpy.array(self.positions[0]) - numpy.array(centroidCenter0))
            d1 = numpy.linalg.norm(numpy.array(self.positions[1]) - numpy.array(centroidCenter0))

            if d0 < d1:
                self.positions = [centroidCenter0, centroidCenter1]
            else:
                self.positions = [centroidCenter1, centroidCenter0]

    def getMaxDistancePoints(self, l1, l2):
        furthest, max_dist = None, None
        for p1 in zip(self.player_x, self.player_y):
            for p2 in zip(self.player_x, self.player_y):
                dist = numpy.linalg.norm(numpy.array(p1) - numpy.array(p2))
                if not max_dist or dist > max_dist:
                    max_dist = dist
                    furthest = [p1, p2]
        return furthest
