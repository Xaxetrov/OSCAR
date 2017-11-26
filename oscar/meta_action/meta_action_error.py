class ActionError(RuntimeError):
    """
    Raise when something went wrong when doing an action.
    """


class NoValidSCVError(ActionError):
    """
    Raise when no valid scv can be selected according to defined rules.
    """


class NoValidBuildingLocationError(ActionError):
    """
    Raise when no valid location to build a building is found in the current screen.
    """


class NoUnitError(ActionError):
    """
    Raise when a unit or building is not present on screen whereas a function is asked to find one.
    """
