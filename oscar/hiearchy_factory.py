import json

from oscar.agent.commander.commander import Commander


def build_hierarchy(configuration_filename: str):
    """
    Builds a hierarchy of agents from a json file
    :param configuration_filename:
    :return: the general subordinates ?
    """
    with open(configuration_filename) as configuration_file:
        configuration = json.load(configuration_file)
    configuration_file.close()

    # Convert structure ids to integers (in place)
    configuration["structure"] = {int(k): [int(i) for i in v] for k, v in configuration["structure"].items()}

    # Check configuration file
    check_configuration(configuration)

    # Build family and return general's children
    general_subordinates = build_children(configuration, 0)
    return general_subordinates


def build_children(configuration, id):
    children_ids = configuration["structure"][id]
    children = []
    for child_id in children_ids:
        child_info = next((agent for agent in configuration["agents"] if agent["id"] == child_id))
        child_class = get_class(child_info["class_name"])
        if issubclass(child_class, Commander):
            little_children = build_children(configuration, child_id)
            child = child_class(little_children)
        else:
            child = child_class()
        children.append(child)
    return children


def check_configuration(configuration):
    valid = check_structure_acyclic(configuration["structure"])
    if not valid:
        raise CyclicStructureError("The loaded structure is cyclic")
    valid = valid and check_agents_are_known(configuration)
    if not valid:
        raise UndefinedAgentError("An agent is declared in the structure but undefined")
    return valid


def check_structure_acyclic(structure):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> check_structure_acyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> check_structure_acyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in structure.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return not any(visit(v) for v in structure)


def check_agents_are_known(configuration):
    known_agents = configuration["agents"]
    known_agents_id = [agent["id"] for agent in known_agents]

    agents_list = configuration["structure"].keys()
    for agent in agents_list:
        if agent not in known_agents_id:
            return False
    return True


def get_class(kls):
    parts = kls.split('.')
    class_module = ".".join(parts[:-1])
    m = __import__(class_module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


class CyclicStructureError(RuntimeError):
    """The loaded structure is cyclic"""


class UndefinedAgentError(RuntimeError):
    """An agent is declared in the structure but undefined"""
