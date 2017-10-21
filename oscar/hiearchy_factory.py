import json


def build_hierarchy(configuration_filename):
    with open(configuration_filename) as configuration_file:
        configuration = json.load(configuration_file)


def check_structure(configuration):
    valid = check_structure_acyclic(configuration["structure"])
    check_agents_are_known()
    return valid


def check_structure_acyclic(structure):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return True
        visited.add(vertex)
        path.add(vertex)
        for neighbour in structure.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return False
        path.remove(vertex)
        return True

    return not any(visit(v) for v in structure)


def check_agents_are_known(configuration):
    known_agents = configuration["agents"]
    known_agents_id = [agent["id"] for agent in known_agents]

    agents_list = configuration["structure"].keys()
    for agent in agents_list:
        if agent not in known_agents_id:
            return False
    return True

