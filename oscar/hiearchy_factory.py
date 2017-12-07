import json


def build_hierarchy(configuration_filename: str):
    """
    Builds a hierarchy of agents from a json file
    :param configuration_filename:
    :return: the general subordinates ?
    """
    with open(configuration_filename) as configuration_file:
        configuration = json.load(configuration_file)

    # Convert structure ids to integers (in place)
    configuration["structure"] = {int(k): [int(i) for i in v] for k, v in configuration["structure"].items()}
    check_configuration(configuration)

    # Create a maintained set of instantiated agents
    instantiated = {}

    # build shared objects to be associated with agents later
    shared_objects = build_shared_list(configuration)

    # Build family and return general's children
    general_subordinate = build_agent(configuration, instantiated, shared_objects, 0)
    return general_subordinate

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------BUILD AGENT-------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def build_agent(configuration, instantiated, shared_objects, agent_id):
    """
    Builds an agent from a configuration.
    Recursive function : builds the children before the agent itself
    :param configuration: the configuration file
    :param instantiated: a maintained set of instanciated agents
    :param shared_objects: a list of objects potentially shared among some agents
    :param agent_id: The id of the agent in the "structure" and "agents" part of the configuration
    :return: the built agent
    """
    agent_info = get_agent_information(configuration["agents"], agent_id)
    agent_class = get_class(agent_info["class_name"])
    agent_arguments = agent_info["arguments"]

    # First check if the agent (and its children) has already been instantiated by another commander
    if agent_id in instantiated:
        return instantiated[agent_id]

    # Generate children
    try:
        children_ids = configuration["structure"][agent_id]
    except KeyError:
        children_ids = []
    if len(children_ids) != 0:
        children = []
        for child_id in children_ids:
            children.append(build_agent(configuration, instantiated, shared_objects, child_id))
        if len(children) > 0:
            agent_arguments["subordinates"] = children

    # Create the thing
    agent = agent_class(**agent_arguments)
    instantiated[agent_id] = agent

    # associate shared objects
    for obj in shared_objects:
        if agent_id in obj["shared_with"]:
            associate_shared_objects(agent, obj)

    return agent


def get_agent_information(agents, agent_id):
    """
    Finds an agent in a list of agents, by id
    :param agents: the list to look into
    :param agent_id: the id to look for
    :return: the agent (if found)
    """
    for agent in agents:
        if agent["id"] == agent_id:
            return agent
    raise ValueError("Agent of id {0} is not in agent_information".format(agent_id))


# ----------------------------------------------------------------------------------------------- #
# -----------------------------------BUILD SHARED OBJ-------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def build_shared_list(configuration):
    """
    Build a list of shared objects
    :param configuration: the config file
    :return: a list of dict with keywords (name, object, shared_with)
    """
    shared_objects = []

    for obj in configuration["shared"]:
        shared_objects.append(build_shared_object(obj))

    return shared_objects


def build_shared_object(object_informations):
    """

    :param object_informations: the config file part representing a shared object
    :return: a dict with keywords (name, object, shared_with) where :
        - name : the key to recognise the object
        - object : the object instantiated
        - shared_with : list of indices, indicate which agents can see the object
    """
    shared = dict()
    shared["name"] = object_informations["name"]
    shared["object"] = get_class(object_informations["class_name"])(**object_informations["arguments"])
    shared["shared_with"] = object_informations["shared_with"]
    return shared


def associate_shared_objects(agent, obj):
    """
    Wrap the association of shared objects for later checks
    :param agent:
    :param obj:
    :return:
    """

    agent.add_shared(obj["name"], obj['object'])


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------CHECK CONFIG------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def check_configuration(configuration):
    """
    runs various integrity tests on a given structured configuration
    :param configuration:
    :return: if all checks are passed
    """
    if not check_structure_acyclic(configuration["structure"]):
        raise CyclicStructureError("The loaded structure is cyclic")
    if not check_agents_are_known(configuration):
        raise UndefinedAgentError("An agent is declared in the structure but undefined")
    return True


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
    """
    Check if all keys of the configuration "structure" are defined in the "agents" section
    :param configuration: the object group to look at
    :return: if all agents are well defined
    """
    known_agents = configuration["agents"]
    known_agents_id = [agent["id"] for agent in known_agents]

    agents_list = configuration["structure"].keys()
    for agent in agents_list:
        if agent not in known_agents_id:
            return False
    return True


def get_class(kls):
    """
    returns the class corresponding to a string
    :param kls: the given string pointing to the class from the working directory
    :return: the class as an object
    """
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
