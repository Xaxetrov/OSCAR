import json

from oscar.env.shared_objects import SharedObjects


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

    # Setup a training memory for agent in training mode
    training_memory = SharedObjects()

    # Build family and return general's children
    general_subordinate = build_agent(configuration, instantiated, 0, training_memory)

    # if the training memory is not set with agent value, then nobody use it
    # then delete it
    if training_memory.action_space is None:
        training_memory = None
    return general_subordinate, training_memory


def build_agent(configuration, instantiated, agent_id, training_memory):
    """
    Builds an agent from a configuration.
    Recursive function : builds the children before the agent itself
    :param agent_id: The id of the agent in the "structure" and "agents" part of the configuration
    :param training_memory: the memory to be used to communicate between the environment and the agent during training
    :return: the built agent
    """
    agent_info = get_agent_information(configuration["agents"], agent_id)
    agent_class = get_class(agent_info["class_name"])
    agent_arguments = get_arguments(agent_info["arguments"], training_memory)

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
            children_agent = build_agent(configuration, instantiated, child_id, training_memory)
            children.append(children_agent)
        agent_arguments["subordinate"] = children

    # Create the thing
    agent = agent_class(**agent_arguments)
    instantiated[agent_id] = agent
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
    if not check_max_one_training_agent(configuration):
        raise AgentArgumentError("A hierarchy can only have one or zero training agent")
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


def check_max_one_training_agent(configuration):
    training_agent_count = 0
    for agent in configuration["agents"]:
        if "train_mode" in agent["arguments"] and agent["arguments"]["train_mode"] == "True":
            training_agent_count += 1
    return training_agent_count < 2


def get_class(class_name):
    """
    returns the class corresponding to a string
    :param class_name: the given string pointing to the class from the working directory
    :return: the class as an object
    """
    parts = class_name.split('.')
    class_module = ".".join(parts[:-1])
    m = __import__(class_module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def get_arguments(args, training_memory):
    """
    Argument Parser
    :param args: dict of all the agent arguments with key and value as string
    :param training_memory: memory to be used for training agent if any
    :return: a dict with at least the same keys and values parsed into objects
    """
    for (arg_name, arg_value) in args.item():
        try:
            arg_value = int(arg_value)
        except ValueError:
            pass
        if arg_value == "True":
            arg_value = True
        elif arg_value == "False":
            arg_value = False
        args[arg_name] = arg_value

    # if train mode is set
    if "train_mode" in args:
        # add a memory object
        if args["train_mode"]:
            args["shared_memory"] = training_memory
        else:
            args["shared_memory"] = None
    return args


class CyclicStructureError(RuntimeError):
    """The loaded structure is cyclic"""


class UndefinedAgentError(RuntimeError):
    """An agent is declared in the structure but undefined"""


class AgentArgumentError(RuntimeError):
    """Arguments set to agents are not consistent with what is expected"""
