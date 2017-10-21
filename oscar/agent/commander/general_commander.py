from oscar.agent.commander.commander import Commander
from oscar.hiearchy_factory import build_hierarchy


class GeneralCommander(Commander):
    def __init__(self, configuration_filename="config/sample_configuration.json"):
        build_hierarchy(configuration_filename)
        super().__init__([])
