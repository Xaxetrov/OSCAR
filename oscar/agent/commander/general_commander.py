from oscar.agent.commander.commander import Commander
from oscar.hiearchy_factory import build_hierarchy

DEFAULT_CONFIGURATION = "config/sample_configuration.json"


class GeneralCommander(Commander):
    def __init__(self, configuration_filename=DEFAULT_CONFIGURATION):
        subordinates = build_hierarchy(configuration_filename)
        super().__init__(subordinates)
        print(self)
