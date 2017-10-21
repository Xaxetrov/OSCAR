import json

from oscar.agent.commander.commander import Commander


class GeneralCommander(Commander):
    def __init__(self):
        super().__init__()

    @classmethod
    def __load_configuration_file(cls, filename):
        with open(filename) as configuration_file:
            configuration = json.load(configuration_file)
