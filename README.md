# OSCAR
5IF project. Work on Starcraft2 + deepmind API. 

## APIs and frameworks
[pysc2](https://github.com/deepmind/pysc2)
[Keras](https://keras.io/) (for neural networks)

## How to run an agent

    python -m pysc2.bin.agent \
        --map <map> --agent <agent> --agent_race <race>
        
Example:

    python -m pysc2.bin.agent \
        --map Simple64 --agent src.agents.simple.Agent