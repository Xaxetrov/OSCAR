# The hierarchical framework

We have developped our bots as hierarchies of subagents, unified by a hierarchical framework. This way, the gameplay is split into subtasks, easier to manage for both scripted and RL approaches.

The framework enables to create bots as directed acyclic graphs containing three types of vertices: *general*, *commander* and *agent*. Each graph should contain one general and at least one agent.

 **General**: at the top of the hierarchy, provides an interface between the *PySc2* API and the bot. It has a unique child, either an agent or a commander    
 **Commander**: a specialized agent that can have several children, agents or other commanders. Its role is to choose which of its children will play.    
 ***Agent***: leaf of the graph. When called, it returns actions to be executed.

<p align="center">
<img align="center" src="https://github.com/Xaxetrov/OSCAR/raw/master/docs/images/hierarchy.png?raw=true" alt="An example of hierarchical bot" title="An example of hierarchical bot" height="300px">
</p>
 
At each step of the game, commanders and agents play accordingly to the graph structure, starting from the general. Once an agent has returned some actions, these ones are sent back to the general and played through PySc2. Afterwards, a callback called by the general notifies the playing agent of the result (success or error) of its actions.
 
### Agent return format
An agent returns a dictionary with the following items to its commander:
* "actions": a list of actions to be executed in a row. If an error occurs before the list is fully executed, its execution is cancelled.
* "success_callback" (optional): the callback to be called if the action is successfully executed.
* "failure_callback" (optional): the callback to be called if the action fails to be executed.
* "locked_choice" (optional): a boolean to ensure that the same agent will be called once the current list of actions is empty. If omitted, the field is set to its default value: False.
 
### Configuration file
Configuration files describe the graph of commanders and agents using the JSON format.
Here is a basic example of such a file.
``` 
{
    "structure": {
        0: [1, 2],
        1: [3, 4, 5]
    },
    "agents": [
        {
            "id": 0,
            "class_name": "oscar.agent.scripted.strategy_manager.StrategyManager",
            "arguments": {}
        },
        ...
    ]
}
```

The field "agents" provides a description of commanders and agents involved in the bot. Each agent and commander description includes an identifier ("id"), the path to the class as well as optional arguments.
In the above example, "1" and "2" are the children of the commander "0", and agents "3", "4" and "5" are the children of "1".