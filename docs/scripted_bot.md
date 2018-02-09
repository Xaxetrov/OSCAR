# The scripted bot

Using our hierarchical framework, we have implemented a simple proof-of-concept scripted bot. This one is very limited. It plays terrans with a simple strategy and is only able to beat the built-in AI of Starcraft in "very easy" mode.

<p align="center">
<img align="center" src="https://github.com/Xaxetrov/OSCAR/blob/docs/docs/images/scripted.png?raw=true" alt="The scripted bot" title="The scripted bot" height="300px">
</p>

It is composed of 2 commanders and 5 agents:

* **Strategy Manager** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/strategy_manager.py)): calls the *Economy Manager*, *Combat Manager* and *Scout* at regular fixed intervals.

* **Economy Manager** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/economy_manager.py)): builds supply depots and SCVs and orders idle workers to harvest minerals.

* **Combat Manager** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/combat_manager.py)): calls the *Army Supplier*, *Attack Manager* and *Micro Manager* at regular fixed intervals.

* **Scout** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/scout.py)): If less than one third of the map is explored or the enemy is not visible on the minimap, sends a worker to a random unexplored place.

* **Army supplier** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/army_supplier.py)): builds barracks and marines.

* **Attack Manager** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/attack_manager.py)): sends the entire army to a random location where there are some enemies.

* **Micro Manager** ([source](https://github.com/Xaxetrov/OSCAR/blob/docs/oscar/agent/scripted/micro_manager.py)): moves the camera to random fight locations and controls units individually to avoid enemies as much as possible.