# Machine Learning Homework 4: Markov Decision Processes

## Requirement

Python 3.6 or above with:

 - [gym](https://github.com/openai/gym) = 0.9.4
 - [pysc2](https://github.com/deepmind/pysc2)
 - [keras-rl](https://github.com/keras-rl/keras-rl)
 - keras
 - tensorflow
 - scipy

to generate plot you will also need:

 - jupyter
 - matplotlib
 - sqlite3 (with extension enabled)
 - seaborn

## Generate Data

### MDP

To generate the data we displayed of our report you just have to run:

Value iteration on both problems: 

    python3 -m ML_homework.value_iteration.basic_value_iteration
    python3 -m ML_homework.value_iteration.complexe_value_iteration
 
Policy iteration on both problem:

    python3 -m ML_homework.policy_iteration.basic_value_iteration
    python3 -m ML_homework.policy_iteration.complex_value_iteration

### DQN

To generate the result for DQN you will need some time, a good computer and running the following scripts:

    python3 -m ML_homework.dqn.command_generator
    ./gtl_local_run.sh
    
## Plots

The plots are generated thanks to the following Jupyter notebook:

 - ML_homework/value_iter.ipynb
 - ML_homework/policy_iter.ipynb
 - ML_homework/dqn.ipynb

