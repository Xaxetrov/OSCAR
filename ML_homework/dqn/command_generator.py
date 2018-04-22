import os
import datetime

load_memory = ['random', 'agent', 'empty']
decreasing_steps = range(10000, 100001, 45000)
config_file = ['config/learning_complex.json', 'config/learning.json']

i = 0
with open("cmd.txt", 'w') as f:
    for lm in load_memory:
        for ds in decreasing_steps:
            for c in config_file:
                cmd = "-lm " + str(lm) + " " \
                      "-ds " + str(ds) + " " \
                      "-c  " + str(c) + " " \
                      "-o ML_homework/results/" + str(datetime.date.today()) + \
                                           "_" + str(i) + "\n"
                f.write(cmd)
                i += 1
print(i)

