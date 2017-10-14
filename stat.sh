#!/bin/bash

NUMBER_OF_RUN=20
EPSILON_STEP=10

mv reward.csv reward.csv.old

echo "$NUMBER_OF_RUN" > "config"
echo "$EPSILON_STEP" >> "config"

python3.6 -m pysc2.bin.agent --map CollectMineralShards --agent src.playagent.PlayAgent --screen_resolution 64 --max_agent_steps $((240*NUMBER_OF_RUN*(100/EPSILON_STEP+1)+1)) --norender

rm config

echo "set title 'Number of run $NUMBER_OF_RUN'
set ylabel 'average reward'
set xlabel 'epsilon'
set key left box
set term png
set output 'graph.png'
set style data linespoints
plot 'reward.csv' title 'reward'" | gnuplot
