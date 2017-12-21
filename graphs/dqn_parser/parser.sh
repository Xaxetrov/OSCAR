#!/bin/bash

#set default parameters
x_axis_name="episode"
x_axis_label="Episode"
x_axis_pos="2"

if [[ $# == "0" || $1 == "--help" ]] ; then
  echo "This script parse the execution log of the learning"
  echo "environement (oscar.env.envs.general_learning_env)"
  echo "to produce a graph of the evolution."
  echo ""
  echo "call to this script must be of the form:"
  echo "$0 [path to log file]"
  echo ""
  echo "The given output is a graph and a csv (wihout"
  echo "separator). Both of them are writed to the same"
  echo "directory as the input file but with a different"
  echo "extantion."
  exit
elif [[ $# == "2" ]] ; then
  if [[ $2 == "-t" || $2 == "--time" ]] ; then
    x_axis_name="time"
    x_axis_label="Time (min)"
    x_axis_pos="1" 
  fi
fi

re='^[0-9]+$'
ref='^[0-9]+([.][0-9]+)?$'
outfilename="${1%\.*}.csv"

inc=0

echo "" > "$outfilename"

episode_num=0
skipped_action_count=0
mean_reward=0
steps=0
win_state=0
median_reward=0
agent_steps=0
time=0

while IFS='' read -r line || [[ -n "$line" ]]; do
  line_prefix=${line:0:25}
  line_end=${line: 26:9}
  if [[ ${line:0:21} == "Failed meta action : " ]] ; then
    skipped_action_count=${line:21}
  elif [[ $line_prefix == "| time                  |" ]] ; then
    time=$line_end
  elif [[ $line_prefix == "| episodes              |" ]] ; then
    episode_num=$line_end
  elif [[ $line_prefix == "| episode steps         |" ]] ; then
    steps=$line_end
  elif [[ $line_prefix == "| learning agent steps  |" ]] ; then
    agent_steps=$line_end
  elif [[ $line_prefix == "| episode reward        |" ]] ; then
    reward=$line_end
  elif [[ $line_prefix == "| mean episode reward   |" ]] ; then
    mean_reward=$line_end
  elif [[ $line_prefix == "| median episode reward |" ]] ; then
    median_reward=$line_end
  elif [[ $line_prefix == "| win state             |" ]] ; then
    win_state=$line_end
    echo "$time $episode_num $reward $mean_reward $median_reward $skipped_action_count $win_state $steps $agent_steps" >> "$outfilename"
  fi
done < "$1"

# get max value point:
xmax=$(awk -v max=0 "{if(\$3>max){want=\$$x_axis_pos; max=\$3}}END{print want} " $outfilename)
ymax=$(awk -v max=0 '{if($3>max){max=$3}}END{print max} ' $outfilename)

maxx=$(awk -v max=0 "{if(\$$x_axis_pos>max){max=\$$x_axis_pos}}END{print max} " $outfilename)

echo "max = ($xmax, $ymax)"

# set object circle at $xmax,$ymax size 1000

echo "set title 'Reward evolution with $x_axis_name from file $1'
set ytics 2000 nomirror
set ylabel 'Reward'
set y2range [0:3]
set y2tics 1 nomirror tc lt 4
set y2label 'Victory score'
set xlabel '$x_axis_label'
set key left box
set term png
set output '${1%\.*}.png'
set style data lines
set label 1 \"($xmax;$ymax)\" at $xmax,$ymax right point pointtype 1 lw 1 ps 2 offset -0.5,0.5
show label
plot '$outfilename' using $x_axis_pos:3 title 'Episode reward',\
     '$outfilename' using $x_axis_pos:4 title 'Mean reward (100 episodes)' with lines linewidth 2,\
     '$outfilename' using $x_axis_pos:5 title 'Median reward (100 episodes)' with lines linewidth 2,\
     '$outfilename' using $x_axis_pos:7 title 'Victory score' axes x1y2 with points" | gnuplot

