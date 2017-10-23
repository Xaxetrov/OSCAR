#!/bin/bash

re='^[0-9]+$'
ref='^[0-9]+([.][0-9]+)?$'
outfilename="${1%\.*}.csv"
rewardoutfile="reward$outfilename"

inc=0

echo "" > $outfilename
echo "" > $rewardoutfile

while IFS='' read -r line || [[ -n "$line" ]]; do
    	stepnum=${line%%/*}
	stepnum=${stepnum##* }
	if [[ $stepnum =~ $re ]] ; then
		reward=${line%Game*}
		reward=${reward##*reward: }
		if [[ $reward =~ $ref ]] ; then
			echo "$((stepnum+inc)) $reward" >> $outfilename
		fi
		if [[ $stepnum == "10000" ]] ; then
			inc=$((inc+10000))
		fi
	fi
	ep_rewards=${line#*episode_reward: }
	ep_rewards=${ep_rewards%%]*}
	lenght=${#ep_rewards}
	if (( lenght < 23 )) && (( lenght > 20 )); then
		mean_reward=${ep_rewards%% *}
		min_reward=${ep_rewards%,*}
		min_reward=${min_reward#*[}
		max_reward=${ep_rewards#*, }
		echo "$inc $mean_reward $min_reward $max_reward" >> $rewardoutfile
	fi
done < "$1"

echo "set title 'Learning data of file $1'
set ylabel 'average step reward'
set xlabel 'step'
set key left box
set term png
set output '${1%\.*}.png'
set style data lines
plot '$outfilename' title 'reward'" | gnuplot

echo "set title 'Learning data of file $1'
set ylabel 'average reward'
set xlabel 'step'
set key left box
set term png
set output 'reward${1%\.*}.png'
set style data linespoints
plot '$rewardoutfile' using 1:2 title 'mean reward', \
	 '$rewardoutfile' using 1:3 title 'min reward', \
	 '$rewardoutfile' using 1:4 title 'max reward'" | gnuplot







