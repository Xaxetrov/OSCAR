# Runs OSCAR with the map provided as parameter.
# Example usage: "sh run.sh Flat64"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi

mkdir -p tmp
echo "$1" > tmp/map_name.txt

cd ./src
python3 -m pysc2.bin.agent --map $1 --agent AI.FindAndDefeatZerglings