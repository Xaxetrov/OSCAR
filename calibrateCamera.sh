# Calibrates camera for map given as parameter and exports result.
# Example usage: "sh calibrateCamera.sh Flat64"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi

mkdir -p data
mkdir -p data/tmp
echo "$1" > data/tmp/map_name.txt
python3 -m pysc2.bin.agent --map $1 --agent oscar.util.coordinates_helper.Calibration
#rm -r ../tmp