# Calibrates camera for map given as parameter and exports result.
# Example usage: "sh calibrateCamera.sh Flat64"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi

mkdir -p tmp
echo "$1" > tmp/map_name.txt
cd ./src
python3 -m pysc2.bin.agent --map $1 --agent coordinatesHelper.Calibration
rm -r ../tmp