#!/bin/bash

echo "Configuration file:\n"
cat configs/transform_flags_linux.cfg

echo "Python script: trasform_replays.py"
cat transform_replays.py

echo -e "\n\n"
read -p "Proceed? [y/n] " PROCEED
if [ $PROCEED == "y" ]; then
    python scripts/transform_replays.py \
        --flagfile=configs/transform_flags_linux.cfg
elif [ $PROCEED == "n" ]; then
    echo "Terminating."
fi


