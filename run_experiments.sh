#!/bin/bash

# Change to the directory where your experiment scripts are located
cd experiments_hw3

# Find all .sh files and execute them
for script in *.sh; do
    if [ -f "$script" ]; then
        echo "Executing $script..."
        ./"$script"
    fi
done

echo "All experiments have been executed."
