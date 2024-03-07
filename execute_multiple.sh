#!/bin/bash

# Define the starting directory
START_DIR="experiments_hw3/"
# Define the starting script from which to begin execution
START_SCRIPT="q5-DDPG-HalfCheetah-v2.sh"

# Find all .sh files starting from the specified directory, sort them,
# and then get the list of scripts starting from the specified script
SCRIPTS=$(find "$START_DIR" -type f -name "*.sh" | sort | awk -v start=$START_SCRIPT '$0 ~ start,0')

# Flag indicating whether we've found the starting script yet
found_start=false

# Execute each script one by one
for script in $SCRIPTS; do
    if [[ "$script" == *"$START_SCRIPT"* ]]; then
        found_start=true
    fi
    if [ "$found_start" = true ]; then
        echo "Executing $script..."
        # Execute the script
        bash "$script"
        echo "Execution of $script completed."

        # Uncomment the line below if you want the script to wait for a user prompt
        # before proceeding to the next one. Comment it out if you want continuous execution.
        read -p "Press enter to continue to the next script..."
    fi
done

if [ "$found_start" = false ]; then
    echo "Starting script $START_SCRIPT not found."
else
    echo "All scripts starting from $START_SCRIPT have been executed."
fi
