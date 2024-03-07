#!/bin/bash

# Target PID to wait for
PID_TO_WAIT_FOR=227119

# Path to your script
SCRIPT_TO_EXECUTE="execute_multiple.sh"

# Check if the process is still running by looking for the PID in the process list
while kill -0 $PID_TO_WAIT_FOR 2> /dev/null; do
    echo "Process $PID_TO_WAIT_FOR is still running. Waiting..."
    # Wait for a bit before checking again to avoid spamming the CPU
    sleep 1
done

echo "Process $PID_TO_WAIT_FOR has finished. Executing $SCRIPT_TO_EXECUTE..."

# Make sure your script is executable
chmod +x $SCRIPT_TO_EXECUTE

# Execute your script
./$SCRIPT_TO_EXECUTE

echo "Script $SCRIPT_TO_EXECUTE executed."

