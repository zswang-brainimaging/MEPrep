#!/bin/bash
# sequential_fmriprep.sh - Process multiple fMRI sessions sequentially

# Configuration
SUBJECT_ID="sub-01"  # Change to your subject ID
NUM_SESSIONS=10
BIDS_DIR="/path/to/bids"
OUTPUT_DIR="/path/to/output"
JSON_FILTER_DIR="/path/to/json_filters"  # Directory containing your session JSON files

# Function to run a single session and wait for completion
run_session() {
    local session=$1
    local json_file="${JSON_FILTER_DIR}/session_${session}.json"
    local service_name="fmriprep_${SUBJECT_ID}_ses${session}"
    
    echo "Starting processing for session ${session}..."
    
    # Remove any existing service with the same name
    docker service rm ${service_name} 2>/dev/null
    
    # Create the service for this session
    docker service create \
        --name ${service_name} \
        --mount type=bind,source=${BIDS_DIR},destination=/data/bids,readonly \
        --mount type=bind,source=${OUTPUT_DIR},destination=/data/output \
        --mount type=bind,source=${json_file},destination=/data/filter.json,readonly \
        nipreps/fmriprep:latest \
        /data/bids /data/output participant \
        --participant-label ${SUBJECT_ID} \
        --bids-filter-file /data/filter.json \
        -w /data/output/work
    
    # Wait for service to complete
    echo "Waiting for session ${session} to complete..."
    
    # Monitor the service until it's gone (completed and removed)
    while docker service ls | grep -q ${service_name}; do
        # Check if any tasks failed
        if docker service ps ${service_name} | grep -q "Failed"; then
            echo "Service for session ${session} failed!"
            docker service rm ${service_name}
            return 1
        fi
        sleep 30
    done
    
    echo "Session ${session} completed successfully."
    return 0
}

# Main execution
echo "Starting sequential processing of ${NUM_SESSIONS} sessions for subject ${SUBJECT_ID}"

for session in $(seq 1 ${NUM_SESSIONS}); do
    run_session ${session}
    
    # Check if the session completed successfully
    if [ $? -ne 0 ]; then
        echo "Error processing session ${session}. Stopping."
        exit 1
    fi
    
    echo "Completed session ${session}. Moving to next session..."
    sleep 10  # Brief pause between sessions
done

echo "All sessions completed successfully!"