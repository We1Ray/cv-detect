#!/bin/bash
# Mac/Linux bash script for logging user prompts
# Compatible with Mac ARM (Apple Silicon), Mac Intel, and Linux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$SCRIPT_DIR/../prompt.log"

# Auto-detect Python path
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif [ -x "/opt/homebrew/bin/python3" ]; then
    # Mac ARM (Apple Silicon)
    PYTHON="/opt/homebrew/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    # Mac Intel
    PYTHON="/usr/local/bin/python3"
else
    PYTHON="python"
fi

# Read JSON from stdin
json=$(cat)

# Parse JSON and extract prompt
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
prompt=$($PYTHON -c "import sys, json; data=json.load(sys.stdin); print(data.get('prompt', '').replace('\n', ' [NL] '))" <<< "$json" 2>/dev/null || echo "")

# Write to log
echo "[$timestamp] USER: $prompt" >> "$LOG_PATH"
