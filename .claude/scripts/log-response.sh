#!/bin/bash
# Mac/Linux bash script for logging assistant responses
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

# Parse JSON and extract response
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

response=$($PYTHON -c "
import sys, json
data = json.load(sys.stdin)
reason = data.get('stop_hook_active', 'unknown')
response = 'N/A'
transcript = data.get('transcript', [])
for msg in reversed(transcript):
    if msg.get('type') == 'assistant':
        content = msg.get('message', {}).get('content', [])
        if isinstance(content, list):
            texts = [c.get('text', '') for c in content if c.get('type') == 'text']
            response = ' '.join(texts)
        else:
            response = str(content)
        break
response = response.replace('\n', ' [NL] ')[:500]
print(f'({reason}): {response}...')
" <<< "$json" 2>/dev/null || echo "(unknown): N/A...")

# Write to log
echo "[$timestamp] ASSISTANT $response" >> "$LOG_PATH"
