<#
.SYNOPSIS
    Telegram Notification Hook for Claude Code
.DESCRIPTION
    FAANG-style notification system for Claude Code events
    Sends structured notifications via Telegram Bot API
.AUTHOR
    Project Team
.VERSION
    1.0.0
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('UserPromptSubmit', 'PermissionRequest', 'PostToolUse', 'PostToolUseFailure', 'Notification', 'Stop')]
    [string]$EventType,

    [Parameter(Mandatory=$false, ValueFromPipeline=$true)]
    [string]$JsonInput,

    [Parameter(Mandatory=$false)]
    [switch]$WriteLog
)

# ============================================================================
# Configuration
# ============================================================================
# Dynamically get project name from root folder (2 levels up from hooks/)
$PROJECT_NAME = Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

$BOT_TOKEN = if ($env:TELEGRAM_BOT_TOKEN) {
    $env:TELEGRAM_BOT_TOKEN
} else {
    "8328084093:AAEgicj_IHGm3o_CUPgHcmEQZJ_YjRcA96o"
}

# Multiple Chat IDs for notifications
$CHAT_IDS = if ($env:TELEGRAM_CHAT_IDS) {
    $env:TELEGRAM_CHAT_IDS -split ','
} elseif ($env:TELEGRAM_CHAT_ID) {
    @($env:TELEGRAM_CHAT_ID)
} else {
    @(
        "6215674302",   # User 1
        "8308709862"    # User 2
    )
}

$TELEGRAM_API_URL = "https://api.telegram.org/bot$BOT_TOKEN/sendMessage"

# ============================================================================
# Encoding Setup
# ============================================================================
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================================
# Read Input from environment variable, parameter, or stdin
# ============================================================================
if (-not $JsonInput) {
    if ($env:HOOK_JSON) {
        $JsonInput = $env:HOOK_JSON
    } else {
        $JsonInput = [Console]::In.ReadToEnd()
    }
}

# Debug log
$debugLogPath = Join-Path $PSScriptRoot "..\hook-debug.log"
$debugTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$debugLine = "[$debugTimestamp] EventType=$EventType, JsonLength=$($JsonInput.Length), JsonPreview=$($JsonInput.Substring(0, [Math]::Min(100, $JsonInput.Length)))"
[System.IO.File]::AppendAllText($debugLogPath, $debugLine + [Environment]::NewLine, [System.Text.Encoding]::UTF8)

# ============================================================================
# Parse JSON Input
# ============================================================================
try {
    $data = $JsonInput | ConvertFrom-Json -ErrorAction Stop
} catch {
    $data = @{}
}

# ============================================================================
# FAANG-style Message Formatting
# ============================================================================
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$hostname = $env:COMPUTERNAME
$username = $env:USERNAME

# Event-specific config (using text symbols for PowerShell compatibility)
$eventConfig = @{
    'UserPromptSubmit' = @{ symbol = "[USER]"; title = "User Prompt Submitted" }
    'PermissionRequest' = @{ symbol = "[LOCK]"; title = "Permission Request" }
    'PostToolUse' = @{ symbol = "[OK]"; title = "Tool Execution Success" }
    'PostToolUseFailure' = @{ symbol = "[FAIL]"; title = "Tool Execution Failed" }
    'Notification' = @{ symbol = "[INFO]"; title = "Notification" }
    'Stop' = @{ symbol = "[END]"; title = "Session Stopped" }
}

$config = $eventConfig[$EventType]
$symbol = $config.symbol
$title = $config.title

# Build detailed message based on event type
$details = ""
switch ($EventType) {
    'UserPromptSubmit' {
        $prompt = if ($data.prompt) {
            $promptStr = $data.prompt.ToString()
            if ($promptStr.Length -gt 500) { $promptStr.Substring(0, 500) + "..." } else { $promptStr }
        } else { "No prompt" }
        $details = "Prompt: $prompt"
    }
    'PermissionRequest' {
        $toolName = if ($data.tool_name) { $data.tool_name } else { "Unknown" }
        $toolInput = if ($data.tool_input) {
            $jsonStr = ($data.tool_input | ConvertTo-Json -Compress -Depth 2)
            if ($jsonStr.Length -gt 200) { $jsonStr.Substring(0, 200) + "..." } else { $jsonStr }
        } else { "N/A" }
        $details = "Tool: $toolName`nInput: $toolInput"
    }
    'PostToolUse' {
        $toolName = if ($data.tool_name) { $data.tool_name } else { "Unknown" }
        $duration = if ($data.duration_ms) { "$($data.duration_ms)ms" } else { "N/A" }
        $details = "Tool: $toolName`nDuration: $duration`nStatus: Success"
    }
    'PostToolUseFailure' {
        $toolName = if ($data.tool_name) { $data.tool_name } else { "Unknown" }
        $errorMsg = if ($data.error) {
            $errStr = $data.error.ToString()
            if ($errStr.Length -gt 300) { $errStr.Substring(0, 300) + "..." } else { $errStr }
        } else { "Unknown error" }
        $details = "Tool: $toolName`nError: $errorMsg`nStatus: FAILED"
    }
    'Notification' {
        $message = if ($data.message) { $data.message } else { "No message" }
        $details = "Message: $message"
    }
    'Stop' {
        $reason = if ($data.stop_hook_active) { $data.stop_hook_active } else { "end_turn" }
        $transcriptCount = if ($data.transcript) { $data.transcript.Count } else { 0 }
        $details = "Reason: $reason`nTranscript Items: $transcriptCount"
    }
}

# ============================================================================
# Write to prompt.log if requested
# ============================================================================
if ($WriteLog) {
    $promptLogPath = Join-Path $PSScriptRoot "..\prompt.log"

    if ($EventType -eq 'UserPromptSubmit') {
        $logPrompt = if ($data.prompt) { $data.prompt -replace "`n", ' [NL] ' } else { "No prompt" }
        $logLine = "[$timestamp] USER: $logPrompt"
        [System.IO.File]::AppendAllText($promptLogPath, $logLine + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    elseif ($EventType -eq 'Stop') {
        $reason = if ($data.stop_hook_active) { $data.stop_hook_active } else { "end_turn" }
        $response = "N/A"
        if ($data.transcript -and $data.transcript.Count -gt 0) {
            $lastMsg = $data.transcript | Where-Object { $_.type -eq 'assistant' } | Select-Object -Last 1
            if ($lastMsg -and $lastMsg.message -and $lastMsg.message.content) {
                $content = $lastMsg.message.content
                if ($content -is [array]) {
                    $response = ($content | Where-Object { $_.type -eq 'text' } | ForEach-Object { $_.text }) -join ' '
                } else {
                    $response = $content
                }
            }
        }
        $response = $response -replace "`n", ' [NL] '
        $truncated = if ($response.Length -gt 500) { $response.Substring(0, 500) } else { $response }
        $logLine = "[$timestamp] ASSISTANT ($reason): $truncated..."
        [System.IO.File]::AppendAllText($promptLogPath, $logLine + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
}

# ============================================================================
# Construct Telegram Message
# ============================================================================
$telegramMessage = "$symbol [$PROJECT_NAME] $title`n`n====================`nTime: $timestamp`nHost: $hostname`nUser: $username`nTrigger: $EventType`n====================`n`n$details`n`n====================`nClaude Code Hook System"

# ============================================================================
# Send Telegram Message to All Chat IDs
# ============================================================================
$headers = @{
    "Content-Type" = "application/json; charset=UTF-8"
}

foreach ($chatId in $CHAT_IDS) {
    $body = @{
        chat_id = $chatId.Trim()
        text = $telegramMessage
    } | ConvertTo-Json -Depth 3

    try {
        $response = Invoke-RestMethod -Uri $TELEGRAM_API_URL -Method Post -Headers $headers -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ErrorAction Stop
    } catch {
        $errorMessage = $_.Exception.Message
        $statusCode = ""
        $responseBody = ""

        if ($_.Exception.Response) {
            $statusCode = " (HTTP $($_.Exception.Response.StatusCode.value__))"
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $reader.BaseStream.Position = 0
                $responseBody = " - Response: $($reader.ReadToEnd())"
            } catch {}
        }

        $errorLog = "[$timestamp] Telegram API Error for chat_id=$chatId$statusCode`: $errorMessage$responseBody"
        $logPath = Join-Path $PSScriptRoot "..\telegram-notification.log"
        [System.IO.File]::AppendAllText($logPath, $errorLog + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
}

exit 0
