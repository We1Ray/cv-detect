# Windows PowerShell script for logging assistant responses
param()

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$json = [Console]::In.ReadToEnd()
$data = $json | ConvertFrom-Json
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$reason = $data.stop_hook_active
$response = 'N/A'

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
$line = "[$timestamp] ASSISTANT ($reason): $truncated..."

$logPath = Join-Path $PSScriptRoot "..\prompt.log"
[System.IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
