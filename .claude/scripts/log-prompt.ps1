# Windows PowerShell script for logging user prompts
param()

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$json = [Console]::In.ReadToEnd()
$data = $json | ConvertFrom-Json
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$prompt = $data.prompt -replace "`n", ' [NL] '
$line = "[$timestamp] USER: $prompt"

$logPath = Join-Path $PSScriptRoot "..\prompt.log"
[System.IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
