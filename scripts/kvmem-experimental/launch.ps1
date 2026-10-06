param(
    [string]$Model,
    [ValidateSet('dense','kvmem','tiered-exact')][string]$Mode = 'dense',
    [ValidateSet(32768,131072)][int]$ViewTokens = 32768,
    [int]$Port = 8111,
    [string]$BindAddress = '127.0.0.1',
    [string]$ApiKeyFile,
    [switch]$CpuVision,
    [string]$VisionMmproj
)
$ErrorActionPreference = 'Stop'
if (-not $Model) { $Model = Join-Path $PSScriptRoot 'models\qwen3_8_27b.ninfer' }
$Model = (Resolve-Path -LiteralPath $Model).Path
$exe = Join-Path $PSScriptRoot 'ninfer-serve.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Run this launcher from the extracted release package.' }
if (-not $ApiKeyFile) { $ApiKeyFile = Join-Path (Split-Path -Parent $Model) 'api-keys.txt' }
$serveArgs = @($Model,'--host',$BindAddress,'--port',"$Port",'--max-context','262144','--max-concurrency','1','--max-pending-requests','16','--pending-timeout-ms','600000','--kv-mode',$Mode,'--spec','mtp','--draft-tokens','3','--lm-head-draft','--preserve-thinking')
if ($Mode -eq 'dense') {
    $serveArgs += @('--kv-dtype','rk4v4-e8','--prefill-chunk','1024')
} else {
    Write-Host 'EXPERIMENTAL: quality evaluation incomplete; sparse Graph off; C=1.'
    $serveArgs += @('--kv-dtype','int8','--kvmem-view-tokens',"$ViewTokens",'--kvmem-prefill','exact','--kvmem-host-archive','auto','--kvmem-mtp-window','32768')
}
if (Test-Path -LiteralPath $ApiKeyFile) {
    $apiKey = Get-Content -LiteralPath $ApiKeyFile | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -First 1
    if ($apiKey) { $serveArgs += @('--api-key',$apiKey); Write-Host 'API authentication enabled from key file.' }
}
if ($CpuVision -or $VisionMmproj) {
    $serveArgs += @('--vision','--vision-device','cpu','--vision-max-tokens','1024')
    if ($VisionMmproj) { $serveArgs += @('--vision-mmproj',(Resolve-Path -LiteralPath $VisionMmproj).Path) }
}
& $exe @serveArgs
exit $LASTEXITCODE
