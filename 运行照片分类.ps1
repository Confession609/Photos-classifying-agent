[CmdletBinding()]
param(
    [string]$InputDirectory,
    [string]$OutputDirectory,
    [switch]$Check,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskEntry = Join-Path $PSScriptRoot 'scripts\classify_daily_photos.py'
$taskConfig = Join-Path $PSScriptRoot 'daily_workflow_config.json'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw '环境缺少项目Python解释器，请先配置 .venv；不会自动安装。'
}
if (-not $Check -and -not $InputDirectory) {
    $taskSettings = Get-Content -LiteralPath $taskConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $taskSettings.input_directory) {
        $InputDirectory = Read-Host '请输入待分类照片文件夹的完整路径（输入为空则退出）'
        if ([string]::IsNullOrWhiteSpace($InputDirectory)) { return }
        $InputDirectory = $InputDirectory.Trim().Trim('"')
    }
}
$taskArguments = @('-B', '-X', 'utf8', $taskEntry, '--config', $taskConfig)
if ($InputDirectory) { $taskArguments += @('--input', $InputDirectory) }
if ($OutputDirectory) { $taskArguments += @('--output', $OutputDirectory) }
if ($Check) { $taskArguments += '--check' }
if ($Plan) { $taskArguments += '--plan' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) {
    throw "工作流退出码为 $LASTEXITCODE；请查看上方错误或本次运行目录中的 progress.json/records。"
}
