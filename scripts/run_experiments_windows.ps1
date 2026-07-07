param(
    [ValidateSet("light", "medium", "hard")]
    [string]$Profile = "light",

    [string]$ResultsRoot = "results\runs",

    [int]$Seed = 12345,

    [switch]$Append,
    [switch]$SkipInstall,
    [switch]$Clean,
    [switch]$OpenOutput,

    [string]$Python = "py -3"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Profiles = @{
    light = @{
        Shots = 1024
        Repetitions = 1
        IndexModes = @("sequential")
        SyntheticSizes = @("100")
        RealFiles = @()
        MaxRealTriples = 100
        TimeoutSeconds = 300
        ScalingRepetitions = 1
        ScalingExtraArgs = @("--skip-real", "--no-compute-decomposed-metrics", "--no-compute-transpiled-metrics")
    }
    medium = @{
        Shots = 10000
        Repetitions = 3
        IndexModes = @("sequential", "paper")
        SyntheticSizes = @("100", "1000")
        RealFiles = @("Aristotle.xml", "exampleV3.ttl", "productsSmall.rdf")
        MaxRealTriples = 500
        TimeoutSeconds = 900
        ScalingRepetitions = 2
        ScalingExtraArgs = @("--max-metric-qubits", "14")
    }
    hard = @{
        Shots = 10000
        Repetitions = 5
        IndexModes = @("sequential", "paper")
        SyntheticSizes = @("100", "1000", "5000", "10000")
        RealFiles = @("Aristotle.xml", "exampleV3.ttl", "productsSmall.rdf", "DecodedOntologies_V2.ttl")
        MaxRealTriples = 2000
        TimeoutSeconds = 1800
        ScalingRepetitions = 3
        ScalingExtraArgs = @("--max-metric-qubits", "14")
    }
}

$Config = $Profiles[$Profile]
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([System.IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRootPath = [System.IO.Path]::GetFullPath($ResultsRoot)
}
else {
    $ResultsRootPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $ResultsRoot))
}
$RunDir = Join-Path $ResultsRootPath "${Timestamp}_${Profile}"

if ((Test-Path -LiteralPath $RunDir) -and $Clean) {
    $resolvedRunDir = [System.IO.Path]::GetFullPath($RunDir)
    if (-not $resolvedRunDir.StartsWith($ResultsRootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a run directory outside ResultsRoot: $resolvedRunDir"
    }
    Remove-Item -LiteralPath $RunDir -Recurse -Force
}

$Directories = @(
    $RunDir,
    (Join-Path $RunDir "logs"),
    (Join-Path $RunDir "tests"),
    (Join-Path $RunDir "chapter9"),
    (Join-Path $RunDir "chapter9_sequential"),
    (Join-Path $RunDir "chapter9_paper"),
    (Join-Path $RunDir "scaling"),
    (Join-Path $RunDir "baselines"),
    (Join-Path $RunDir "schema_matching"),
    (Join-Path $RunDir "figures"),
    (Join-Path $RunDir "reports"),
    (Join-Path $RunDir "tables")
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

$ManifestPath = Join-Path $RunDir "run_manifest.json"
$CommandLogPath = Join-Path $RunDir "command_log.jsonl"

function Split-PythonCommand {
    param([string]$Command)
    $parts = $Command -split "\s+"
    if ($parts.Count -lt 1 -or [string]::IsNullOrWhiteSpace($parts[0])) {
        throw "Python command is empty."
    }
    return @{
        File = $parts[0]
        Args = @($parts | Select-Object -Skip 1)
    }
}

$PythonCommand = Split-PythonCommand $Python
$PythonFile = $PythonCommand.File
$PythonArgs = @($PythonCommand.Args)

function Write-Manifest {
    param(
        [string]$Status,
        [string]$Message = ""
    )
    $manifest = [ordered]@{
        status = $Status
        message = $Message
        profile = $Profile
        run_dir = $RunDir
        results_root = $ResultsRootPath
        seed = $Seed
        shots = $Config.Shots
        repetitions = $Config.Repetitions
        synthetic_sizes = $Config.SyntheticSizes
        real_files = $Config.RealFiles
        append = [bool]$Append
        skip_install = [bool]$SkipInstall
        started_at = $script:StartedAt
        updated_at = (Get-Date).ToString("o")
        finished_at = if ($Status -in @("SUCCESS", "FAILED")) { (Get-Date).ToString("o") } else { $null }
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Write-CommandRecord {
    param([hashtable]$Record)
    ($Record | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $CommandLogPath -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )
    $logPath = Join-Path (Join-Path $RunDir "logs") "$Name.log"
    $commandText = "$FilePath $($Arguments -join ' ')"
    $started = Get-Date
    Write-Host ""
    Write-Host ">>> $commandText"
    Write-CommandRecord @{
        event = "start"
        name = $Name
        command = $commandText
        cwd = $WorkingDirectory
        started_at = $started.ToString("o")
        log = $logPath
    }
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }
    }
    finally {
        Pop-Location
    }
    $finished = Get-Date
    Write-CommandRecord @{
        event = "finish"
        name = $Name
        command = $commandText
        cwd = $WorkingDirectory
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        exit_code = $exitCode
        log = $logPath
    }
    if ($exitCode -ne 0) {
        throw "Command '$Name' failed with exit code $exitCode. See $logPath"
    }
}

$script:StartedAt = (Get-Date).ToString("o")
Write-Manifest -Status "RUNNING"

try {
    Invoke-LoggedCommand -Name "python_version" -FilePath $PythonFile -Arguments ($PythonArgs + @("--version"))

    if (-not $SkipInstall) {
        $VenvDir = Join-Path $RepoRoot ".venv"
        $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            Invoke-LoggedCommand -Name "create_venv" -FilePath $PythonFile -Arguments ($PythonArgs + @("-m", "venv", ".venv"))
        }
        $PythonFile = $VenvPython
        $PythonArgs = @()
        Invoke-LoggedCommand -Name "pip_install" -FilePath $PythonFile -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
    }

    Invoke-LoggedCommand -Name "pip_freeze" -FilePath $PythonFile -Arguments @("-m", "pip", "freeze")
    Copy-Item -LiteralPath (Join-Path $RunDir "logs\pip_freeze.log") -Destination (Join-Path $RunDir "pip_freeze.txt") -Force

    $envText = @(
        "Profile: $Profile",
        "RunDir: $RunDir",
        "RepoRoot: $RepoRoot",
        "Seed: $Seed",
        "Shots: $($Config.Shots)",
        "Repetitions: $($Config.Repetitions)",
        "StartedAt: $script:StartedAt",
        "ComputerName: $env:COMPUTERNAME",
        "OS: $([System.Environment]::OSVersion.VersionString)"
    )
    $envText | Set-Content -LiteralPath (Join-Path $RunDir "environment.txt") -Encoding UTF8

    Invoke-LoggedCommand -Name "pytest" -FilePath $PythonFile -Arguments @(
        "-m", "pytest",
        "--junitxml", (Join-Path $RunDir "tests\pytest.xml")
    )

    foreach ($IndexMode in $Config.IndexModes) {
        $ChapterDir = Join-Path $RunDir "chapter9_$IndexMode"
        Invoke-LoggedCommand -Name "chapter9_$IndexMode" -FilePath $PythonFile -Arguments @(
            "scripts\run_chapter9_experiments.py",
            "--shots", [string]$Config.Shots,
            "--repetitions", [string]$Config.Repetitions,
            "--index-mode", $IndexMode,
            "--seed", [string]$Seed,
            "--output-dir", $ChapterDir
        )
    }

    $ScalingArgs = @(
        "scripts\run_all_experiments.py",
        "--results-root", (Join-Path $RunDir "scaling"),
        "--synthetic-sizes"
    ) + $Config.SyntheticSizes + @(
        "--shots", [string]$Config.Shots,
        "--repetitions", [string]$Config.ScalingRepetitions,
        "--timeout-seconds", [string]$Config.TimeoutSeconds,
        "--max-real-triples", [string]$Config.MaxRealTriples
    )
    if ($Config.RealFiles.Count -gt 0) {
        $ScalingArgs += @("--real-files") + $Config.RealFiles
    }
    $ScalingArgs += $Config.ScalingExtraArgs
    if ($Append) {
        $ScalingArgs += "--append"
    }
    Invoke-LoggedCommand -Name "scaling" -FilePath $PythonFile -Arguments $ScalingArgs

    Invoke-LoggedCommand -Name "classical_baselines" -FilePath $PythonFile -Arguments @(
        "scripts\run_classical_baselines.py",
        "--output-dir", (Join-Path $RunDir "baselines"),
        "--seed", [string]$Seed
    )

    Invoke-LoggedCommand -Name "validation_report" -FilePath $PythonFile -Arguments @(
        "scripts\build_validation_report.py",
        "--run-dir", $RunDir
    )

    Get-ChildItem -LiteralPath $RunDir -Recurse |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize |
        Out-String -Width 240 |
        Set-Content -LiteralPath (Join-Path $RunDir "result_tree.txt") -Encoding UTF8

    Write-Manifest -Status "SUCCESS"
    Write-Host ""
    Write-Host "Experiment run complete: $RunDir"
    if ($OpenOutput) {
        Invoke-Item -LiteralPath $RunDir
    }
}
catch {
    Write-Manifest -Status "FAILED" -Message $_.Exception.Message
    Get-ChildItem -LiteralPath $RunDir -Recurse -ErrorAction SilentlyContinue |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize |
        Out-String -Width 240 |
        Set-Content -LiteralPath (Join-Path $RunDir "result_tree.txt") -Encoding UTF8
    Write-Error $_.Exception.Message
    exit 1
}
