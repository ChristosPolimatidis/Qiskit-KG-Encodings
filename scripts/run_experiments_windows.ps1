[CmdletBinding()]
param(
    [ValidateSet("light", "medium", "hard")]
    [string]$Profile = "light",

    [string]$ResultsRoot = "results\runs",

    [int]$Seed = 12345,

    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$Profiles = @{
    light = [ordered]@{
        Shots = 1000
        Repetitions = 1
        DatasetSize = 6
        SyntheticSizes = @(100)
        RealFiles = @()
        MaxRealTriples = 0
        TimeoutSeconds = 300
        ScalingRequired = $true
        ScalingExtraArgs = @("--skip-real", "--no-compute-decomposed-metrics", "--no-compute-transpiled-metrics")
    }
    medium = [ordered]@{
        Shots = 10000
        Repetitions = 3
        DatasetSize = 1000
        SyntheticSizes = @(100, 1000)
        RealFiles = @("Aristotle.xml", "exampleV3.ttl", "productsSmall.rdf")
        MaxRealTriples = 500
        TimeoutSeconds = 1200
        ScalingRequired = $true
        ScalingExtraArgs = @("--max-metric-qubits", "14", "--no-compute-decomposed-metrics")
    }
    hard = [ordered]@{
        Shots = 10000
        Repetitions = 5
        DatasetSize = 10000
        SyntheticSizes = @(100, 1000, 5000, 10000)
        RealFiles = @("Aristotle.xml", "exampleV3.ttl", "productsSmall.rdf", "DecodedOntologies_V2.ttl")
        MaxRealTriples = 2000
        TimeoutSeconds = 2400
        ScalingRequired = $false
        ScalingExtraArgs = @("--max-metric-qubits", "14")
    }
}

$Config = $Profiles[$Profile]
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ScriptCommandLine = $MyInvocation.Line
if ([System.IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRootPath = [System.IO.Path]::GetFullPath($ResultsRoot)
}
else {
    $ResultsRootPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $ResultsRoot))
}
$RunDir = Join-Path $ResultsRootPath "${Timestamp}_${Profile}"
$ZipPath = "$RunDir.zip"

$OutputFolders = @("logs", "tables", "plots", "json", "raw", "summary", "circuits", "histograms")
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
foreach ($Folder in $OutputFolders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $Folder) | Out-Null
}

$CommandLogPath = Join-Path $RunDir "command_log.jsonl"
$ManifestPath = Join-Path $RunDir "run_manifest.json"
$RunConfigPath = Join-Path $RunDir "run_config.json"
$EnvironmentInfoPath = Join-Path $RunDir "json\environment_info.json"
$script:CommandRecords = @()
$script:RunStatus = "RUNNING"
$script:StartedAt = (Get-Date).ToString("o")

function Resolve-Python {
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $VenvPython) {
        return (Resolve-Path -LiteralPath $VenvPython).Path
    }
    $PythonCommand = Get-Command python -ErrorAction Stop
    return $PythonCommand.Source
}

function ConvertTo-RelativePath {
    param([string]$Path)
    try {
        return [System.IO.Path]::GetRelativePath($RunDir, $Path)
    }
    catch {
        return $Path
    }
}

function Quote-CommandArgument {
    param([string]$Value)
    if ($null -eq $Value) {
        return '""'
    }
    if ($Value -match '^[A-Za-z0-9_\-\.\\/:=]+$') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Format-CommandText {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    $Parts = @((Quote-CommandArgument $FilePath))
    foreach ($Argument in $Arguments) {
        $Parts += Quote-CommandArgument $Argument
    }
    return ($Parts -join " ")
}

function Write-CommandRecord {
    param([hashtable]$Record)
    ($Record | ConvertTo-Json -Compress -Depth 12) |
        Add-Content -LiteralPath $CommandLogPath -Encoding UTF8
}

function Write-RunManifest {
    param([string]$Status)
    $CoreManifestPath = Join-Path $RunDir "summary\core_experiment_manifest.json"
    $CoreExperiments = @()
    if (Test-Path -LiteralPath $CoreManifestPath) {
        try {
            $CorePayload = Get-Content -LiteralPath $CoreManifestPath -Raw | ConvertFrom-Json
            if ($null -ne $CorePayload.experiments) {
                $CoreExperiments = @($CorePayload.experiments)
            }
        }
        catch {
            $CoreExperiments = @()
        }
    }
    $FailedRequired = @(
        $script:CommandRecords |
            Where-Object { $_.required -and $_.status -eq "failed" }
    )
    $Manifest = [ordered]@{
        profile = $Profile
        status = $Status
        started_at = $script:StartedAt
        finished_at = if ($Status -ne "RUNNING") { (Get-Date).ToString("o") } else { $null }
        results_folder = $RunDir
        zip_file = if (Test-Path -LiteralPath $ZipPath) { $ZipPath } else { $null }
        seed = $Seed
        shots = $Config.Shots
        repetitions = $Config.Repetitions
        dataset_sizes = $Config.SyntheticSizes
        command_count = @($script:CommandRecords).Count
        failed_required_command_count = @($FailedRequired).Count
        commands = $script:CommandRecords
        experiments = $CoreExperiments
        summary_csv = Join-Path $RunDir "summary\results_summary.csv"
    }
    $Manifest | ConvertTo-Json -Depth 16 |
        Set-Content -LiteralPath $ManifestPath -Encoding UTF8
}

function Write-RunConfig {
    $EnvironmentInfo = @{}
    if (Test-Path -LiteralPath $EnvironmentInfoPath) {
        try {
            $EnvironmentInfo = Get-Content -LiteralPath $EnvironmentInfoPath -Raw | ConvertFrom-Json
        }
        catch {
            $EnvironmentInfo = @{}
        }
    }
    $RunConfig = [ordered]@{
        profile = $Profile
        timestamp = $Timestamp
        results_folder = $RunDir
        results_root = $ResultsRootPath
        python_executable = $PythonPath
        python_version = $EnvironmentInfo.python_version
        package_versions = $EnvironmentInfo.package_versions
        shots = $Config.Shots
        repetitions = $Config.Repetitions
        dataset_sizes = $Config.SyntheticSizes
        real_dataset_files = $Config.RealFiles
        seed = $Seed
        git_commit = $EnvironmentInfo.git_commit
        os = $EnvironmentInfo.os
        command_line = $ScriptCommandLine
        profile_config = $Config
    }
    $RunConfig | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $RunConfigPath -Encoding UTF8
}

function Append-LogSection {
    param(
        [string]$LogPath,
        [string]$Header,
        [string]$SourcePath
    )
    Add-Content -LiteralPath $LogPath -Value ""
    Add-Content -LiteralPath $LogPath -Value "[$Header]"
    if (Test-Path -LiteralPath $SourcePath) {
        Get-Content -LiteralPath $SourcePath -Raw |
            Add-Content -LiteralPath $LogPath
    }
}

function Run-Step {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [bool]$Required = $true,
        [int]$TimeoutSeconds = 0
    )

    $LogPath = Join-Path $RunDir "logs\$Name.log"
    $StdoutPath = Join-Path $RunDir "logs\$Name.stdout.tmp"
    $StderrPath = Join-Path $RunDir "logs\$Name.stderr.tmp"
    $CommandText = Format-CommandText -FilePath $FilePath -Arguments $Arguments
    $Started = Get-Date
    $Record = [ordered]@{
        name = $Name
        command = $CommandText
        cwd = $RepoRoot
        required = [bool]$Required
        timeout_seconds = $TimeoutSeconds
        started_at = $Started.ToString("o")
        finished_at = $null
        runtime_seconds = $null
        exit_code = $null
        status = "running"
        log_file = $LogPath
        error_message = ""
    }

    Set-Content -LiteralPath $LogPath -Encoding UTF8 -Value @(
        "Command: $CommandText",
        "Working directory: $RepoRoot",
        "Started: $($Started.ToString("o"))"
    )

    try {
        $ArgumentText = ($Arguments | ForEach-Object { Quote-CommandArgument $_ }) -join " "
        $Process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentText `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $StdoutPath `
            -RedirectStandardError $StderrPath `
            -WindowStyle Hidden `
            -PassThru

        if ($TimeoutSeconds -gt 0) {
            try {
                Wait-Process -InputObject $Process -Timeout $TimeoutSeconds -ErrorAction Stop
                $Exited = $true
            }
            catch {
                $Exited = $false
            }
        }
        else {
            Wait-Process -InputObject $Process -ErrorAction Stop
            $Exited = $true
        }

        if (-not $Exited) {
            try {
                $Process.Kill($true)
            }
            catch {
                $Process.Kill()
            }
            $Record.exit_code = -1
            $Record.status = "failed"
            $Record.error_message = "Timed out after $TimeoutSeconds seconds."
        }
        else {
            $Process.Refresh()
            $ExitCode = [int]$Process.ExitCode
            $Record.exit_code = $ExitCode
            if ($ExitCode -eq 0) {
                $Record.status = "success"
            }
            else {
                $Record.status = "failed"
                $Record.error_message = "Exited with code $ExitCode."
            }
        }
    }
    catch {
        $Record.exit_code = -1
        $Record.status = "failed"
        $Record.error_message = $_.Exception.Message
    }
    finally {
        $Finished = Get-Date
        Append-LogSection -LogPath $LogPath -Header "stdout" -SourcePath $StdoutPath
        Append-LogSection -LogPath $LogPath -Header "stderr" -SourcePath $StderrPath
        Add-Content -LiteralPath $LogPath -Value ""
        Add-Content -LiteralPath $LogPath -Value "Finished: $($Finished.ToString("o"))"
        Remove-Item -LiteralPath $StdoutPath, $StderrPath -Force -ErrorAction SilentlyContinue
        $Record.finished_at = $Finished.ToString("o")
        $Record.runtime_seconds = [math]::Round(($Finished - $Started).TotalSeconds, 6)
        $script:CommandRecords += $Record
        Write-CommandRecord -Record $Record
        Write-RunManifest -Status "RUNNING"
    }

    if ($Record.status -eq "failed") {
        $Message = "Command failed: $CommandText`nLog path: $LogPath`n$($Record.error_message)"
        if ($Required) {
            throw $Message
        }
        Write-Warning $Message
    }
}

function Run-ZipStep {
    $Name = "zip_results"
    $LogPath = Join-Path $RunDir "logs\$Name.log"
    $ArchivePath = Join-Path $RunDir "*"
    $CommandText = "Compress-Archive -Path $(Quote-CommandArgument $ArchivePath) -DestinationPath $(Quote-CommandArgument $ZipPath) -Force"
    $Started = Get-Date
    $Record = [ordered]@{
        name = $Name
        command = $CommandText
        cwd = $RepoRoot
        required = $true
        timeout_seconds = 0
        started_at = $Started.ToString("o")
        finished_at = $null
        runtime_seconds = $null
        exit_code = $null
        status = "running"
        log_file = $LogPath
        error_message = ""
    }

    Set-Content -LiteralPath $LogPath -Encoding UTF8 -Value @(
        "Command: $CommandText",
        "Working directory: $RepoRoot",
        "Started: $($Started.ToString("o"))",
        "",
        "[stdout]"
    )

    try {
        Compress-Archive -Path $ArchivePath -DestinationPath $ZipPath -Force *>&1 |
            Out-String |
            Add-Content -LiteralPath $LogPath
        $Record.exit_code = 0
        $Record.status = "success"
    }
    catch {
        $Record.exit_code = -1
        $Record.status = "failed"
        $Record.error_message = $_.Exception.Message
        Add-Content -LiteralPath $LogPath -Value ""
        Add-Content -LiteralPath $LogPath -Value "[stderr]"
        Add-Content -LiteralPath $LogPath -Value $_.Exception.Message
    }
    finally {
        $Finished = Get-Date
        Add-Content -LiteralPath $LogPath -Value ""
        Add-Content -LiteralPath $LogPath -Value "Finished: $($Finished.ToString("o"))"
        $Record.finished_at = $Finished.ToString("o")
        $Record.runtime_seconds = [math]::Round(($Finished - $Started).TotalSeconds, 6)
        $script:CommandRecords += $Record
        Write-CommandRecord -Record $Record
        Write-RunManifest -Status "RUNNING"
    }

    if ($Record.status -eq "failed") {
        throw "Command failed: $CommandText`nLog path: $LogPath`n$($Record.error_message)"
    }
}

$PythonPath = Resolve-Python
$env:PYTHONPATH = $RepoRoot
$env:MPLBACKEND = "Agg"

Write-RunManifest -Status "RUNNING"

try {
    Run-Step -Name "environment_check" -FilePath $PythonPath -Arguments @(
        "scripts\write_environment_info.py",
        "--output-path", $EnvironmentInfoPath,
        "--repo-root", $RepoRoot
    ) -Required $true -TimeoutSeconds 120

    Write-RunConfig

    if ($InstallDeps) {
        Run-Step -Name "install_dependencies" -FilePath $PythonPath -Arguments @(
            "-m", "pip", "install", "-r", "requirements.txt"
        ) -Required $true -TimeoutSeconds 900
    }
    else {
        Run-Step -Name "dependency_check" -FilePath $PythonPath -Arguments @(
            "-c",
            "import qiskit, qiskit_aer, rdflib, numpy, matplotlib"
        ) -Required $true -TimeoutSeconds 120
    }

    if (Test-Path -LiteralPath (Join-Path $RepoRoot "tests")) {
        Run-Step -Name "pytest" -FilePath $PythonPath -Arguments @(
            "-m", "pytest", "-q"
        ) -Required $true -TimeoutSeconds 600
    }

    Run-Step -Name "core_experiment_suite" -FilePath $PythonPath -Arguments @(
        "scripts\run_experiment_suite.py",
        "--output-dir", $RunDir,
        "--shots", [string]$Config.Shots,
        "--repetitions", [string]$Config.Repetitions,
        "--seed", [string]$Seed,
        "--dataset", "data\running_example.ttl",
        "--dataset-size", [string]$Config.DatasetSize,
        "--profile", $Profile
    ) -Required $true -TimeoutSeconds $Config.TimeoutSeconds

    $ScalingArgs = @(
        "scripts\run_all_experiments.py",
        "--results-root", (Join-Path $RunDir "raw\scaling"),
        "--synthetic-sizes"
    )
    foreach ($Size in $Config.SyntheticSizes) {
        $ScalingArgs += [string]$Size
    }
    $ScalingArgs += @(
        "--shots", [string]$Config.Shots,
        "--repetitions", [string]$Config.Repetitions,
        "--timeout-seconds", [string]$Config.TimeoutSeconds,
        "--max-real-triples", [string]$Config.MaxRealTriples
    )
    if (@($Config.RealFiles).Count -gt 0) {
        $ScalingArgs += "--real-files"
        foreach ($File in $Config.RealFiles) {
            $ScalingArgs += $File
        }
    }
    foreach ($ExtraArg in $Config.ScalingExtraArgs) {
        $ScalingArgs += $ExtraArg
    }

    Run-Step -Name "scaling_experiments" -FilePath $PythonPath -Arguments $ScalingArgs `
        -Required ([bool]$Config.ScalingRequired) `
        -TimeoutSeconds $Config.TimeoutSeconds

    Run-Step -Name "classical_baselines" -FilePath $PythonPath -Arguments @(
        "scripts\run_classical_baselines.py",
        "--output-dir", (Join-Path $RunDir "raw\classical_baselines"),
        "--seed", [string]$Seed
    ) -Required $true -TimeoutSeconds 300

    Run-Step -Name "collect_outputs" -FilePath $PythonPath -Arguments @(
        "scripts\collect_run_outputs.py",
        "--run-dir", $RunDir,
        "--shots", [string]$Config.Shots,
        "--repetitions", [string]$Config.Repetitions
    ) -Required $true -TimeoutSeconds 300

    Run-ZipStep

    $RequiredFailures = @($script:CommandRecords | Where-Object { $_.required -and $_.status -eq "failed" })
    if (@($RequiredFailures).Count -gt 0) {
        $script:RunStatus = "FAILED"
    }
    else {
        $script:RunStatus = "SUCCESS"
    }
}
catch {
    $script:RunStatus = "FAILED"
    Write-Host "Command failure:"
    Write-Host $_.Exception.Message
}
finally {
    if (Test-Path -LiteralPath $EnvironmentInfoPath) {
        Write-RunConfig
    }
    Write-RunManifest -Status $script:RunStatus
    Write-Host "results folder path: $RunDir"
    Write-Host "zip file path: $ZipPath"
    Write-Host "status: $script:RunStatus"
    if ($script:RunStatus -ne "SUCCESS") {
        exit 1
    }
}
