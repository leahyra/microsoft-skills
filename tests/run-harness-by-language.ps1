#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run harness evaluations on all skills for a specified language.

.DESCRIPTION
Discovers all skills matching a language pattern (e.g., *-rust, *-py, *-dotnet, *-ts, *-java)
and runs both mock and real Copilot harness evaluations for each skill.

.PARAMETER Language
The language suffix to filter skills by (e.g., 'rust', 'py', 'dotnet', 'ts', 'java').
Case-insensitive.

.PARAMETER Mock
If specified, only run mock harness evaluations (skips real Copilot).

.PARAMETER Real
If specified, only run real Copilot evaluations (skips mock).

.PARAMETER ShowDetails
Show detailed output for each scenario.

.PARAMETER OutputFile
Optional path to save results as JSON. Defaults to stdout only.

.EXAMPLE
# Run both mock and real harness for all Rust skills
.\run-harness-by-language.ps1 -Language rust

# Run only mock harness for Python skills
.\run-harness-by-language.ps1 -Language py -Mock

# Run real harness with details output and save results
.\run-harness-by-language.ps1 -Language dotnet -Real -ShowDetails -OutputFile results.json

.NOTES
Requires pnpm to be installed and in PATH.
Must be run from the tests directory or with proper CWD setup.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Language,
    
    [switch]$Mock,
    [switch]$Real,
    [switch]$ShowDetails,
    [string]$OutputFile,
    [ValidateRange(1, 100)]
    [int]$MaxConcurrent = 1  # Sequential by default; set higher for parallel (use with caution)
)

# Normalize language
$Language = $Language.ToLower().TrimStart('*-').TrimEnd('-*')

# Determine which modes to run
$runMock = $true
$runReal = $true
if ($Mock -and -not $Real) { $runReal = $false }
if ($Real -and -not $Mock) { $runMock = $false }

# Find repository root
function Find-RepoRoot {
    $current = Get-Location
    for ($i = 0; $i -lt 5; $i++) {
        if (Test-Path (Join-Path $current ".github/skills")) {
            return $current
        }
        if (Test-Path (Join-Path $current "tests/scenarios")) {
            return Split-Path $current
        }
        $current = Split-Path $current
    }
    return $PWD
}

$repoRoot = Find-RepoRoot
$testsDir = Join-Path $repoRoot "tests"
$scenariosDir = Join-Path $testsDir "scenarios"

if (-not (Test-Path $scenariosDir)) {
    Write-Host "Error: scenarios directory not found at $scenariosDir" -ForegroundColor Red
    exit 1
}

# Discover skills
$skillPattern = "*-$Language"
$skills = @()
if (Test-Path $scenariosDir) {
    $skills = Get-ChildItem $scenariosDir -Directory `
    | Where-Object { $_.Name -like $skillPattern } `
    | Select-Object -ExpandProperty Name `
    | Sort-Object
}

if ($skills.Count -eq 0) {
    Write-Host "No skills found matching pattern: $skillPattern" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($skills.Count) skill(s) for language '$Language':" -ForegroundColor Cyan
$skills | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
Write-Host ""

# Initialize results tracking
$results = @{
    language  = $Language
    timestamp = Get-Date -AsUTC
    skills    = @()
    summary   = @{
        mock = @{ passed = 0; failed = 0; errors = @() }
        real = @{ passed = 0; failed = 0; errors = @() }
    }
}

# Run harness for each skill
Set-Location $testsDir

foreach ($skill in $skills) {
    $skillResult = @{
        name = $skill
        mock = $null
        real = $null
    }
    
    Write-Host "Evaluating: $skill" -ForegroundColor Yellow
    Write-Host "-" * 60
    
    # Mock harness
    if ($runMock) {
        Write-Host "  [MOCK] Running..." -ForegroundColor Gray -NoNewline
        try {
            $mockOutput = & pnpm harness $skill --mock 2>&1
            $mockSuccess = $LASTEXITCODE -eq 0
            
            if ($mockSuccess) {
                Write-Host " ✓ PASS" -ForegroundColor Green
                $results.summary.mock.passed++
                $skillResult.mock = @{
                    status = "PASS"
                    output = $mockOutput
                }
            }
            else {
                Write-Host " ✗ FAIL" -ForegroundColor Red
                $results.summary.mock.failed++
                $results.summary.mock.errors += $skill
                $skillResult.mock = @{
                    status = "FAIL"
                    output = $mockOutput
                }
            }
            
            if ($ShowDetails) {
                Write-Host $mockOutput -ForegroundColor DarkGray
            }
        }
        catch {
            Write-Host " ✗ ERROR" -ForegroundColor Red
            $results.summary.mock.failed++
            $results.summary.mock.errors += "$skill (Exception: $_)"
            $skillResult.mock = @{
                status = "ERROR"
                error  = $_.Exception.Message
            }
        }
    }
    
    # Real harness
    if ($runReal) {
        Write-Host "  [REAL]  Running..." -ForegroundColor Gray -NoNewline
        try {
            $realOutput = & pnpm harness $skill 2>&1
            $realSuccess = $LASTEXITCODE -eq 0
            
            if ($realSuccess) {
                Write-Host " ✓ PASS" -ForegroundColor Green
                $results.summary.real.passed++
                $skillResult.real = @{
                    status = "PASS"
                    output = $realOutput
                }
            }
            else {
                Write-Host " ✗ FAIL" -ForegroundColor Red
                $results.summary.real.failed++
                $results.summary.real.errors += $skill
                $skillResult.real = @{
                    status = "FAIL"
                    output = $realOutput
                }
            }
            
            if ($ShowDetails) {
                Write-Host $realOutput -ForegroundColor DarkGray
            }
        }
        catch {
            Write-Host " ✗ ERROR" -ForegroundColor Red
            $results.summary.real.failed++
            $results.summary.real.errors += "$skill (Exception: $_)"
            $skillResult.real = @{
                status = "ERROR"
                error  = $_.Exception.Message
            }
        }
    }
    
    Write-Host ""
    $results.skills += $skillResult
}

# Summary Report
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "EVALUATION SUMMARY - Language: $Language" -ForegroundColor Cyan
Write-Host "=" * 60

if ($runMock) {
    $mockTotal = $results.summary.mock.passed + $results.summary.mock.failed
    $mockPassRate = if ($mockTotal -gt 0) { [math]::Round(($results.summary.mock.passed / $mockTotal) * 100, 1) } else { 0 }
    
    Write-Host ""
    Write-Host "MOCK MODE:" -ForegroundColor Yellow
    Write-Host "  Passed:  $($results.summary.mock.passed)/$mockTotal ($mockPassRate%)" -ForegroundColor $(if ($results.summary.mock.failed -eq 0) { "Green" } else { "Yellow" })
    if ($results.summary.mock.errors.Count -gt 0) {
        Write-Host "  Failed skills: $($results.summary.mock.errors -join ', ')" -ForegroundColor Red
    }
}

if ($runReal) {
    $realTotal = $results.summary.real.passed + $results.summary.real.failed
    $realPassRate = if ($realTotal -gt 0) { [math]::Round(($results.summary.real.passed / $realTotal) * 100, 1) } else { 0 }
    
    Write-Host ""
    Write-Host "REAL COPILOT MODE:" -ForegroundColor Yellow
    Write-Host "  Passed:  $($results.summary.real.passed)/$realTotal ($realPassRate%)" -ForegroundColor $(if ($results.summary.real.failed -eq 0) { "Green" } else { "Yellow" })
    if ($results.summary.real.errors.Count -gt 0) {
        Write-Host "  Failed skills: $($results.summary.real.errors -join ', ')" -ForegroundColor Red
    }
}

# Save results if requested
if ($OutputFile) {
    try {
        $results | ConvertTo-Json -Depth 10 | Out-File -FilePath $OutputFile -Encoding UTF8
        Write-Host ""
        Write-Host "Results saved to: $OutputFile" -ForegroundColor Green
    }
    catch {
        Write-Host "Warning: Could not save results to $OutputFile - $_" -ForegroundColor Yellow
    }
}

# Final status
$totalFailed = $results.summary.mock.failed + $results.summary.real.failed
if ($totalFailed -gt 0) {
    Write-Host ""
    Write-Host "Some evaluations failed. Check output above." -ForegroundColor Red
    exit 1
}
else {
    Write-Host ""
    Write-Host "All evaluations passed!" -ForegroundColor Green
    exit 0
}
