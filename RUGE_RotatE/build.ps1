$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceRoot = Join-Path $projectRoot 'src'
$outputRoot = Join-Path $projectRoot 'bin'
$jarPath = Join-Path $projectRoot 'RUGE_RotatE.jar'
$sources = @(Get-ChildItem -LiteralPath $sourceRoot -Filter '*.java' -Recurse | ForEach-Object { $_.FullName })
$jarCommand = Get-Command jar -ErrorAction SilentlyContinue
$jarTool = if ($jarCommand) { $jarCommand.Source } else { $null }

if (-not $jarTool -and $env:JAVA_HOME) {
    $javaHomeJar = Join-Path $env:JAVA_HOME 'bin\jar.exe'
    if (Test-Path -LiteralPath $javaHomeJar) {
        $jarTool = $javaHomeJar
    }
}

if (-not $jarTool) {
    $javaInstallRoot = Join-Path $env:ProgramFiles 'Java'
    if (Test-Path -LiteralPath $javaInstallRoot) {
        $jdk = Get-ChildItem -LiteralPath $javaInstallRoot -Directory -Filter 'jdk*' |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($jdk) {
            $javaHomeJar = Join-Path $jdk.FullName 'bin\jar.exe'
            if (Test-Path -LiteralPath $javaHomeJar) {
                $jarTool = $javaHomeJar
            }
        }
    }
}

if ($sources.Count -eq 0) {
    throw "No Java source files found under $sourceRoot"
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
& javac -encoding UTF-8 -d $outputRoot $sources
if ($LASTEXITCODE -ne 0) {
    throw "Java compilation failed (exit code $LASTEXITCODE)."
}

if (-not $jarTool) {
    throw "jar.exe not found. Set JAVA_HOME to a full JDK installation."
}

& $jarTool cfe $jarPath rotate.train.Program_RUGE_RotatE -C $outputRoot .
if ($LASTEXITCODE -ne 0) {
    throw "JAR creation failed (exit code $LASTEXITCODE)."
}

Write-Output "Created JAR: $jarPath"
