param([string]$BasePath = '/')
$ErrorActionPreference = 'Stop'
if ($BasePath -notmatch '^/([^?#\\]*/)?$') {
    throw 'BasePath deve iniciar e terminar com /, por exemplo / ou /bytecraft/.'
}
$backend = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$frontend = [IO.Path]::GetFullPath((Join-Path $backend '../frontend'))
$dist = Join-Path $frontend 'dist'
$webapp = [IO.Path]::GetFullPath((Join-Path $backend 'src/main/webapp'))
$expected = [IO.Path]::GetFullPath((Join-Path $backend 'src/main/webapp'))
if ($webapp -ne $expected -or -not $webapp.StartsWith($backend + [IO.Path]::DirectorySeparatorChar)) {
    throw 'Destino de sincronizacao fora do backend.'
}
$previousBase = $env:VITE_BASE_PATH
Push-Location $frontend
try {
    $env:VITE_BASE_PATH = $BasePath
    & npm.cmd ci --include=dev --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'npm ci falhou.' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'npm run build falhou; webapp nao foi alterada.' }
    if (-not (Test-Path -LiteralPath (Join-Path $dist 'index.html'))) { throw 'dist/index.html ausente.' }
    # webapp is exclusively generated. Never copy the dist directory itself.
    if (Test-Path -LiteralPath $webapp) { Remove-Item -LiteralPath $webapp -Recurse -Force }
    New-Item -ItemType Directory -Path $webapp -Force | Out-Null
    Get-ChildItem -LiteralPath $dist -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $webapp -Recurse -Force
    }
    # Maven WAR staging can otherwise retain assets removed by a later Vite build.
    $staging = [IO.Path]::GetFullPath((Join-Path $backend 'target/bytecraft'))
    if (-not $staging.StartsWith($backend + [IO.Path]::DirectorySeparatorChar)) { throw 'Staging invalido.' }
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
} finally {
    Pop-Location
    $env:VITE_BASE_PATH = $previousBase
}
