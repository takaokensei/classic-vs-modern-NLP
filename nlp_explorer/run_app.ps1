# Script PowerShell para executar o NLP Explorer
# Execute com: .\run_app.ps1 ou powershell -ExecutionPolicy Bypass -File .\run_app.ps1

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Iniciando NLP Explorer..." -ForegroundColor Cyan
Write-Host "Verificando ambiente virtual..." -ForegroundColor Yellow

# Tentar diferentes locais para o .venv
$possibleVenvPaths = @(
    (Join-Path $scriptPath "..\.venv\Scripts\python.exe"),  # Raiz do projeto
    (Join-Path $scriptPath ".venv\Scripts\python.exe"),     # Dentro de nlp_explorer
    (Join-Path (Split-Path $scriptPath -Parent) ".venv\Scripts\python.exe")  # Raiz absoluta
)

$venvPython = $null
foreach ($path in $possibleVenvPaths) {
    $resolvedPath = Resolve-Path $path -ErrorAction SilentlyContinue
    if ($resolvedPath -and (Test-Path $resolvedPath)) {
        $venvPython = $resolvedPath.Path
        break
    }
}

if (-not $venvPython) {
    Write-Host "ERRO: Ambiente virtual (.venv) não encontrado!" -ForegroundColor Red
    Write-Host "Caminhos verificados:" -ForegroundColor Yellow
    foreach ($path in $possibleVenvPaths) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
    Write-Host "`nCertifique-se de que o .venv existe no projeto." -ForegroundColor Yellow
    Write-Host "Crie com: python -m venv .venv" -ForegroundColor Cyan
    Read-Host "`nPressione Enter para sair"
    exit 1
}

Write-Host "Ambiente virtual encontrado!" -ForegroundColor Green
Write-Host "Python usado: $venvPython" -ForegroundColor Gray
Write-Host "Executando Streamlit..." -ForegroundColor Cyan

# Executar streamlit
& $venvPython -m streamlit run app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nErro ao executar Streamlit. Exit code: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Pressione Enter para sair"
}

