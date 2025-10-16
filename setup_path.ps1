# RAGStrict 一键配置脚本 (Windows PowerShell)
# 使用方法: 右键以管理员身份运行PowerShell,然后执行: .\setup_path.ps1

Write-Host "正在配置RAGStrict环境变量..." -ForegroundColor Green

# 获取当前用户的Python Scripts目录
$pythonScripts = Get-ChildItem -Path "$env:LOCALAPPDATA\Packages" -Filter "Python*" -Directory | 
    Get-ChildItem -Filter "LocalCache" | 
    Get-ChildItem -Filter "local-packages" | 
    Get-ChildItem -Filter "Python*" | 
    Get-ChildItem -Filter "Scripts" | 
    Select-Object -First 1 -ExpandProperty FullName

if ($pythonScripts) {
    Write-Host "找到Python Scripts目录: $pythonScripts" -ForegroundColor Cyan
    
    # 获取当前PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    # 检查是否已经存在
    if ($currentPath -like "*$pythonScripts*") {
        Write-Host "PATH已经包含此目录,无需添加" -ForegroundColor Yellow
    } else {
        # 添加到用户PATH
        $newPath = $currentPath + ";" + $pythonScripts
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "成功添加到PATH!" -ForegroundColor Green
    }
    
    # 临时添加到当前会话
    $env:PATH += ";$pythonScripts"
    
    # 验证
    Write-Host "`n正在验证..." -ForegroundColor Cyan
    
    try {
        $version = & rags version 2>&1
        Write-Host "✓ 配置成功! 版本: $version" -ForegroundColor Green
        Write-Host "`n现在可以使用 rags 命令了!" -ForegroundColor Green
        Write-Host "下一步: 执行 'rags init' 初始化项目" -ForegroundColor Cyan
    } catch {
        Write-Host "✗ 验证失败,请重新打开PowerShell后再试" -ForegroundColor Red
        Write-Host "或手动添加到PATH: $pythonScripts" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ 未找到Python Scripts目录" -ForegroundColor Red
    Write-Host "请确保已执行: pip install -e ." -ForegroundColor Yellow
}

Write-Host "`n按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
