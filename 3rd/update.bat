@echo off
setlocal enabledelayedexpansion

rem 删除cutlass.origin目录
rd /s /q cutlass.origin

rem 克隆cutlass仓库
git clone https://github.com/NVIDIA/cutlass.git --depth=1 cutlass.origin

rem 删除cutlass目录并创建新的cutlass目录
rd /s /q cutlass
mkdir cutlass

rem 复制cutlass.origin/include到cutlass
robocopy cutlass.origin\include cutlass\include /mir

rem 获取cutlass.origin的HEAD并写入cutlass/readme
for /f "delims=" %%i in ('git -C cutlass.origin rev-parse HEAD') do (
    echo %%i > cutlass\readme
)

rem 删除cutlass.origin目录
rd /s /q cutlass.origin

endlocal
