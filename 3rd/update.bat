@echo off
rmdir /s /q cutlass.origin

git clone https://github.com/NVIDIA/cutlass.git --depth=1 cutlass.origin

rmdir /s /q cutlass
mkdir cutlass
xcopy /E /I cutlass.origin\include cutlass

cd cutlass.origin
git rev-parse HEAD > ../cutlass/readme
cd ..

rmdir /s /q cutlass.origin
