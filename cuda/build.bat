@echo off

:: nvcc fatal: cannot find compiler 'cl.exe' in path
:: add D:\software\visual studio\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64 to path

:: 11.4.1 expected
nvcc --version
echo[
echo compiling %1.cu ****************

nvcc -ptx %1.cu -o ptx/%1.ptx

echo Done!