@echo off

:: nvcc fatal: cannot find compiler 'cl.exe' in path
:: add D:\software\visual studio\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64 to path

echo compiling %1.cu ****************

nvcc -ptx %1.cu -o ptx/%1.ptx

echo Done!