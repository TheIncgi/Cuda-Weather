@echo off

echo compiling %1.cu ****************

nvcc -ptx %1.cu -o ptx/%1.ptx

echo Done!