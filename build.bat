@echo off

echo compiling %1.cu ****************

nvcc -ptx %1.cu -o %1.ptx

echo Done!