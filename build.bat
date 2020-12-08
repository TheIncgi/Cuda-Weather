@echo off

goto %1

:vectorAdd
echo ********* Vector Add ********
nvcc -ptx JCudaVectorAddKernal.cu -o JCudaVectorAddKernal.ptx
goto exit



:weatherSim
echo ********* Weather Sim ********
nvcc -ptx WeatherSim.cu -o WeatherSim.ptx

:exit