@echo OFF

set CONDAPATH=C:\Users\Hazman\anaconda3

set ENVNAME=py3.8

echo Running data...!
if %ENVNAME%==base (call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%) else (call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%\envs\%ENVNAME%)

python main.py --source bentham --arch bluche --image %1

rem exit /b

rem move nul 2>&0

call conda deactivate