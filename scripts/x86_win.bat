set VS_BIN_PATH = "C://Program Files (x86)/Microsoft Visual Studio 14.0/Common7/IDE/devenv"
if exist build_win( rd /s /q build_win)
mkdir build_win
cd build_win
cmake .. -G "Visual Studio 14 2015 Win64"  -DCMAKE_BUILD_TYPE="Debug" -DARCH="x86_64" -DTHINKER_SHARED_LIB=ON  -DTHINKER_DUMP=OFF  -DTHINKER_USE_VENUS=ON
cd ..
pause