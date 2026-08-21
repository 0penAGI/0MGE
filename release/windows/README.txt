To build on Windows:
1. Install Visual Studio 2022 + CMake
2. cd vst && mkdir build && cd build
3. cmake -G "Visual Studio 17 2022" -A x64 ..
4. cmake --build . --config Release
5. Copy build/ZeroMGE_Project_artefacts/Release/VST3/0MGE.vst3
   to C:\Program Files\Common Files\VST3\
