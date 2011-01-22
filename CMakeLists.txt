PROJECT(multigrid)

ADD_EXECUTABLE(multigrid multi_array.cpp rectangularborderhandler.cpp programstate.cpp functionhandler.cpp multigridsolver0.cpp auxiliary.cpp buffer.cpp clcontextloader.cpp main.cpp)
target_link_libraries(multigrid OpenCL pthread)

