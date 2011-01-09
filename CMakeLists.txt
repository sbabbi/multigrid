PROJECT(multigrid)

ADD_EXECUTABLE(multigrid buffer.cpp clcontextloader.cpp main.cpp)
target_link_libraries(multigrid OpenCL)

