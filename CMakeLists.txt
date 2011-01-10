PROJECT(multigrid)

ADD_EXECUTABLE(multigrid multigridsolver0.cpp buffer.cpp clcontextloader.cpp main.cpp)
target_link_libraries(multigrid OpenCL pthread)

