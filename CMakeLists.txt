PROJECT(multigrid)

set(SOURCES
			auxiliary.cpp
			buffer.cpp
			clcontextloader.cpp
			functionhandler.cpp
			main.cpp
			multi_array.cpp
			programstate.cpp
			rectangularborderhandler.cpp)

add_executable(multigrid2D ${SOURCES} multigridsolver2D.cpp)
add_executable(multigrid3D ${SOURCES} multigridsolver3D.cpp)

set(CL_SOURCES
        common_kernels.cl
        mg_0.cl
        mg_1.cl
        real.cl)

foreach(file ${CL_SOURCES})
        configure_file(${CMAKE_SOURCE_DIR}/${file} ${CMAKE_BINARY_DIR}/${file} COPYONLY)
endforeach()

set_target_properties(multigrid2D PROPERTIES COMPILE_DEFINITIONS BIDIM)

target_link_libraries(multigrid2D OpenCL pthread)
target_link_libraries(multigrid3D OpenCL pthread)
