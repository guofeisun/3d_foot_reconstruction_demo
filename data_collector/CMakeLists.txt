cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
set (_projname "collector")
project(${_projname})

find_package(OpenCV REQUIRED)

set(${_projname}_SOURCES
	./src/data_process.h
	./src/data_read.h
	./src/data_process.cpp
	./src/data_read.cpp
	./src/main.cpp
)

include_directories($ENV{OPEN_NI_INCLUDE})# ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
#link_directories($ENV{OPEN_NI_LIB})# ${OpenCV_LIB_DIR} ${Boost_LIBRARY_DIRS})

add_executable(${_projname} ${${_projname}_SOURCES})
set(OPENNI_LIBRARIES libOpenNI.so)
target_link_libraries(${_projname} ${OpenCV_LIBS} ${OPENNI_LIBRARIES})



