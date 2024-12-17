#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.19)

#Name your project here
project(rsf3d LANGUAGES CXX CUDA)
enable_language(CUDA)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

#find_package(ZLIB REQUIRED)
find_package(TIRA REQUIRED)

# fix warnings with BOOST and CMake
#set(Boost_NO_WARN_NEW_VERSIONS 1)
find_package(Boost REQUIRED COMPONENTS program_options)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif ( MSVC )

#if(UNIX)
#	find_package(X11)
#endif(UNIX)

configure_file(data/kesm_image.npy kesm_image.npy COPYONLY)
configure_file(data/kesm_centerlines.npy kesm_centerlines.npy COPYONLY)

#create an executable
add_executable(rsf3d
               	kernel.cu
	        	main.cpp
#				cnpy/cnpy.cpp
#				cnpy/cnpy.h
)

include_directories(#${ZLIB_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_BINARY_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}
					${TIRA_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                )

#if(MSVC)
target_link_libraries(rsf3d
#					  ZLIB::ZLIB
                      Boost::program_options
)
#endif(MSVC)

#if(UNIX)
#target_link_libraries(	rsf3d
#					  	ZLIB::ZLIB
#                      	Boost::program_options
#                      	X11
#                    )


#endif(UNIX)

