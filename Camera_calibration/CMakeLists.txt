cmake_minimum_required (VERSION 2.6)
PROJECT(camera_calibration)
FIND_PACKAGE( OpenCV REQUIRED )

SET(CMAKE_BUILD_TYPE "Release")

file(GLOB SOURCE_FILES ./*.cpp)

ADD_EXECUTABLE(camera_calibration ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(camera_calibration ${OpenCV_LIBS})

# add the install targets
install (TARGETS camera_calibration DESTINATION ~/bin)
