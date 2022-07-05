cmake_minimum_required(VERSION 3.4)
project(oakd_ros)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(depthai CONFIG REQUIRED)
if(depthai_FOUND)
  message(WARNING "depthai FOUND, Version:" ${depthai_VERSION} " directory: " ${depthai_DIR})
endif()
find_package(roscpp REQUIRED)
include_directories(
  ${roscpp_INCLUDE_DIRS}
)
find_package(roslib REQUIRED)
include_directories(
  ${roslib_INCLUDE_DIRS}
)
find_package(sensor_msgs REQUIRED)
include_directories(
  ${sensor_msgs_INCLUDE_DIRS}
)
find_package(cv_bridge REQUIRED)
include_directories(
  ${cv_bridge_INCLUDE_DIRS}
)
find_package(pcl_conversions REQUIRED)
include_directories(
  ${pcl_conversions_INCLUDE_DIRS}
)
find_package(pcl_ros REQUIRED)
include_directories(
  ${pcl_ros_INCLUDE_DIRS}
)
find_package(message_generation REQUIRED)
include_directories(
  ${message_generation_INCLUDE_DIRS}
)
find_package(message_runtime REQUIRED)
include_directories(
  ${message_runtime_INCLUDE_DIRS}
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  message_generation
  message_runtime
)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
if(OPENCV_FOUND)
  message(WARNING "OpenCV FOUND, Version:" ${OpenCV_VERSION} " directory: " ${OpenCV_INCLUDE_DIRS})
endif()



add_message_files(
  FILES
    bbox.msg
    bboxes.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package()

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/main_ros.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBS} depthai::core depthai::opencv)
