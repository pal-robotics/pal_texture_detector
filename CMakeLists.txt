cmake_minimum_required(VERSION 2.8.3)
project(pal_texture_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  image_transport
  pal_detection_msgs
  geometry_msgs  
)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pal_texture_detector
  CATKIN_DEPENDS roscpp roslib pal_detection_msgs geometry_msgs cv_bridge image_transport
)

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(pal_texture_detector src/texture_detector.cpp)
target_link_libraries(pal_texture_detector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(pal_texture_detector_node src/node.cpp)
add_dependencies(pal_texture_detector_node pal_detection_msgs_generate_messages_cpp)
target_link_libraries(pal_texture_detector_node pal_texture_detector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


install(TARGETS pal_texture_detector 
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(TARGETS pal_texture_detector_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}) 

install(DIRECTORY launch config objects 
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

