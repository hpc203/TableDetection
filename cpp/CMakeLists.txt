project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/predictor.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/include/opencv4")

target_link_libraries(${PROJECT_NAME}
        "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgcodecs.so.4.8.0"
        "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/lib/libopencv_highgui.so.4.8.0"
        "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgproc.so.4.8.0"
        "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/lib/libopencv_core.so.4.8.0"
        "/home/wangbo/libs/opencv-4.8.0/temp_install_dir/lib/libopencv_dnn.so.4.8.0")