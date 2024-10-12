# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)
project("translate_transformer")

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240820-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)

#find_package命令会告诉CMake去查找名为ncnn的库或模块。它会尝试找到该库的相关文件，并将其加载到项目中使用。
#CMake会尝试查找有关ncnn的配置文件，通常是一个名为ncnnConfig.cmake或ncnn-config.cmake的文件。这些文件通常位于ncnn库安装时指定的路径中。
#一旦找到该库的配置文件，CMake会从中获取相关的编译和链接信息，如头文件的路径、库文件的路径等，这些信息会被后续的build过程所使用
find_package(ncnn REQUIRED)
add_library(${CMAKE_PROJECT_NAME} SHARED translate.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} android log)
target_link_libraries(${CMAKE_PROJECT_NAME} ncnn)

