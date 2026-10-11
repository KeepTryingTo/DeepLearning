# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

cmake_minimum_required(VERSION 3.22.1)
project("app_classification_cpp_ncnn")

#CMAKE_SOURCE_DIR是一个指向CMakeLists.txt文件所在目录的完整路径的变量。这是CMake运行时设定的一个内置变量，指代项目的根目录，也就是不需要设置，直接使用
#ANDROID_ABI是由Android NDK构建系统定义的变量，表示当前构建的应用程序二进制接口（ABI），例如：armeabi-v7a, arm64-v8a, x86, 和 x86_64。
#在构建过程中，Android Studio会通过Gradle设置这个变量，无需手动设置。
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240820-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)

#find_package命令会告诉CMake去查找名为ncnn的库或模块。它会尝试找到该库的相关文件，并将其加载到项目中使用。
#CMake会尝试查找有关ncnn的配置文件，通常是一个名为ncnnConfig.cmake或ncnn-config.cmake的文件。这些文件通常位于ncnn库安装时指定的路径中。
#一旦找到该库的配置文件，CMake会从中获取相关的编译和链接信息，如头文件的路径、库文件的路径等，这些信息会被后续的build过程所使用
find_package(ncnn REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED
        classification.cpp
        objectDetect.cpp
)
target_link_libraries(${CMAKE_PROJECT_NAME} android log)
target_link_libraries(${CMAKE_PROJECT_NAME} ncnn)