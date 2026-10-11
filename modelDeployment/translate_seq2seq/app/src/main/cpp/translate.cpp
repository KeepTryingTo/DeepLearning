#include <jni.h>
#include <string>

#include "net.h"
#include "android/log.h"
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"

std::string  modelName;
AAssetManager * manager = nullptr;
ncnn::Net model;
ncnn::Net model_gpu;

#define TAG "KTG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__);
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__);

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_translate_1transformer_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
