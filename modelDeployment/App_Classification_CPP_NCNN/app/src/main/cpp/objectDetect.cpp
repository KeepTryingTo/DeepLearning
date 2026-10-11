#include <jni.h>
#include <string>

#include "net.h"
#include "benchmark.h"
#include "simpleocv.h"

#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>


std::string modelName_det;//选择的模型名称，仅仅是模型名称，不包含后缀名之类的.param,.bin
AAssetManager * manager_det;

static ncnn::Net det_model;
static ncnn::Net det_model_gpu;

#define TAG "KTG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__);
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__);

extern "C"
JNIEXPORT void JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_objectDetection_getModelName(JNIEnv *env,
                             jobject objectDetectThis,
                             jstring model_name) {
    jclass  imageCls = env->GetObjectClass(objectDetectThis);
    jfieldID modelNameID = env->GetFieldID(imageCls,"modelName", "Ljava/lang/String;");

    //将jstring 转换为C++中的string，并且这里已经获得了选择的模型名称
    jboolean  isCopy;
    modelName_det = env->GetStringUTFChars(model_name, &isCopy);
    LOGD("JNI modelName: %s\n",modelName_det.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_objectDetection_loadModel(JNIEnv *env, jobject thiz,
                                                                          jobject asset_manager) {
    //根据选择的模型名称，这里将assert管理权交给manager
    manager_det = AAssetManager_fromJava(env,asset_manager);
    //加载ncnn.param和ncnn.bin模型结构描述文件和权重参数文件
    int ret = det_model.load_param(manager_det,(modelName_det + "_torchscript.ncnn.param").c_str());
    if(ret != 0){
        LOGE("load param model is failed");
        return;
    }

    ret = det_model.load_model(manager_det,(modelName_det + "_torchscript.ncnn.bin").c_str());
    if(ret != 0){
        LOGE("load bin model is failed");
        return;
    }

    //如果设备存在GPU的话，那么就加载GPU版本的
    if(ncnn::get_gpu_count() > 0){
        //使用vulkan作为计算
        det_model_gpu.opt.use_vulkan_compute = true;
        //加载ncnn.param和ncnn.bin模型结构描述文件和权重参数文件
        ret = det_model_gpu.load_param(manager_det,(modelName_det + "_torchscript.ncnn.param").c_str());
        if(ret != 0){
            LOGE("gpu load param model is failed");
            return;
        }

        ret = det_model_gpu.load_model(manager_det,(modelName_det + "_torchscript.ncnn.bin").c_str());
        if(ret != 0){
            LOGE("gpu load bin model is failed");
            return;
        }
    }
}
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_objectDetection_ObjectDetect(JNIEnv *env,
                             jobject thiz,
                             jobject bitmap,
                             jfloatArray no__mean__rgb,
                             jfloatArray no__std__rgb,
                             jint image_size,
                             jboolean is_use_gpu) {
    if(is_use_gpu == true && ncnn::get_gpu_count() == 0){
        LOGE("the is_use_gpu is true,but you don't hvae gpu is avaiable!");
        return nullptr;
    }

    double start_time = ncnn::get_current_time();
    AndroidBitmapInfo  imageInfo;
    AndroidBitmap_getInfo(env,bitmap,&imageInfo);
    //获得图像的高宽
    int width = imageInfo.width;
    int height = imageInfo.height;
    //判断当前的图像是否和指定的图像大小相同
    if(width != image_size || height != image_size){
        return nullptr;
    }
    //判断当前的图像格式是否正确
    if(imageInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888){
        return nullptr;
    }

    //下面的操作就和之前使用ncnn对图像进行推理过程以及代码是差不多的了
    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env,bitmap,ncnn::Mat::PIXEL_BGR);

    //正式进入
    jboolean isCopy;
    jfloat  * jni_float = env->GetFloatArrayElements(no__mean__rgb,&isCopy);
    float mean_value[3];
    jsize size = env->GetArrayLength(no__mean__rgb);

    for(int i = 0 ; i < size; i++){
        mean_value[i] = jni_float[i] * 255.0;
    }
    env->ReleaseFloatArrayElements(no__mean__rgb,jni_float,0);

    jni_float = env->GetFloatArrayElements(no__std__rgb,&isCopy);
    float std_value[3];
    for(int i = 0 ; i < size; i++){
        std_value[i] = 1.0 / jni_float[i];
        std_value[i] = std_value[i] / 255.0;
    }
    env->ReleaseFloatArrayElements(no__std__rgb,jni_float,0);

    //对图像进行归一化操作
    in.substract_mean_normalize(mean_value,std_value);

    ncnn::Extractor ex = det_model.create_extractor();
    if(is_use_gpu){
        ex = det_model_gpu.create_extractor();
    }

    ex.input("in0",in);
    ncnn::Mat out;
    ex.extract("out0",out);

    std::vector<std::vector<float>> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<float>predictions;
    jfloatArray result;

    if(modelName_det.compare("yolov5s") == 0){
        float *data = (float *)out;
        const int rows = 25200;

        for (int i = 0; i < rows; ++i) {
            float confidence = data[4];
            float * classes_scores = data + 5;

            if(confidence > 0.1){
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                std::vector<float>vec;
                vec.push_back(x);
                vec.push_back(y);
                vec.push_back(w);
                vec.push_back(h);

                boxes.push_back(vec);
                confidences.push_back(confidence);

                float maxValue = -99999999;
                int classId = 0;
                for(int k = 5 ; k < 85; k++){
                    if(classes_scores[k - 5] > maxValue){
                        maxValue = classes_scores[k - 5];
                        classId = k - 5;
                    }
                }
                classIds.push_back(classId);
            }
            data += 85;
        }

        for(int i = 0 ; i < boxes.size(); i++){
            for(int j = 0; j < 4; j++){
                predictions.push_back(boxes[i][j]);
            }
            predictions.push_back(confidences[i]);
            predictions.push_back(classIds[i]);
        }

        LOGD("inference time: %lf",ncnn::get_current_time() - start_time);

        result = env->NewFloatArray(predictions.size());
        env->SetFloatArrayRegion(result,0,predictions.size(),predictions.data());
    }else if(modelName_det.compare("ssdlite320_mobilenet_v3_large")){

    }

    return result;
}