#include <jni.h>
#include <string>

#include "net.h"
#include "benchmark.h"

#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>


std::string modelName;//选择的模型名称，仅仅是模型名称，不包含后缀名之类的.param,.bin
AAssetManager * manager;

static ncnn::Net cls_model;
static ncnn::Net cls_model_gpu;

#define TAG "KTG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__);
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__);

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_imageClassification_getModelName(JNIEnv *env,
                                                                                 jobject imageClassificationThis,
                                                                                 jstring modeName_t) {
    jclass  imageCls = env->GetObjectClass(imageClassificationThis);
    jfieldID modelNameID = env->GetFieldID(imageCls,"modelName", "Ljava/lang/String;");

    //将jstring 转换为C++中的string，并且这里已经获得了选择的模型名称
    jboolean  isCopy;
    modelName = env->GetStringUTFChars(modeName_t, &isCopy);
    LOGD("JNI modelName: %s\n",modelName.c_str());
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_imageClassification_loadModel(JNIEnv *env,
                                          jobject thiz,
                                          jobject asset_manager) {
    //根据选择的模型名称，这里将assert管理权交给manager
    manager = AAssetManager_fromJava(env,asset_manager);
    //加载ncnn.param和ncnn.bin模型结构描述文件和权重参数文件
    int ret = cls_model.load_param(manager,(modelName + ".ncnn.param").c_str());
    if(ret != 0){
        LOGE("load param model is failed");
        return;
    }

    ret = cls_model.load_model(manager,(modelName + ".ncnn.bin").c_str());
    if(ret != 0){
        LOGE("load bin model is failed");
        return;
    }

    //如果设备存在GPU的话，那么就加载GPU版本的
    if(ncnn::get_gpu_count() > 0){
        //使用vulkan作为计算
        cls_model_gpu.opt.use_vulkan_compute = true;
        //加载ncnn.param和ncnn.bin模型结构描述文件和权重参数文件
        ret = cls_model_gpu.load_param(manager,(modelName + ".ncnn.param").c_str());
        if(ret != 0){
            LOGE("gpu load param model is failed");
            return;
        }

        ret = cls_model_gpu.load_model(manager,(modelName + ".ncnn.bin").c_str());
        if(ret != 0){
            LOGE("gpu load bin model is failed");
            return;
        }
    }
}

//用于计算前TopK最大类别概率的索引
std::vector<int> getTopK(std::vector<float> predictions,int topk,int num_classes){
    float * topProbabilities = new float [num_classes];
    //首先向predictionIndexs向量中添加topk个元素作为初始化
    std::vector<int>predictionIndexs;
    for(int i = 0 ; i < topk; i++){
        predictionIndexs.push_back(0);
    }

    //遍历类别数
    for (int i = 0; i < num_classes; i++) {
        //遍历Top-k的插入排序
        for (int j = 0; j < topk; j++) {
            if (predictions[i] >= topProbabilities[j]) {
                // 插入新的概率
                for (int k = topk - 1; k > j; k--) {
                    topProbabilities[k] = topProbabilities[k - 1];
                    predictionIndexs[k] = predictionIndexs[k - 1];
                }
                topProbabilities[j] = predictions[i];
                predictionIndexs[j] = i;
                break;
            }
        }
    }
    for (int i = 0; i < topk; i++) {
        LOGD("prediction index: %d",predictionIndexs[i]);
    }
    return predictionIndexs;
}


extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_app_1classification_1cpp_1ncnn_imageClassification_DetectImage(JNIEnv *env,
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
    std::vector<float>scores;
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
    ncnn::Extractor ex = cls_model.create_extractor();
    if(is_use_gpu){
        ex = cls_model_gpu.create_extractor();
    }

    ex.input("in0",in);
    ncnn::Mat out;
    ex.extract("out0",out);

    scores.resize(out.w);
    for(int i = 0 ; i < out.w; i++){
        scores[i] = out[i];
    }
    //用于保存预测的前TopK个最大类别概率索引
    std::vector<int>predictionIndexs = getTopK(scores,3,out.w);

    LOGD("inference time: %lf",ncnn::get_current_time() - start_time);
    jintArray  resultIndexs = env->NewIntArray(predictionIndexs.size());
    env->SetIntArrayRegion(resultIndexs,0,predictionIndexs.size(),predictionIndexs.data());

    return resultIndexs;
}
