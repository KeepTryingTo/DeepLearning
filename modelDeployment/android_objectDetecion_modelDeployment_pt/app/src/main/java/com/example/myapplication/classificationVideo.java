package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageProxy;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;


public class classificationVideo extends AbstractCameraXActivity<classificationVideo.AnalysisResult>  {

    private ResultView mResultView;
    private Module model;
    classifyUtils utls = new classifyUtils();
    public List<String> classes;
    private String modelName;
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};
    //用于保存摄像头捕获的帧
    static class AnalysisResult {
        private final String predictions;
        public AnalysisResult(String result) {
            predictions = result;
        }
    }

    Context context = this;

    public void configuration(){
        modelName = getIntent().getStringExtra("modelName");
        System.out.println("video: " + modelName);
        //根据选择的模型来加载当前模型和类别文件
        if(modelName.length() > 0){
            System.out.println("select model is : " + modelName);
            //读取类别文件
            //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
            try {
                if(modelName.equals("custom_model")){
                    classes = utls.readClassesFile(context,"class_custom.txt");
                    model = utls.loadModel(context,"custom_model.pt", false);
                }else if(modelName.equals("mobilenetv3")){
                    classes = utls.readClassesFile(context,"imagenet_classes.txt");
                    model = utls.loadModel(context,"mobilenet_v3_small.pt",false);
                }
                System.out.println("classes size: " + classes.size());

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        System.out.println("video classes size: " + classes.size());
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_classification_video;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        //inflate() 方法被调用来展开（或实例化）ViewStub 指向的布局。
        return ((ViewStub) findViewById(R.id.detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);//在 ViewStub 展开后加载的布局中定义，并用于显示相机的实时预览。
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.predictions);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    //根据预测的结果返回最大类别概率的索引
    public int[] predict(Tensor inputTensor){
        //从输出中提取数据,输出shape = [1,1000]
        final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
//        getDataAsFloatArray() 方法将 outputTensor 中的数据转换为一个一维 float 数组
        float [] pred = outputTensor.getDataAsFloatArray();
        float [] result = null;
        if(modelName.equals("mobilenetv3")){
            result = utls.softMax(pred);
        }else{
            result = pred;
        }
        int[] predictionIndex = utls.getTopk(result,3);
        return predictionIndex;
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        System.out.println("analyzeImage");
        //根据加载的模型和选择的图像进行前向推理
        Bitmap bitmap = imgToBitmap(image.getImage());
        Tensor imgProcess = utls.preprocessImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,224,224,true);
        configuration();

        String predictName = "";

        int[] predictionIndex = predict(imgProcess);

        for(int i = 0 ; i < predictionIndex.length; i++){
            if(i == predictionIndex.length - 1){
                predictName += classes.get(predictionIndex[i]);
            }else{
                predictName += classes.get(predictionIndex[i]) + "\n";
            }
        }

        return new AnalysisResult(predictName);
    }
}