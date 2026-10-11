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

import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class classificationVideo extends AbstractCameraXActivity<classificationVideo.AnalysisResult>  {

    private ResultView mResultView;
    classifyUtils utls = new classifyUtils();
    public List<String> classes;
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
        String modelName = getIntent().getStringExtra("modelName");
        System.out.println("video: " + modelName);
        //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
        if (modelName.equals("custom_model")) {
            try {
                classes = utls.readClassesFile(context,"class_custom.txt");
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
            try {
                utls.OnnxModel(context,"best_5_finetune.onnx");
            } catch (OrtException ex) {
                throw new RuntimeException(ex);
            }
        } else if (modelName.equals("mobilenetv3")) {
            try {
                classes = utls.readClassesFile(context,"imagenet_classes.txt");
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
            try {
                utls.OnnxModel(context,"mobilenet_v3_small.onnx");
            } catch (OrtException ex) {
                throw new RuntimeException(ex);
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
    public int[] predict(FloatBuffer inputData) throws OrtException {
        // 创建输入格式特征和输出格式结果
//        FloatBuffer floatBuffer = FloatBuffer.wrap(inputData);

//        System.out.println("floatbuffer size: " + floatBuffer.capacity());
        OnnxTensor inputTensor = OnnxTensor.createTensor(utls.env, inputData,new long []{1,3,224,224});
        OrtSession.Result results = utls.session.run(Collections.singletonMap("input", inputTensor));

//         从输出中提取数据,输出shape = [1,1000]
        OnnxValue pred = results.get("predictions").get();
        float[][] predictions = (float [][]) pred.getValue();

        float[] prediction = utls.softMax(predictions[0]);
        int[] predictionIndex = utls.getTopk(prediction,3);
        return predictionIndex;
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        System.out.println("analyzeImage");
        //根据加载的模型和选择的图像进行前向推理
        Bitmap bitmap = imgToBitmap(image.getImage());
        FloatBuffer imgProcess = utls.preprocessImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,224,224,true);
        configuration();

        String predictName = "";
        try {
            int[] predictionIndex = predict(imgProcess);

            for(int i = 0 ; i < predictionIndex.length; i++){
                if(i == predictionIndex.length - 1){
                    predictName += classes.get(predictionIndex[i]);
                }else{
                    predictName += classes.get(predictionIndex[i]) + "\n";
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return new AnalysisResult(predictName);
    }
}