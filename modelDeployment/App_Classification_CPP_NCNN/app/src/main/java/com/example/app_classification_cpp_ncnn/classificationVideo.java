package com.example.app_classification_cpp_ncnn;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;


public class classificationVideo extends AbstractCameraXActivity<classificationVideo.AnalysisResult>  {
    //加载本地库
    static {
        System.loadLibrary("app_classification_cpp_ncnn");
    }

    private ResultView mResultView;
    classifyUtils utls = new classifyUtils();
    public List<String> classes;
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

    AssetManager assetManager;
    public Bitmap bitmap;
    public int [] predictionIndex;
    public String modelName = "custom_model";
    imageClassification imgSingleCls = new imageClassification();

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
            if(modelName.equals("custom_model")){
                try {
                    classes = utls.readClassesFile(context,"class_custom.txt");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                imgSingleCls.getModelName(modelName);
                imgSingleCls.loadModel(getAssets());
            }else if(modelName.equals("mobilenetv3")){
                try {
                    classes = utls.readClassesFile(context,"imagenet_classes.txt");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                imgSingleCls.getModelName(modelName);
                imgSingleCls.loadModel(getAssets());
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

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        configuration();

        bitmap = imgToBitmap(
                image.getImage()
        );
        //对图像缩放到指定的大小
        bitmap = classifyUtils.preprocessImage(bitmap,224,224,true);

        //开始对图像进行分类
        predictionIndex = imgSingleCls.DetectImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,224,false);
        String predictName = "";
        for(int i = 0 ; i < predictionIndex.length; i++) {
            if (i == predictionIndex.length - 1) {
                predictName += classes.get(predictionIndex[i]);
            } else {
                predictName += classes.get(predictionIndex[i]) + "\n";
            }
        }
        System.out.println("predictionIndex size: " + predictionIndex.length);

        return new AnalysisResult(predictName);
    }
}