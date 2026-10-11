package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;


public class segmentationVideo extends AbstractCameraXActivity {

    classifyUtils cls_utils = new classifyUtils();

    private Context context = this;
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

    private String modelName;
    private Integer img_size;
    private List<List<Integer>> platte;
    private float[][][][]out;
    private float[][][][]aux;

    private Module model;


    private void configuration(){
        modelName = getIntent().getStringExtra("modelName");
        img_size = getIntent().getIntExtra("imgSize",0);
        //根据选择的模型来加载当前模型和类别文件
        if(modelName.length() > 0){
            System.out.println("video select model is : " + modelName);
            //读取类别文件
            //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
            try {
                if(modelName.equals("deeplabv3_mobilenet_v3_large")){
                    img_size = 512;
                    platte = cls_utils.readClassesFile(context, "platte.txt");
                    model = cls_utils.loadModel(context,"deeplabv3_mobilenet_v3_large.pt");
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_segmentation_video;
    }

    @Override
    protected void applyToUiAnalyzeImageResult(Bitmap bitmap) {
        ImageView imageView = findViewById(R.id.object_segmentation_view);
        imageView.setImageBitmap(bitmap);
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
    public void predict(Tensor inputData) {
        final Tensor outputTensor = model.forward(IValue.from(inputData)).toTensor();
//        final Tensor output = outputTensor[0].toTensor();
        final Tensor output = outputTensor;

        long[]sizes = output.shape();
        int numDimensions = sizes.length;
        System.out.println("output dimensions: " + numDimensions);

        int N = (int)sizes[0];
        int C = (int)sizes[1];
        int H = (int)sizes[2];
        int W = (int)sizes[3];
        out = new float[N][C][H][W];

        float[]flatData = output.getDataAsFloatArray();
        int index = 0;
        for(int n = 0 ; n < N; n++){
            for(int c = 0 ; c < C; c++){
                for(int i = 0 ; i < H; i++){
                    for(int j = 0 ; j < W; j++){
                        out[n][c][i][j] = flatData[index ++];
                    }
                }
            }
        }
        System.out.println("predict is done!");
    }

    @Override
    @WorkerThread
    @Nullable
    protected Bitmap analyzeImage(ImageProxy image, int rotationDegrees) {
        configuration();

        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        float imgScaleX = (float)bitmap.getWidth() / img_size;
        float imgScaleY = (float)bitmap.getHeight() / img_size;

        Tensor imgProcess = cls_utils.preprocessImage(bitmap,NO_MEAN_RGB,
                NO_STD_RGB,img_size,img_size,true);
        predict(imgProcess);


        Bitmap resizedBitmap = null;
        // 根据模型输入大小对图像进行缩放;第四个参数 true：指示是否使用双线性滤波来进行缩放。开启后能获得更好的图片质量
        resizedBitmap = Bitmap.createScaledBitmap(bitmap, img_size, img_size, true);
        //根据预测的结果对应每一个像素的21类别的最大类别概率所对应的类别mask
        int [][]mask = cls_utils.getMask(out);
        //得到对应的掩码之后需要根据最大类别概率给预测的区域添加模版颜色
        bitmap = cls_utils.getPlatteBitmap(mask,resizedBitmap,platte);

        int dst_img_size_w = (int)(img_size * imgScaleX);
        int dst_img_size_h = (int)(img_size * imgScaleY);
        resizedBitmap = Bitmap.createScaledBitmap(bitmap,
                dst_img_size_w,dst_img_size_h, true);
        Bitmap finalResizedBitmap = resizedBitmap;
        return finalResizedBitmap;
    }
}