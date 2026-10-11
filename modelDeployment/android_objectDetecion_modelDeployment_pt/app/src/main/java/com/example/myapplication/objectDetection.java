package com.example.myapplication;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class objectDetection extends AppCompatActivity implements View.OnClickListener,Runnable {

    public List<String> classes;
    private Button btn_open;
    private Button btn_detect;
    private ImageView imageView;
    private Spinner select_model;
    private Spinner select_conf_threshold;
    private Spinner select_iou_threshold;
    private ProgressBar progressBar;
    private ActivityResultLauncher<Intent> register;
    private ArrayAdapter<CharSequence> adapter = null;
    private String modelName = "yolov5s";
    private Uri uri;
    private static String[] model_name_list = {"yolov5s","ssdlite320_mobilenet_v3_large"};
    private static ArrayList<Float>  threshold;
    private TextView displayResult;
    private Button btn_video;
    private Context context = this;
    private Float conf_threshold;
    private Float iou_threshold;
    private int img_size = 320;
    //根据加载的模型和选择的图像进行前向推理
    private Bitmap mBitmap = null;

    private float[][] boxes;
    private float[] scores;
    private long[] labels;
    private float[][] outputs;
    private int mOutputRow = 100;
    private Module model = null;

    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    classifyUtils utls = new classifyUtils();
    objectDetectUtils det_utls = new objectDetectUtils();
    objectResultView mResultView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_object_detection);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        imageView = findViewById(R.id.detectImageView);
        mResultView = findViewById(R.id.imageView2);
        mResultView.setVisibility(View.INVISIBLE);

        btn_open = findViewById(R.id.btn_open);
        btn_detect = findViewById(R.id.btn_detect);
        select_model = findViewById(R.id.select_model);
        progressBar = findViewById(R.id.progressBar);
        select_conf_threshold = findViewById(R.id.select_conf_threshold);
        select_iou_threshold = findViewById(R.id.select_iou_threshold);
        btn_video = findViewById(R.id.btn_video);

        threshold = new ArrayList<>();
        for (Float i = 0.25f; i <= 0.80f; i += 0.05f) {
            threshold.add(i);
        }

        //最开始设置图像
        //    String filePath = assertFilePath(this,"cat.png");
//            File file = new File(filePath);
//            uri = Uri.fromFile(file);
//            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open("cat.png"));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
        imageView.setImageBitmap(mBitmap);
        System.out.println("setmbitmap调用一次");

        //监听按钮
        btn_open.setOnClickListener(this);
        btn_detect.setOnClickListener(this);
        btn_video.setOnClickListener(this);

        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult o) {
                        Intent intent = o.getData();
                        if (intent != null){
                            uri = intent.getData();
                            imageView.setImageURI(uri);
                        }
                    }

                    private ContentResolver getContentResolver() {
                        return null;
                    }
                });

        //字符串给到适配器中
        adapter = new ArrayAdapter<CharSequence>(this,
                android.R.layout.simple_spinner_dropdown_item,model_name_list);
        select_model.setAdapter(adapter);

        //获取下拉菜单中的值
        select_model.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                modelName = adapterView.getItemAtPosition(i).toString();

                //根据选择的模型来加载当前模型和类别文件
                if(modelName.length() > 0){
                    System.out.println("select model is : " + modelName);
                    //读取类别文件
                    //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
                    try {
                        if(modelName.equals("ssdlite320_mobilenet_v3_large")){
                            img_size = 320;
                            classes = utls.readClassesFile(context, "pytorch_classes.txt");
                            model = utls.loadModel(context,"ssdlite320_mobilenet_v3_large.pt",true);
                        }else if(modelName.equals("yolov5s")){
                            img_size = 640;
                            mOutputRow = 25200;
                            classes = utls.readClassesFile(context, "yolov5_classes.txt");
                            model = utls.loadModel(context,"yolov5s.torchscript.pt",true);
                        }
                        System.out.println("classes size: " + classes.size());
                        objectDetectUtils.mClasses = classes;
//                       for(String className : classes){
//                           System.out.println("className: " + className);
//                       }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        //字符串给到适配器中
        ArrayAdapter<Float> adapter_conf = new ArrayAdapter<Float>(this,
                android.R.layout.simple_spinner_dropdown_item,threshold);
        select_conf_threshold.setAdapter(adapter_conf);
        //获取下拉菜单中的值
        select_conf_threshold.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                conf_threshold = (Float) adapterView.getItemAtPosition(i);
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        //字符串给到适配器中
        ArrayAdapter<Float> adapter_iou = new ArrayAdapter<Float>(this,
                android.R.layout.simple_spinner_dropdown_item,threshold);
        select_iou_threshold.setAdapter(adapter_iou);
        //获取下拉菜单中的值
        select_iou_threshold.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                iou_threshold = (Float) adapterView.getItemAtPosition(i);
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });
    }

    //根据预测的结果返回最大类别概率的索引
    public void predict(Tensor inputTensor){
        // 创建输入格式特征和输出格式结果 [1,25200,85]
        final IValue[] outputTensor = model.forward(IValue.from(inputTensor)).toTuple();
//        // 获取 outputTensor 的维度和大小
        final Tensor output = outputTensor[0].toTensor();
        long[] sizes = output.shape();
        int numDimensions = sizes.length;
        float[][][] outputData = new float[(int) sizes[0]][(int) sizes[1]][(int) sizes[2]];

        float[] flatData = output.getDataAsFloatArray();
        int index = 0;
        for (int i = 0; i < sizes[0]; i++) {
            if (numDimensions == 3) {
                outputData[i] = new float[(int) sizes[1]][(int) sizes[2]];
                for (int j = 0; j < sizes[1]; j++) {
                    for (int k = 0; k < sizes[2]; k++) {
                        outputData[i][j][k] = flatData[index++];
                    }
                }
            }
        }

        if(modelName.equals("yolov5s")){
            outputs = outputData[0];
        }else if(modelName.equals("ssdlite320_mobilenet_v3_large")){

        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.btn_open:
                mResultView.setVisibility(View.INVISIBLE);
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                register.launch(intent);
                break;
            case R.id.btn_detect:
                try {
                    if(uri != null){
                        mBitmap = getBitmapFromUri(this,uri);
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                progressBar.setVisibility(ProgressBar.VISIBLE);
                btn_detect.setEnabled(false);

                mImgScaleX = (float)mBitmap.getWidth() / img_size;
                mImgScaleY = (float)mBitmap.getHeight() / img_size;
                //如果宽度比高度值更大，则水平方向按照宽度的比率进行缩放；否则按照高度的比率进行缩放
                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)imageView.getWidth() / mBitmap.getWidth() :
                        (float)imageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)imageView.getHeight() / mBitmap.getHeight() :
                        (float)imageView.getWidth() / mBitmap.getWidth());
                //缩放之后的高宽和指定画布大小之间的差距
                mStartX = (imageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (imageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                System.out.println("mBitmap width: " + (float)mBitmap.getWidth());
                System.out.println("mBitmap height: " + (float)mBitmap.getHeight());
                System.out.println("img_size width: " + (float)img_size);
                System.out.println("img_size height: " + (float)img_size);
                System.out.println("imageView width: " + (float)imageView.getWidth());
                System.out.println("imageView height: " + (float)imageView.getHeight());

                Thread thread = new Thread(objectDetection.this);
                thread.start();

                break;
            case R.id.btn_video:
                Intent intentVideo = new Intent(objectDetection.this, objectDetectionVideo.class);
                intentVideo.putExtra("modelName",modelName);
                intentVideo.putExtra("imgSize",img_size);
                intentVideo.putExtra("confThreshold",conf_threshold);
                intentVideo.putExtra("iouThreshold",iou_threshold);
                System.out.println("video start");
                register.launch(intentVideo);
//                startActivity(intentVideo);
                break;
            default:
                break;
        }
    }

    @Override
    public void run() {
        ArrayList<Result> results;
        Tensor imgProcess = utls.preprocessImage(mBitmap,NO_MEAN_RGB,NO_STD_RGB,
                                                      img_size,img_size,true);
        predict(imgProcess);

        if(modelName.equals("ssdlite320_mobilenet_v3_large")){
            results = det_utls.myOutputsToNMSPredictions(boxes,scores,labels,
                        mImgScaleX, mImgScaleY,mIvScaleX, mIvScaleY, mStartX, mStartY,conf_threshold,iou_threshold);
        }else if(modelName.equals("yolov5s")){
            results =  det_utls.outputsToNMSPredictions(outputs,mOutputRow, mImgScaleX, mImgScaleY,
                                                mIvScaleX, mIvScaleY, mStartX, mStartY,conf_threshold,iou_threshold);
        } else {
            results = null;
        }
        runOnUiThread(() -> {
            btn_detect.setEnabled(true);
            btn_detect.setText(getString(R.string.detect));
            progressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });
    }

    public Bitmap getBitmapFromUri(Context context, Uri uri) throws IOException {
        //context.getContentResolver()：获取应用的 content resolver，它允许访问和操作设备上的数据（如文件、数据库、内容提供者等）。
        //ParcelFileDescriptor 对象，该对象表示文件的底层文件描述符
//        ParcelFileDescriptor parcelFileDescriptor =
//                context.getContentResolver().openFileDescriptor(uri, "r");
//
//        if (parcelFileDescriptor == null) throw new FileNotFoundException("Failed to load image.");
////        从 ParcelFileDescriptor 对象中获取文件描述符。FileDescriptor 是一个指向打开文件的句柄，可以与文件相关的操作一起使用。
//        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
////        使用文件描述符解码图像并将其转换为 Bitmap 对象。这个方法能够处理不同格式的图像（如 JPEG、PNG等）
//        Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);
//        parcelFileDescriptor.close();
        Bitmap image = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
        return image;
    }

    //获取图像路径
    public static String assertFilePath(Context context, String assertName) throws IOException {
        File file = new File(context.getFilesDir(),assertName);
        if(file.exists() && file.length() > 0){
            return file.getAbsolutePath();
        }
        try {
            InputStream inputStream = context.getAssets().open(assertName);
            OutputStream outputStream = new FileOutputStream(file);
            byte [] buffer = new byte[4 * 1024];
            int read;
            while((read = inputStream.read(buffer)) != -1){
                outputStream.write(buffer,0,read);
            }
            outputStream.flush();
        }catch (IOException e){
            e.printStackTrace();
        }
        return file.getAbsolutePath();
    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
        super.onPointerCaptureChanged(hasCapture);
    }
}