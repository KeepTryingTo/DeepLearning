package com.example.myapplication;

import android.annotation.SuppressLint;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.IntentSender;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.database.DatabaseErrorHandler;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.os.UserHandle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.view.TextureView;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ImageProxy;
import androidx.core.app.ActivityCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

//import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


import ai.onnxruntime.*;

public class imageClassification extends AppCompatActivity implements View.OnClickListener {

    public List<String> classes;
    private Button btn_open;
    private Button btn_classify;
    private ImageView imageView;
    private Spinner spinner;
    private ActivityResultLauncher<Intent> register;
    private ArrayAdapter<CharSequence> adapter = null;
    private String modelName = "custom_model";
    private Uri uri;
    private static String[] model_name_list = {"custom_model","mobilenetv3"};
    private TextView displayResult;
    private Button btn_video;
    private Context context = this;

    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
    static float[] NO_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};
//    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
//    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    classifyUtils utls = new classifyUtils();

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_image_classification);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        btn_open = findViewById(R.id.btn_open);
        btn_classify = findViewById(R.id.btn_classify);
        imageView = findViewById(R.id.imageView);
        spinner = findViewById(R.id.spinner);
        displayResult = findViewById(R.id.dislayResult);
        btn_video = findViewById(R.id.btn_video);
        //最开始设置图像
        try {
            String filePath = assertFilePath(this,"cat.png");
            File file = new File(filePath);
            uri = Uri.fromFile(file);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            imageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //监听按钮
        btn_open.setOnClickListener(this);
        btn_classify.setOnClickListener(this);
        btn_video.setOnClickListener(this);

        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult o) {
                Intent intent = o.getData();
                if (intent != null){
                    uri = intent.getData();
                    imageView.setImageURI(intent.getData());
                }
            }
        });

        //字符串给到适配器中
        adapter = new ArrayAdapter<CharSequence>(this,
                android.R.layout.simple_spinner_dropdown_item,model_name_list);
        spinner.setAdapter(adapter);

        //获取下拉菜单中的值
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                modelName = adapterView.getItemAtPosition(i).toString();

                //根据选择的模型来加载当前模型和类别文件
                if(modelName.length() > 0){
                    System.out.println("select model is : " + modelName);
                    //读取类别文件
                    //虽然raw文件夹中的文件默认是可以读取的，但如果你的代码尝试以某种方式（如通过文件路径直接访问）来读取这些文件，可能会遇到权限问题。
                    try {
                        if(modelName.equals("custom_model")){
                            classes = utls.readClassesFile(context,"class_custom.txt");
                            utls.OnnxModel(context,"best_5_finetune.onnx");
                        }else if(modelName.equals("mobilenetv3")){
                            classes = utls.readClassesFile(context,"imagenet_classes.txt");
                            utls.OnnxModel(context,"mobilenet_v3_small.onnx");
                        }
                        System.out.println("classes size: " + classes.size());
//                       for(String className : classes){
//                           System.out.println("className: " + className);
//                       }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    catch (OrtException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
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

    //根据预测的结果返回最大类别概率的索引
    public int[] predict(FloatBuffer inputData) throws OrtException {
        // 创建输入格式特征和输出格式结果
//        FloatBuffer floatBuffer = FloatBuffer.wrap(inputData);

//        System.out.println("floatbuffer size: " + floatBuffer.capacity());
        OnnxTensor inputTensor = OnnxTensor.createTensor(utls.env,inputData ,new long []{1,3,224,224});
        OrtSession.Result results = utls.session.run(Collections.singletonMap("input", inputTensor));

//         从输出中提取数据,输出shape = [1,1000]
        OnnxValue pred = results.get("predictions").get();
        float[][] predictions = (float [][]) pred.getValue();

        float[] prediction = null;
        if(modelName.equals("mobilenetv3")){
            prediction = utls.softMax(predictions[0]);
        }else{
            prediction = predictions[0];
        }
        int[] predictionIndex = utls.getTopk(prediction,3);
        return predictionIndex;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.btn_open:
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                register.launch(intent);
                break;
            case R.id.btn_classify:
                //根据加载的模型和选择的图像进行前向推理
                Bitmap bitmap = null;
                try {
                    bitmap = getBitmapFromUri(this,uri);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                FloatBuffer imgProcess = utls.preprocessImage(bitmap,NO_MEAN_RGB,NO_STD_RGB,
                        224,224,true);
                try {
                    int[] predictionIndex = predict(imgProcess);
                    String predictName = "";
                    for(int i = 0 ; i < predictionIndex.length; i++){
                        if(i == predictionIndex.length - 1){
                            predictName += classes.get(predictionIndex[i]);
                        }else{
                            predictName += classes.get(predictionIndex[i]) + "\n";
                        }
                    }
                    displayResult.setText(predictName);
                } catch (OrtException e) {
                    throw new RuntimeException(e);
                }
                break;
            case R.id.btn_video:
                Intent intentVideo = new Intent(imageClassification.this, classificationVideo.class);
                intentVideo.putExtra("modelName",modelName);
                System.out.println("video start");
//                register.launch(intentVideo);
                startActivity(intentVideo);
                break;
            default:
                break;
        }
    }

    //获取图像路径
    public String assertFilePath(Context context, String assertName) throws IOException {
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