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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class segmentationImage extends AppCompatActivity implements Runnable, View.OnClickListener {
    classifyUtils utls = new classifyUtils();
    private ActivityResultLauncher<Intent> register;
    private ArrayAdapter<CharSequence> adapter = null;

    private TextView displayResult;
    private Button btn_video;
    private Context context = this;
    private List<List<Integer>> platte;
    private float mImgScaleX,mImgScaleY;

    private Button btn_open;
    private Button btn_detect;
    private ImageView imageView;
    private Spinner select_model;
    private ProgressBar progressBar;

    private String modelName = "deeplabv3_mobilenet_v3_large";
    private Uri uri = null;
    private int img_size = 512;
    private static String[] model_name_list = {"deeplabv3_mobilenet_v3_large","deeplabv3_mobilenet_v3_large"};
    private float[][][][]out;
    private float[][][][]aux;

    //根据加载的模型和选择的图像进行前向推理
    private Bitmap mBitmap = null;

    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_segmentation_image);
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

        btn_open = findViewById(R.id.btn_open);
        btn_detect = findViewById(R.id.btn_detect);
        select_model = findViewById(R.id.select_model);
        progressBar = findViewById(R.id.progressBar);
        btn_video = findViewById(R.id.btn_video);

        btn_detect.setOnClickListener(this);
        btn_video.setOnClickListener(this);
        btn_open.setOnClickListener(this);

        //设置默认的图像
//        String filePath = null;
//        try {
//            filePath = assertFilePath(this,"cat.png");
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        File file = new File(filePath);
//        uri = Uri.fromFile(file);
//        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
//            Uri bitmap = MediaStore.Images.Media.getContentUri(this.getContentResolver(),uri);
//        }
        try {
            InputStream  inputStream = getAssets().open("cat.png");
            byte[] buffer = new byte[inputStream.available()];
            int bytesRead = inputStream.read(buffer);
            if(bytesRead != -1){
                mBitmap = BitmapFactory.decodeByteArray(buffer,0,bytesRead);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        imageView.setImageBitmap(mBitmap);

        //从相册系统里面获取图像
        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult o) {
                        Intent intent = o.getData();
                        if(intent != null){
                            uri = intent.getData();
                            imageView.setImageURI(uri);
                        }
                    }
                    private ContentResolver getContentResolver(){
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
                        if(modelName.equals("deeplabv3_mobilenet_v3_large")){
                            img_size = 512;
                            platte = utls.readClassesFile(context, "platte.txt");
                            utls.OnnxModel(context,"deeplabv3_mobilenet_v3_large.onnx");
                        }
                        System.out.println("read model and platte is successfully!");
                        System.out.println("platte size : " + platte.size());
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

    //根据预测的结果返回最大类别概率的索引
    public void predict(FloatBuffer inputData) throws OrtException {
        System.out.println("floatbuffer size: " + inputData.capacity());
        OnnxTensor inputTensor = OnnxTensor.createTensor(utls.env,inputData ,new long []{1,3,img_size,img_size});
        OrtSession.Result results = null;
        if(modelName.equals("deeplabv3_mobilenet_v3_large")){
            results = utls.session.run(Collections.singletonMap("input", inputTensor));
        }else{
            results = utls.session.run(Collections.singletonMap("input", inputTensor));
        }
        OnnxValue pred_out = results.get("out").get();
        OnnxValue pred_aux = results.get("aux").get();
        out = (float[][][][])pred_out.getValue();
        aux = (float[][][][])pred_aux.getValue();
        System.out.println("predict is done!");
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.btn_open:
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

                Thread thread = new Thread(segmentationImage.this);
                thread.start();

                break;
            case R.id.btn_video:
                Intent intentVideo = new Intent(segmentationImage.this, segmentationVideo.class);
                intentVideo.putExtra("modelName",modelName);
                intentVideo.putExtra("imgSize",img_size);
                System.out.println("video start");
                register.launch(intentVideo);
                break;
            default:
                break;
        }
    }

    @Override
    public void run() {

        FloatBuffer imgProcess = utls.preprocessImage(mBitmap,NO_MEAN_RGB,NO_STD_RGB,
                img_size,img_size,true);

        try {
            predict(imgProcess);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        Bitmap resizedBitmap = null;
        // 根据模型输入大小对图像进行缩放;第四个参数 true：指示是否使用双线性滤波来进行缩放。开启后能获得更好的图片质量
        resizedBitmap = Bitmap.createScaledBitmap(mBitmap, img_size, img_size, true);
        //根据预测的结果对应每一个像素的21类别的最大类别概率所对应的类别mask
        int [][]mask = utls.getMask(out);
        //得到对应的掩码之后需要根据最大类别概率给预测的区域添加模版颜色
        mBitmap = utls.getPlatteBitmap(mask,resizedBitmap,platte);

        int dst_img_size_w = (int)(img_size * mImgScaleX);
        int dst_img_size_h = (int)(img_size * mImgScaleY);
        resizedBitmap = Bitmap.createScaledBitmap(mBitmap,
                dst_img_size_w,dst_img_size_h, true);
        Bitmap finalResizedBitmap = resizedBitmap;

        runOnUiThread(() -> {
            imageView.setImageBitmap(finalResizedBitmap);
            btn_detect.setEnabled(true);
            btn_detect.setText(getString(R.string.detect));
            progressBar.setVisibility(ProgressBar.INVISIBLE);
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