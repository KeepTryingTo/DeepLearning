package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class ImageDetection extends AppCompatActivity implements View.OnClickListener {

    private Button cls_btn;
    private Button det_btn;
    private String classes;
    private ActivityResultLauncher<Intent> register;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_image_detection);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        cls_btn = findViewById(R.id.cls_btn);
        det_btn = findViewById(R.id.det_btn);

        cls_btn.setOnClickListener(this);
        det_btn.setOnClickListener(this);

        register = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult o) {

            }
        });
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.cls_btn:
                //跳转到忘记密码的界面
                Intent intent_cls = new Intent(this, imageClassification.class);
                register.launch(intent_cls);

                break;
            case R.id.det_btn:
                //跳转到忘记密码的界面
                Intent intent_det = new Intent(this, objectDetection.class);
                register.launch(intent_det);
                break;
            default:
                break;
        }
    }
}