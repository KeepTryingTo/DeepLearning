package com.example.translate_transformer;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class LoginForgetActivity extends AppCompatActivity implements View.OnClickListener {
    private EditText et_password_first;
    private EditText et_password_second;

    private String mPhone;
    private String mVerifyCode;
    private EditText et_verifycode;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_login_forget);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        et_password_first = findViewById(R.id.et_password_first);
        et_password_second = findViewById(R.id.et_password_second);
        et_verifycode = findViewById(R.id.et_verifycode);

        mPhone = getIntent().getStringExtra("phone");

        findViewById(R.id.btn_verifycode).setOnClickListener(this);
        findViewById(R.id.btn_confirm).setOnClickListener(this);

        //给输入号码和输入密码文本编辑框添加监听事件
        et_password_first.addTextChangedListener(new HideTextWatcher(et_password_first,6));
        et_password_second.addTextChangedListener(new HideTextWatcher(et_password_second,6));
    }

    @Override
    public void onClick(View view) {
//        switch (view.getId()){
//            //点击了获取验证码
//            case R.id.btn_verifycode:
//                //生成随机的6为验证码
//                mVerifyCode = String.format("%06d",new Random().nextInt(999999));
//                //弹出一个窗口，提示用户记住验证码数字
//                AlertDialog.Builder builder = new AlertDialog.Builder(this);
//                builder.setTitle("请记住验证码");
//                builder.setMessage("手机号" + mPhone + ",本次验证码是" + mVerifyCode + ",请输入验证码");
//                builder.setPositiveButton("好的",null);
//                AlertDialog dialog = builder.create();
//                dialog.show();
//                break;
//                //点击确定按钮
//            case R.id.btn_confirm:
//                String password_first = et_password_first.getText().toString();
//                String password_second = et_password_second.getText().toString();
//                if(password_first.length() < 6){
//                    Toast.makeText(this,"密码不足6位",Toast.LENGTH_SHORT).show();
//                    return;
//                }
//                if(password_second.length() < 6){
//                    Toast.makeText(this,"密码不足6位",Toast.LENGTH_SHORT).show();
//                    return;
//                }
//                if(!password_second.equals(password_first)){
//                    Toast.makeText(this,"密码与第一次不匹配",Toast.LENGTH_SHORT).show();
//                    return;
//                }
//                if(!mVerifyCode.equals(et_verifycode.getText().toString())){
//                    Toast.makeText(this,"输入验证码错误",Toast.LENGTH_SHORT).show();
//                    return;
//                }
//                Toast.makeText(this,"修改密码成功",Toast.LENGTH_SHORT).show();
//
//                //密码修改成功之后就将修改之后的密码传回上一个页面
//                Intent intent = new Intent();
//                intent.putExtra("new_password",password_first);
//                setResult(Activity.RESULT_OK,intent);
//                finish();
//                break;
//        }
    }

    //定义一个编辑框监听器，在输入文本达到指定长度时自动隐藏输入法
    public class HideTextWatcher implements TextWatcher {
        private EditText mView;
        private int mMaxLength;
        public HideTextWatcher(EditText et, int maxLength) {
            this.mView = et;
            this.mMaxLength = maxLength;
        }

        @Override
        public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

        }

        @Override
        public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

        }

        @Override
        public void afterTextChanged(Editable s) {
            if(s.toString().length() == mMaxLength){
                //输入法给隐藏
                ViewUtil.hideOneInputMethod(LoginForgetActivity.this,mView);
            }
        }
    }
}