// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;


public class ResultView extends View {
    private Paint mPaintText;//这里只需要绘制文本
    private String mResults;

    public ResultView(Context context) {
        super(context);
    }

    public ResultView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintText = new Paint();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        //在图像的下面绘制预测的结果
        if (mResults == null) return;
        mPaintText.setColor(Color.CYAN);
        mPaintText.setTextSize(70);
        mPaintText.setAntiAlias(true);

        String[] lines = mResults.split("\n");
        int yOffset = 1800;
        for (String line : lines){
            canvas.drawText(line,50,yOffset,mPaintText);
            yOffset += 115;
        }
    }

    public void setResults(String results) {
        mResults = results;
    }
}
