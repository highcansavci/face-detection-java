package com.savci.facedetection.service;

import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import java.nio.ByteBuffer;

public class ImageUtils {

    private ImageUtils() {
        throw new UnsupportedOperationException("Utility class.");
    }

    public static Mat byteArrayToMat(byte[] imageData) {
        Mat mat = new Mat(1, imageData.length, opencv_core.CV_8UC1);
        mat.data().put(imageData);
        return opencv_imgcodecs.imdecode(mat, opencv_imgcodecs.IMREAD_COLOR);
    }

    public static byte[] convertMatToByteArray(Mat mat) {
        ByteBuffer buffer = ByteBuffer.allocate((int) (mat.total()));
        opencv_imgcodecs.imencode(".png", mat, buffer);
        return buffer.array(); // Return the byte array
    }

}
