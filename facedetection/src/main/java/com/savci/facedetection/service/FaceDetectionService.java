package com.savci.facedetection.service;

import com.savci.facedetection.entity.FaceDetectionResult;
import com.savci.facedetection.exception.FaceDetectionException;
import com.savci.facedetection.model.FaceAnnotation;
import com.savci.facedetection.model.MTCNN;
import java.util.concurrent.CompletableFuture;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class FaceDetectionService {
    private static final Logger LOGGER = LoggerFactory.getLogger(FaceDetectionService.class);

    private final MTCNN mtcnn;

    public FaceDetectionService(MTCNN mtcnn) {
        this.mtcnn = mtcnn;
    }

    public CompletableFuture<FaceDetectionResult> detectFaces(byte[] imageData) {
        Assert.notNull(imageData, "Image data must not be null");

        try {
            // Convert byte array to OpenCV Mat
            Mat image = ImageUtils.byteArrayToMat(imageData);
            // Validate image
            if (image.empty()) {
                CompletableFuture.failedFuture(new IllegalArgumentException("Could not read image data"));
            }

            // Detect faces using MTCNN
            FaceAnnotation[] faceAnnotations = mtcnn.detectFace(image);

            // Create result object
            FaceDetectionResult result = new FaceDetectionResult();
            result.setFaces(faceAnnotations);
            result.setTotalFaces(faceAnnotations.length);

            return CompletableFuture.completedFuture(result);

        } catch (Exception e) {
            LOGGER.error("Error detecting faces in image", e);
            return CompletableFuture
                    .failedFuture(new FaceDetectionException("Failed to process image for face detection", e));
        }
    }
}
