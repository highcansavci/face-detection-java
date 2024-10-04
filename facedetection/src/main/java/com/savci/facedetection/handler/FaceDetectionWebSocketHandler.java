package com.savci.facedetection.handler;

import com.savci.facedetection.model.FaceAnnotation;
import com.savci.facedetection.model.MTCNNUtils;
import com.savci.facedetection.service.FaceDetectionService;
import com.savci.facedetection.service.ImageUtils;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;
import org.springframework.web.util.UriComponents;
import org.springframework.web.util.UriComponentsBuilder;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.bytedeco.opencv.global.opencv_imgproc.circle;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public class FaceDetectionWebSocketHandler extends BinaryWebSocketHandler {

    private final Map<String, WebSocketSession> userSessions = new ConcurrentHashMap<>();
    private final Map<String, Map<Integer, byte[]>> frameChunks = new ConcurrentHashMap<>();
    private static final Logger LOGGER = LoggerFactory.getLogger(FaceDetectionWebSocketHandler.class);
    private static final ExecutorService SERVICE = Executors
            .newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private final FaceDetectionService faceDetectionService;

    public FaceDetectionWebSocketHandler(FaceDetectionService faceDetectionService) {
        this.faceDetectionService = faceDetectionService;
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        String userId = extractUserId(session);
        userSessions.put(userId, session);
        LOGGER.info("WebSocket connection established for user: {}", userId);
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) throws Exception {
        String userId = extractUserId(session);
        ByteBuffer buffer = message.getPayload();
        byte[] payload = new byte[buffer.remaining()];
        buffer.get(payload);

        String frameId = new String(payload, 0, 36); // Assuming UUID as frameId
        int chunkIndex = ByteBuffer.wrap(payload, 36, 4).getInt();
        int totalChunks = ByteBuffer.wrap(payload, 40, 4).getInt();
        byte[] chunkData = new byte[payload.length - 44];
        System.arraycopy(payload, 44, chunkData, 0, chunkData.length);

        String frameKey = userId + ";" + frameId;
        Map<Integer, byte[]> chunks = frameChunks.computeIfAbsent(frameKey, k -> new ConcurrentHashMap<>());
        chunks.put(chunkIndex, chunkData);

        LOGGER.info("Received chunk {}/{} for frame {} from user {}", chunkIndex + 1, totalChunks, frameId, userId);

        // Check if all chunks have been received
        if (chunks.size() == totalChunks) {
            processCompleteFrame(userId, frameId, chunks, totalChunks);
        }
    }

    private void processCompleteFrame(String userId, String frameId, Map<Integer, byte[]> chunks, int totalChunks) {
        ByteBuffer completeFrame = ByteBuffer.allocate(chunks.values().stream().mapToInt(chunk -> chunk.length).sum());
        for (int i = 0; i < totalChunks; i++) {
            completeFrame.put(chunks.get(i));
        }

        byte[] frameData = completeFrame.array();
        frameChunks.remove(userId + ";" + frameId);

        LOGGER.info("Processing complete frame: {} for user: {}", frameId, userId);

        faceDetectionService.detectFaces(frameData)
                .thenAccept(result -> {
                    try {
                        Mat image = ImageUtils.byteArrayToMat(frameData);
                        List<Mat> alignedFace = new ArrayList<>();
                        FaceAnnotation[] faceAnnotations = result.getFaces();
                        if (faceAnnotations.length != 0) {
                            for (FaceAnnotation faceAnnotation : faceAnnotations) {

                                alignedFace.add(MTCNNUtils.faceAligner(image, faceAnnotation));

                                FaceAnnotation.BoundingBox bbox = faceAnnotation.getBoundingBox();
                                Point x1y1 = new Point(bbox.getX(), bbox.getY());
                                Point x2y2 = new Point(bbox.getX() + bbox.getW(), bbox.getY() + bbox.getH());
                                rectangle(image, x1y1, x2y2, new Scalar(0, 255, 0, 0));
                                for (FaceAnnotation.Landmark lm : faceAnnotation.getLandmarks()) {
                                    Point keyPoint = new Point(lm.getPosition().getX(), lm.getPosition().getY());
                                    circle(image, keyPoint, 2, new Scalar(0, 255, 0, 0), -1, 0, 0);
                                }
                            }
                        }
                        byte[] imageArray = ImageUtils.convertMatToByteArray(image);
                        // Send the result back to the client via WebSocket
                        sendDetectionResult(userId, imageArray);
                        LOGGER.info("Face detection result sent for frame: {} to user: {}", frameId, userId);
                    } catch (Exception e) {
                        LOGGER.error("Error sending face detection result for user: {}", userId, e);
                    }
                })
                .exceptionally(ex -> {
                    LOGGER.error("Error processing frame {} for user {}", frameId, userId, ex);
                    return null;
                });
    }

    public CompletableFuture<Void> sendDetectionResult(String userId, byte[] result) throws Exception {
        WebSocketSession session = userSessions.get(userId);
        if (session == null || !session.isOpen()) {
            return CompletableFuture.failedFuture(new IllegalStateException("WebSocket session is not open"));
        }

        String frameId = UUID.randomUUID().toString();
        int chunkSize = 1024;
        int totalChunks = (int) Math.ceil((double) result.length / chunkSize);

        List<CompletableFuture<Void>> futures = new ArrayList<>();

        for (int i = 0; i < totalChunks; i++) {
            final int chunkIndex = i;
            futures.add(CompletableFuture.runAsync(() -> {
                synchronized (session) { // Synchronize to prevent overlapping writes
                    try {
                        int start = chunkIndex * chunkSize;
                        int end = Math.min(start + chunkSize, result.length);
                        byte[] chunk = new byte[end - start];
                        System.arraycopy(result, start, chunk, 0, chunk.length);

                        // Allocate ByteBuffer with correct size
                        ByteBuffer message = ByteBuffer.allocate(36 + 4 + 4 + chunk.length);
                        message.put(frameId.getBytes()); // Put the frameId (UUID)
                        message.putInt(chunkIndex); // Put the chunkIndex
                        message.putInt(totalChunks); // Put totalChunks
                        message.put(chunk); // Put the actual chunk data
                        message.flip();

                        // Send binary message
                        session.sendMessage(new BinaryMessage(message));
                    } catch (Exception e) {
                        throw new CompletionException(e);
                    }
                }
            }, SERVICE));
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenRun(() -> System.out.println("All result chunks sent successfully."));
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        String userId = extractUserId(session);
        userSessions.remove(userId);
        LOGGER.info("WebSocket connection closed for user: {}. Status: {}", userId, status);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        String userId = extractUserId(session);
        LOGGER.error("Transport error occurred for user: {}. Error: {}", userId, exception.getMessage());
    }

    private String extractUserId(WebSocketSession session) {
        String path = session.getUri().getPath();
        UriComponents uriComponents = UriComponentsBuilder.fromUriString(path).build();
        return uriComponents.getPathSegments().get(1); // Assumes path is /face-detection/{userId}
    }
}