package com.savci.facedetection.client;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;

import com.savci.facedetection.service.ImageUtils;

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

import java.awt.image.BufferedImage;

import java.lang.Thread;

public class FaceDetectionWebSocketClient {

    private final String userId;
    private final int chunkSize;
    private final ExecutorService executorService;
    private WebSocketSession session;
    private final CompletableFuture<Void> connectFuture = new CompletableFuture<>();

    // Map to hold chunks for each frame
    private final Map<String, Map<Integer, byte[]>> resultChunks = new ConcurrentHashMap<>();

    public FaceDetectionWebSocketClient(String userId, int chunkSize, int threadPoolSize) {
        this.userId = userId;
        this.chunkSize = chunkSize;
        this.executorService = Executors.newFixedThreadPool(threadPoolSize);
    }

    public CompletableFuture<Void> connect(String baseUrl) {
        String fullUrl = baseUrl + "/" + userId;
        StandardWebSocketClient client = new StandardWebSocketClient();
        client.execute(new BinaryWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(WebSocketSession session) {
                FaceDetectionWebSocketClient.this.session = session;
                connectFuture.complete(null);
            }

            @Override
            public void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
                ByteBuffer buffer = message.getPayload();
                byte[] payload = new byte[buffer.remaining()];
                buffer.get(payload);

                // Extract frameId (UUID), chunkIndex, totalChunks, and chunkData
                String frameId = new String(payload, 0, 36);
                int chunkIndex = ByteBuffer.wrap(payload, 36, 4).getInt();
                int totalChunks = ByteBuffer.wrap(payload, 40, 4).getInt();
                byte[] chunkData = new byte[payload.length - 44];
                System.arraycopy(payload, 44, chunkData, 0, chunkData.length);

                // Store the chunk data
                Map<Integer, byte[]> chunks = resultChunks.computeIfAbsent(frameId, k -> new ConcurrentHashMap<>());
                chunks.put(chunkIndex, chunkData);

                System.out.println("Received chunk " + (chunkIndex + 1) + "/" + totalChunks + " for frame: " + frameId
                        + " chunk size: " + chunks.size());

                // Check if all chunks for this frame have been received
                if (chunks.size() == totalChunks) {
                    processCompleteResult(frameId, chunks, totalChunks);
                }
            }
        }, fullUrl);
        return connectFuture;
    }

    private void processCompleteResult(String frameId, Map<Integer, byte[]> chunks, int totalChunks) {
        ByteBuffer completeResult = ByteBuffer.allocate(
                chunks.values().stream().mapToInt(chunk -> chunk.length).sum());
        for (int i = 0; i < totalChunks; i++) {
            completeResult.put(chunks.get(i));
        }

        byte[] resultData = completeResult.array();
        // Handle the complete result data (e.g., log or process it)
        System.out.println("Processing complete result for frame: " + frameId + ", size: " + resultData.length);
        // Convert byte array to Mat
        Mat img = ImageUtils.byteArrayToMat(resultData);
        opencv_imgcodecs.imwrite("output.png", img);
        resultChunks.remove(frameId); // Clear the chunks for this frame after processing
    }

    public CompletableFuture<Void> sendFrame(byte[] frameData) {
        if (session == null || !session.isOpen()) {
            return CompletableFuture.failedFuture(new IllegalStateException("WebSocket session is not open"));
        }

        String frameId = UUID.randomUUID().toString();
        System.out.println("Data length: " + frameData.length);
        int totalChunks = (int) Math.ceil((double) frameData.length / chunkSize);

        @SuppressWarnings("unchecked")
        CompletableFuture<Void>[] futures = new CompletableFuture[totalChunks];

        for (int i = 0; i < totalChunks; i++) {
            final int chunkIndex = i;
            futures[i] = CompletableFuture.runAsync(() -> {
                synchronized (session) {
                    try {
                        int start = chunkIndex * chunkSize;
                        int end = Math.min(start + chunkSize, frameData.length);
                        byte[] chunk = new byte[end - start];
                        System.arraycopy(frameData, start, chunk, 0, chunk.length);

                        ByteBuffer message = ByteBuffer.allocate(44 + chunk.length);
                        message.put(frameId.getBytes());
                        message.putInt(chunkIndex);
                        message.putInt(totalChunks);
                        message.put(chunk);
                        message.flip();

                        session.sendMessage(new org.springframework.web.socket.BinaryMessage(message));
                    } catch (Exception e) {
                        throw new CompletionException(e);
                    }
                }
            }, executorService);
        }
        System.out.println("Total chunks: " + totalChunks);
        return CompletableFuture.allOf(futures);
    }

    public CompletableFuture<Void> close() {
        if (session != null && session.isOpen()) {
            try {
                session.close();
            } catch (Exception e) {
                return CompletableFuture.failedFuture(e);
            }
        }
        executorService.shutdown();
        return CompletableFuture.completedFuture(null);
    }

    public byte[] convertPNGToByteArray(String imagePath) throws IOException {
        // Read the PNG file
        File imageFile = new File(imagePath);
        BufferedImage bufferedImage = ImageIO.read(imageFile);

        // Create a ByteArrayOutputStream
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        try {
            // Write the image to ByteArrayOutputStream
            ImageIO.write(bufferedImage, "png", baos);

            // Convert ByteArrayOutputStream to byte array
            return baos.toByteArray();
        } finally {
            // Close the stream
            baos.close();
        }
    }

    public static void main(String[] args) {
        String userId = UUID.randomUUID().toString();
        String serverUrl = "ws://localhost:8080/face-detection";
        FaceDetectionWebSocketClient client = new FaceDetectionWebSocketClient(userId, 1024, 4);

        client.connect(serverUrl)
                .thenCompose(v -> {
                    System.out.println("Connected to WebSocket server");
                    String imagePath = "facedetection/src/main/java/com/savci/facedetection/client/image.png";
                    byte[] imageBytes = null;
                    try {
                        imageBytes = client.convertPNGToByteArray(imagePath);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    return client.sendFrame(imageBytes);
                })
                .thenRun(() -> {
                    System.out.println("Frame sent successfully");
                    try {
                        Thread.sleep(15000); // Wait for 10 seconds to receive the response
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                })
                .thenCompose(v -> client.close())
                .whenComplete((v, ex) -> {
                    if (ex != null) {
                        System.err.println("An error occurred: " + ex.getMessage());
                        ex.printStackTrace();
                    } else {
                        System.out.println("Client closed successfully");
                    }
                })
                .join();
    }
}
