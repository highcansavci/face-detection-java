package com.savci.facedetection.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.handler.PerConnectionWebSocketHandler;
import com.savci.facedetection.handler.FaceDetectionWebSocketHandler;
import com.savci.facedetection.model.MTCNN;
import com.savci.facedetection.service.FaceDetectionService;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(faceDetectionWebSocketHandler(), "/face-detection/{userId}")
                .setAllowedOrigins("*");
    }

    @Bean
    public WebSocketHandler faceDetectionWebSocketHandler() {
        return new PerConnectionWebSocketHandler(FaceDetectionWebSocketHandler.class);
    }

    @Bean
    public FaceDetectionService faceDetectionService() {
        return new FaceDetectionService(mtcnn());
    }

    @Bean
    public MTCNN mtcnn() {
        return new MTCNN();
    }
}