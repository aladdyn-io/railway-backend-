/**
 * API Service Layer - Connects to our 4 separate APIs
 * Copy this file to: src/services/api.ts (or similar location)
 */

const MEDIAPIPE_API = 'http://localhost:5001';
const YOLO_API = 'http://localhost:5002';
const OPENCV_API = 'http://localhost:5003';
const FLASK_API = 'http://localhost:5000';

// ==================== MediaPipe API ====================

export interface HandDetection {
  hand_id: number;
  hand_type: string;
  landmarks: Array<{ x: number; y: number; z: number }>;
}

export interface PoseDetection {
  detected: boolean;
  landmarks: Array<{ x: number; y: number; z: number; visibility: number }>;
}

export interface HandSignalDetection {
  signal_detected: boolean;
  detections: Array<{
    hand_id: number;
    hand_raised: boolean;
    hand_above_head: boolean;
  }>;
}

export const mediapipeAPI = {
  detectHands: async (imageFile: File): Promise<{ success: boolean; num_hands: number; detections: HandDetection[] }> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    const response = await fetch(`${MEDIAPIPE_API}/detect/hands`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('MediaPipe hand detection failed');
    return response.json();
  },

  detectPose: async (imageFile: File): Promise<{ success: boolean; detected: boolean; landmarks: any[] }> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    const response = await fetch(`${MEDIAPIPE_API}/detect/pose`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('MediaPipe pose detection failed');
    return response.json();
  },

  detectHandSignal: async (imageFile: File): Promise<HandSignalDetection> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    const response = await fetch(`${MEDIAPIPE_API}/detect/hand_signal`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('MediaPipe hand signal detection failed');
    return response.json();
  },

  health: async (): Promise<{ status: string }> => {
    const response = await fetch(`${MEDIAPIPE_API}/health`);
    if (!response.ok) throw new Error('MediaPipe API not available');
    return response.json();
  },
};

// ==================== YOLO API ====================

export interface ObjectDetection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface PhoneDetection {
  phone_detected: boolean;
  num_phones: number;
  detections: Array<{ confidence: number; bbox: { x1: number; y1: number; x2: number; y2: number } }>;
}

export interface BagDetection {
  bags_detected: boolean;
  num_bags: number;
  detections: Array<{ class_name: string; confidence: number; bbox: { x1: number; y1: number; x2: number; y2: number } }>;
}

export const yoloAPI = {
  detectObjects: async (imageFile: File, confidence: number = 0.25): Promise<{ success: boolean; num_detections: number; detections: ObjectDetection[] }> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('confidence', confidence.toString());
    const response = await fetch(`${YOLO_API}/detect/objects`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('YOLO object detection failed');
    return response.json();
  },

  detectPhone: async (imageFile: File, confidence: number = 0.25): Promise<PhoneDetection> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('confidence', confidence.toString());
    const response = await fetch(`${YOLO_API}/detect/phone`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('YOLO phone detection failed');
    return response.json();
  },

  detectBags: async (imageFile: File, confidence: number = 0.25): Promise<BagDetection> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('confidence', confidence.toString());
    const response = await fetch(`${YOLO_API}/detect/bags`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('YOLO bag detection failed');
    return response.json();
  },

  health: async (): Promise<{ status: string }> => {
    const response = await fetch(`${YOLO_API}/health`);
    if (!response.ok) throw new Error('YOLO API not available');
    return response.json();
  },
};

// ==================== OpenCV API ====================

export interface VideoInfo {
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  duration_seconds: number;
}

export interface FrameExtraction {
  success: boolean;
  frame_number: number;
  frame_base64: string;
}

export const opencvAPI = {
  getVideoInfo: async (videoFile: File): Promise<{ success: boolean; video_info: VideoInfo }> => {
    const formData = new FormData();
    formData.append('video', videoFile);
    const response = await fetch(`${OPENCV_API}/video/info`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('OpenCV video info failed');
    return response.json();
  },

  extractFrame: async (videoFile: File, frameNumber: number = 0): Promise<FrameExtraction> => {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('frame_number', frameNumber.toString());
    const response = await fetch(`${OPENCV_API}/video/extract_frame`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error('OpenCV frame extraction failed');
    return response.json();
  },

  health: async (): Promise<{ status: string }> => {
    const response = await fetch(`${OPENCV_API}/health`);
    if (!response.ok) throw new Error('OpenCV API not available');
    return response.json();
  },
};

// ==================== Combined Detection ====================

export interface DetectionResult {
  timestamp: string;
  handSignal?: HandSignalDetection;
  phone?: PhoneDetection;
  bags?: BagDetection;
  objects?: ObjectDetection[];
  pose?: PoseDetection;
}

export const processFrame = async (imageFile: File): Promise<DetectionResult> => {
  const timestamp = new Date().toISOString();
  try {
    const [handSignal, phone, bags, objects, pose] = await Promise.allSettled([
      mediapipeAPI.detectHandSignal(imageFile),
      yoloAPI.detectPhone(imageFile),
      yoloAPI.detectBags(imageFile),
      yoloAPI.detectObjects(imageFile),
      mediapipeAPI.detectPose(imageFile),
    ]);
    return {
      timestamp,
      handSignal: handSignal.status === 'fulfilled' ? handSignal.value : undefined,
      phone: phone.status === 'fulfilled' ? phone.value : undefined,
      bags: bags.status === 'fulfilled' ? bags.value : undefined,
      objects: objects.status === 'fulfilled' ? objects.value.detections : undefined,
      pose: pose.status === 'fulfilled' ? pose.value : undefined,
    };
  } catch (error) {
    console.error('Error processing frame:', error);
    return { timestamp };
  }
};

export const checkAPIsHealth = async (): Promise<{ mediapipe: boolean; yolo: boolean; opencv: boolean }> => {
  const [mediapipe, yolo, opencv] = await Promise.allSettled([
    mediapipeAPI.health(),
    yoloAPI.health(),
    opencvAPI.health(),
  ]);
  return {
    mediapipe: mediapipe.status === 'fulfilled',
    yolo: yolo.status === 'fulfilled',
    opencv: opencv.status === 'fulfilled',
  };
};



