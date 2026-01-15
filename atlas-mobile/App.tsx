import { StatusBar } from 'expo-status-bar';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity,
  SafeAreaView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { 
  Camera, 
  useCameraDevice, 
  useCameraPermission,
} from 'react-native-vision-camera';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useState, useCallback, useEffect, useRef } from 'react';
import Ionicons from '@expo/vector-icons/Ionicons';

// Atlas color scheme
const COLORS = {
  background: '#1a1a1a',
  primary: '#4CAF50',       // Atlas green
  primaryDark: '#2E7D32',
  secondary: '#2196F3',     // Atlas blue
  text: '#ffffff',
  textMuted: '#888888',
  overlay: 'rgba(0, 0, 0, 0.6)',
};

// Model configuration
const MODEL_INPUT_SIZE = 300;
const CONFIDENCE_THRESHOLD = 0.5;

// COCO labels (subset for common objects)
const COCO_LABELS: { [key: number]: string } = {
  0: 'background',
  1: 'person',
  2: 'bicycle',
  3: 'car',
  4: 'motorcycle',
  5: 'airplane',
  6: 'bus',
  7: 'train',
  8: 'truck',
  9: 'boat',
  10: 'traffic light',
  11: 'fire hydrant',
  13: 'stop sign',
  14: 'parking meter',
  15: 'bench',
  16: 'bird',
  17: 'cat',
  18: 'dog',
  19: 'horse',
  20: 'sheep',
  21: 'cow',
  22: 'elephant',
  23: 'bear',
  24: 'zebra',
  25: 'giraffe',
  27: 'backpack',
  28: 'umbrella',
  31: 'handbag',
  32: 'tie',
  33: 'suitcase',
  34: 'frisbee',
  35: 'skis',
  36: 'snowboard',
  37: 'sports ball',
  38: 'kite',
  39: 'baseball bat',
  40: 'baseball glove',
  41: 'skateboard',
  42: 'surfboard',
  43: 'tennis racket',
  44: 'bottle',
  46: 'wine glass',
  47: 'cup',
  48: 'fork',
  49: 'knife',
  50: 'spoon',
  51: 'bowl',
  52: 'banana',
  53: 'apple',
  54: 'sandwich',
  55: 'orange',
  56: 'broccoli',
  57: 'carrot',
  58: 'hot dog',
  59: 'pizza',
  60: 'donut',
  61: 'cake',
  62: 'chair',
  63: 'couch',
  64: 'potted plant',
  65: 'bed',
  67: 'dining table',
  70: 'toilet',
  72: 'tv',
  73: 'laptop',
  74: 'mouse',
  75: 'remote',
  76: 'keyboard',
  77: 'cell phone',
  78: 'microwave',
  79: 'oven',
  80: 'toaster',
  81: 'sink',
  82: 'refrigerator',
  84: 'book',
  85: 'clock',
  86: 'vase',
  87: 'scissors',
  88: 'teddy bear',
  89: 'hair drier',
  90: 'toothbrush',
};

export default function App() {
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice(facing);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const modelRef = useRef<TensorflowModel | null>(null);
  const cameraRef = useRef<Camera>(null);

  // Load the TensorFlow Lite model on mount
  useEffect(() => {
    async function loadModel() {
      try {
        console.log('Loading Atlas TFLite Model...');
        const model = await loadTensorflowModel(
          require('./assets/models/atlas_mobilenet_quant.tflite')
        );
        modelRef.current = model;
        setModelLoaded(true);
        console.log('Atlas TFLite Model loaded successfully!');
        console.log('Model inputs:', model.inputs);
        console.log('Model outputs:', model.outputs);
      } catch (error) {
        console.error('Failed to load model:', error);
        setModelError(error instanceof Error ? error.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    }
    loadModel();
  }, []);

  // Toggle camera facing (front/back)
  const toggleCameraFacing = useCallback(() => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }, []);

  // Camera permissions are still loading or model is loading
  if (isLoading) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loadingText}>Loading Atlas AI...</Text>
        </View>
      </View>
    );
  }

  // Model error
  if (modelError) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Ionicons name="alert-circle" size={64} color="#FF5252" />
          <Text style={styles.errorTitle}>Model Error</Text>
          <Text style={styles.loadingText}>{modelError}</Text>
        </View>
      </View>
    );
  }

  // Camera permissions are not granted yet
  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <SafeAreaView style={styles.permissionContainer}>
          {/* Atlas Logo/Header */}
          <View style={styles.header}>
            <Text style={styles.logoText}>ATLAS</Text>
            <Text style={styles.tagline}>Vision Assist</Text>
          </View>

          {/* Permission Request */}
          <View style={styles.permissionContent}>
            <View style={styles.cameraIconContainer}>
              <Ionicons name="camera" size={48} color={COLORS.primary} />
            </View>
            <Text style={styles.permissionTitle}>Camera Access Required</Text>
            <Text style={styles.permissionMessage}>
              Atlas needs access to your camera to detect and describe objects in your environment.
            </Text>
            <TouchableOpacity 
              style={styles.permissionButton} 
              onPress={requestPermission}
              activeOpacity={0.8}
            >
              <Text style={styles.permissionButtonText}>Grant Camera Permission</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  // No camera device found
  if (device == null) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Ionicons name="camera" size={64} color={COLORS.textMuted} />
          <Text style={styles.loadingText}>No camera device found</Text>
        </View>
      </View>
    );
  }

  // Main camera view
  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      
      {/* Camera View - Full Screen */}
      <Camera 
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />

      {/* Top Overlay */}
      <SafeAreaView style={styles.topOverlay}>
        <View style={styles.topBar}>
          <Text style={styles.logoTextSmall}>ATLAS</Text>
          <TouchableOpacity 
            style={styles.flipButton} 
            onPress={toggleCameraFacing}
            activeOpacity={0.7}
          >
            <Ionicons name="camera-reverse-outline" size={28} color="#ffffff" />
          </TouchableOpacity>
        </View>
      </SafeAreaView>

      {/* Bottom Overlay with Status */}
      <View style={styles.bottomOverlay}>
        {/* Status Indicator */}
        <View style={styles.statusContainer}>
          <View style={[
            styles.statusDot,
            { backgroundColor: modelLoaded ? COLORS.primary : COLORS.textMuted }
          ]} />
          <Text style={styles.statusText}>
            {modelLoaded ? 'AI Ready • Camera active' : 'Loading model...'}
          </Text>
        </View>

        {/* Model Info */}
        <Text style={styles.fpsText}>
          Model: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} UINT8
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  
  // Loading State
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  loadingText: {
    color: COLORS.textMuted,
    fontSize: 16,
    marginTop: 16,
    textAlign: 'center',
  },
  errorTitle: {
    color: COLORS.text,
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 16,
  },

  // Permission Screen
  permissionContainer: {
    flex: 1,
    justifyContent: 'space-between',
    paddingHorizontal: 30,
    paddingVertical: 50,
  },
  header: {
    alignItems: 'center',
    marginTop: 40,
  },
  logoText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: COLORS.primary,
    letterSpacing: 8,
  },
  tagline: {
    fontSize: 16,
    color: COLORS.textMuted,
    marginTop: 8,
    letterSpacing: 2,
  },
  permissionContent: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
  },
  cameraIconContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: 16,
  },
  permissionMessage: {
    fontSize: 16,
    color: COLORS.textMuted,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 40,
    paddingHorizontal: 20,
  },
  permissionButton: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    minWidth: 280,
  },
  permissionButtonText: {
    color: COLORS.text,
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  
  // Top Overlay
  topOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.overlay,
    paddingTop: Platform.OS === 'android' ? 30 : 0,
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  logoTextSmall: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.primary,
    letterSpacing: 4,
  },
  flipButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Bottom Overlay
  bottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.overlay,
    paddingBottom: Platform.OS === 'ios' ? 40 : 30,
    paddingTop: 20,
    alignItems: 'center',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusText: {
    color: COLORS.text,
    fontSize: 16,
    textAlign: 'center',
  },
  fpsText: {
    color: COLORS.textMuted,
    fontSize: 12,
    textAlign: 'center',
    marginTop: 4,
  },
});
