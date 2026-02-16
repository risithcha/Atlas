import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  Platform,
  ActivityIndicator,
  Dimensions,
  type ViewStyle,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useState, useCallback, useEffect, useRef } from 'react';
import { Worklets } from 'react-native-worklets-core';
import Ionicons from '@expo/vector-icons/Ionicons';
import {
  decodePredictions,
  filterByMinArea,
  mapBoxToScreen,
  type Detection,
  type TFLiteOutputs,
  type FrameInfo,
} from './src/utils/tensor_decoder';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const COLORS = {
  background: '#1a1a1a',
  primary: '#4CAF50',       // Atlas green
  primaryDark: '#2E7D32',
  secondary: '#2196F3',     // Atlas blue
  text: '#ffffff',
  textMuted: '#888888',
  overlay: 'rgba(0, 0, 0, 0.6)',
};

const MODEL_INPUT_SIZE = 300;      // Our quantized model expects 300x300
const CONFIDENCE_THRESHOLD = 0.45; // Min score to show a detection
const MAX_DETECTIONS = 10;         // Cap results per frame
const INFERENCE_FPS = 5;           // How many times/sec we run the model
const MIN_BOX_AREA = 0.005;        // Filter out tiny noise detections

// Palette for bounding-box colours (one per class-id, wraps around)
const BOX_COLORS = [
  '#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
  '#00BCD4', '#FFEB3B', '#FF5722', '#795548', '#607D8B',
];

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// ---------------------------------------------------------------------------
// Bounding-box overlay component
// ---------------------------------------------------------------------------
interface DetectionOverlayProps {
  detections: Detection[];
  frameInfo: FrameInfo | null;
}

function DetectionOverlay({ detections, frameInfo }: DetectionOverlayProps) {
  if (detections.length === 0 || frameInfo == null) return null;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {detections.map((det, idx) => {
        const color = BOX_COLORS[det.classId % BOX_COLORS.length];
        const pct = Math.round(det.score * 100);

        // Map model coords -> screen pixels, accounting for center-crop,
        // buffer rotation, and camera preview "cover" mode.
        const screenBox = mapBoxToScreen(
          det.box,
          frameInfo,
          SCREEN_W,
          SCREEN_H,
          MODEL_INPUT_SIZE,
        );

        const boxStyle: ViewStyle = {
          position: 'absolute',
          left: screenBox.x,
          top: screenBox.y,
          width: screenBox.width,
          height: screenBox.height,
          borderWidth: 2,
          borderColor: color,
          borderRadius: 4,
        };

        // Place the label inside the box at the top if the box is too
        // close to the top of the screen, otherwise above the box.
        const labelAbove = screenBox.y > 22;

        return (
          <View key={`${det.label}-${idx}`} style={boxStyle}>
            <View
              style={[
                styles.labelBadge,
                { backgroundColor: color },
                labelAbove
                  ? { top: -18, left: -2 }
                  : { top: 2, left: 2 },
              ]}
            >
              <Text style={styles.labelText} numberOfLines={1}>
                {det.label} {pct}%
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
export default function App() {
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice(facing);

  // --- Model loading via hook (manages state internally) ---
  const tfModel = useTensorflowModel(
    require('./assets/models/atlas_mobilenet_quant.tflite'),
  );
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;

  // --- Resize plugin (GPU-accelerated frame -> 300×300 RGB uint8) ---
  const { resize } = useResizePlugin();

  // --- Detection state (updated from worklet thread -> JS thread) ---
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameInfo, setFrameInfo] = useState<FrameInfo | null>(null);
  const [fps, setFps] = useState(0);
  const lastInferenceRef = useRef(Date.now());

  // Bridge: worklet -> JS thread.  Receives raw output arrays + frame
  // dimensions, decodes on the JS thread (cheap), then updates React state.
  const onDetectionResults = Worklets.createRunOnJS(
    (
      rawBoxes: number[],
      rawClasses: number[],
      rawScores: number[],
      rawCount: number,
      fWidth: number,
      fHeight: number,
      fOrientation: string,
    ) => {
      const now = Date.now();
      const delta = now - lastInferenceRef.current;
      lastInferenceRef.current = now;
      if (delta > 0) setFps(Math.round(1000 / delta));

      // Store frame info for coordinate mapping in the overlay
      setFrameInfo({
        frameWidth: fWidth,
        frameHeight: fHeight,
        frameOrientation: fOrientation,
      });

      const outputs: TFLiteOutputs = {
        boxes: rawBoxes,
        classes: rawClasses,
        scores: rawScores,
        count: rawCount,
      };

      let results = decodePredictions(outputs, {
        threshold: CONFIDENCE_THRESHOLD,
        maxDetections: MAX_DETECTIONS,
      });

      // Filter out tiny noise detections
      results = filterByMinArea(results, MIN_BOX_AREA);

      setDetections(results);
    },
  );

  // --- Frame Processor (runs on worklet thread) ---
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (model == null) return;

      // Throttle heavy inference so the camera stays at full preview FPS
      runAtTargetFps(INFERENCE_FPS, () => {
        'worklet';

        // Compute rotation so the model always sees upright (portrait) content.
        // The raw camera buffer is typically landscape; without rotation the model
        // receives sideways images, destroying both classification and localisation.
        const orientation = frame.orientation;
        const rotation =
          orientation === 'landscape-left'
            ? '90deg'
            : orientation === 'landscape-right'
              ? '270deg'
              : orientation === 'portrait-upside-down'
                ? '180deg'
                : '0deg';

        // Resize the camera frame -> 300x300 RGB uint8
        // Pipeline: center-crop (1:1 on raw buffer) -> scale -> rotate
        const resized = resize(frame, {
          scale: {
            width: MODEL_INPUT_SIZE,
            height: MODEL_INPUT_SIZE,
          },
          rotation,
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        // Run synchronous inference on the worklet thread (off UI thread)
        const outputs = model.runSync([resized]);

        // Extract raw arrays (small – typically 10 detections)
        // Convert from TypedArrays to plain arrays so they cross the
        // worklet->JS bridge without issues.
        const rawBoxes = Array.from(outputs[0] as unknown as number[]);
        const rawClasses = Array.from(outputs[1] as unknown as number[]);
        const rawScores = Array.from(outputs[2] as unknown as number[]);
        const rawCount = outputs[3]
          ? (outputs[3] as unknown as number[])[0]
          : 0;

        // Send to JS thread for decoding + state update
        // Include frame dimensions & orientation for coordinate mapping
        onDetectionResults(
          rawBoxes,
          rawClasses,
          rawScores,
          rawCount,
          frame.width,
          frame.height,
          frame.orientation,
        );
      });
    },
    [model, resize, onDetectionResults],
  );

  // Toggle camera facing (front/back)
  const toggleCameraFacing = useCallback(() => {
    setFacing((c) => (c === 'back' ? 'front' : 'back'));
  }, []);

  // --- Render: loading / error / permission / camera ---

  if (tfModel.state === 'loading') {
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

  if (tfModel.state === 'error') {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Ionicons name="alert-circle" size={64} color="#FF5252" />
          <Text style={styles.errorTitle}>Model Error</Text>
          <Text style={styles.loadingText}>
            {tfModel.error?.message ?? 'Unknown error'}
          </Text>
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

  // ---- Main camera view with live detection overlay ----
  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Camera – full screen, feeds frames into our processor */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        resizeMode="cover"
      />

      {/* Bounding-box overlay */}
      <DetectionOverlay detections={detections} frameInfo={frameInfo} />

      {/* Top bar */}
      <SafeAreaView style={styles.topOverlay}>
        <View style={styles.topBar}>
          <Text style={styles.logoTextSmall}>ATLAS</Text>

          {/* Detection count badge */}
          {detections.length > 0 && (
            <View style={styles.countBadge}>
              <Text style={styles.countText}>{detections.length}</Text>
            </View>
          )}

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
        <View style={styles.statusContainer}>
          <View
            style={[
              styles.statusDot,
              {
                backgroundColor:
                  model != null ? COLORS.primary : COLORS.textMuted,
              },
            ]}
          />
          <Text style={styles.statusText}>
            {model != null
              ? `Detecting \u2022 ${fps} inf/s`
              : 'Loading model...'}
          </Text>
        </View>

        {/* Model Info */}
        <Text style={styles.fpsText}>
          Model: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} UINT8 •{' '}
          {detections.length} object{detections.length !== 1 ? 's' : ''}
        </Text>

        {/* Mini detection list */}
        {detections.length > 0 && (
          <View style={styles.detectionList}>
            {detections.slice(0, 3).map((d, i) => (
              <Text key={i} style={styles.detectionItem}>
                {d.label} ({Math.round(d.score * 100)}%)
              </Text>
            ))}
          </View>
        )}
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
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

  // Detection count badge (top bar)
  countBadge: {
    backgroundColor: COLORS.primary,
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 2,
    minWidth: 24,
    alignItems: 'center',
  },
  countText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: 'bold',
  },

  // Bottom overlay
  bottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: COLORS.overlay,
    paddingBottom: Platform.OS === 'ios' ? 40 : 30,
    paddingTop: 16,
    alignItems: 'center',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
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

  // Mini detection list at the bottom
  detectionList: {
    marginTop: 8,
    alignItems: 'center',
  },
  detectionItem: {
    color: COLORS.text,
    fontSize: 13,
    opacity: 0.85,
  },

  // Bounding-box label badge
  labelBadge: {
    position: 'absolute',
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 3,
  },
  labelText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
});
