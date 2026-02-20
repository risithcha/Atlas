/**
 * VisionScreen – Camera-based object detection with real-time bounding boxes.
 *
 * Extracted from the original monolithic App.tsx.  Now lives as a dedicated
 * screen inside the React Navigation tab navigator.
 *
 * Key lifecycle behaviour:
 *   • Camera `isActive` is tied to `useIsFocused()` so the camera pauses
 *     when the user switches to another tab (saves battery and hides the iOS
 *     green privacy indicator).
 */
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Platform,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useState, useCallback, useRef } from 'react';
import { Worklets } from 'react-native-worklets-core';
import { useIsFocused } from '@react-navigation/native';
import Ionicons from '@expo/vector-icons/Ionicons';

import { DetectionOverlay } from '../components/DetectionOverlay';
import {
  decodePredictions,
  filterByMinArea,
  type Detection,
  type TFLiteOutputs,
  type FrameInfo,
} from '../utils/tensor_decoder';
import {
  COLORS,
  MODEL_INPUT_SIZE,
  CONFIDENCE_THRESHOLD,
  MAX_DETECTIONS,
  INFERENCE_FPS,
  MIN_BOX_AREA,
} from '../theme';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// ---------------------------------------------------------------------------
// VisionScreen
// ---------------------------------------------------------------------------
export default function VisionScreen() {
  const isFocused = useIsFocused();

  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice(facing);

  // --- Model loading ---
  const tfModel = useTensorflowModel(
    require('../../assets/models/atlas_mobilenet_quant.tflite'),
  );
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;

  // --- Resize plugin ---
  const { resize } = useResizePlugin();

  // --- Detection state ---
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameInfo, setFrameInfo] = useState<FrameInfo | null>(null);
  const [fps, setFps] = useState(0);
  const lastInferenceRef = useRef(Date.now());

  // Bridge: worklet → JS thread
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
      results = filterByMinArea(results, MIN_BOX_AREA);
      setDetections(results);
    },
  );

  // --- Frame Processor ---
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (model == null) return;

      runAtTargetFps(INFERENCE_FPS, () => {
        'worklet';

        const orientation = frame.orientation;
        const rotation =
          orientation === 'landscape-left'
            ? '90deg'
            : orientation === 'landscape-right'
              ? '270deg'
              : orientation === 'portrait-upside-down'
                ? '180deg'
                : '0deg';

        const resized = resize(frame, {
          scale: {
            width: MODEL_INPUT_SIZE,
            height: MODEL_INPUT_SIZE,
          },
          rotation,
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        const outputs = model.runSync([resized]);

        const rawBoxes = Array.from(outputs[0] as unknown as number[]);
        const rawClasses = Array.from(outputs[1] as unknown as number[]);
        const rawScores = Array.from(outputs[2] as unknown as number[]);
        const rawCount = outputs[3]
          ? (outputs[3] as unknown as number[])[0]
          : 0;

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

  // Toggle camera facing
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

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <SafeAreaView style={styles.permissionContainer}>
          <View style={styles.header}>
            <Text style={styles.logoText}>ATLAS</Text>
            <Text style={styles.tagline}>Vision Assist</Text>
          </View>
          <View style={styles.permissionContent}>
            <View style={styles.cameraIconContainer}>
              <Ionicons name="camera" size={48} color={COLORS.primary} />
            </View>
            <Text style={styles.permissionTitle}>Camera Access Required</Text>
            <Text style={styles.permissionMessage}>
              Atlas needs access to your camera to detect and describe objects in
              your environment.
            </Text>
            <TouchableOpacity
              style={styles.permissionButton}
              onPress={requestPermission}
              activeOpacity={0.8}
            >
              <Text style={styles.permissionButtonText}>
                Grant Camera Permission
              </Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </View>
    );
  }

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

  // ---- Main camera view ----
  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Camera – isActive driven by navigation focus */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        resizeMode="cover"
      />

      {/* Bounding-box overlay */}
      <DetectionOverlay detections={detections} frameInfo={frameInfo} />

      {/* Top bar */}
      <SafeAreaView style={styles.topOverlay} edges={['top']}>
        <View style={styles.topBar}>
          <Text style={styles.logoTextSmall}>ATLAS</Text>
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
            <Ionicons
              name="camera-reverse-outline"
              size={28}
              color="#ffffff"
            />
          </TouchableOpacity>
        </View>
      </SafeAreaView>

      {/* Bottom overlay */}
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

        <Text style={styles.fpsText}>
          Model: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} UINT8 •{' '}
          {detections.length} object{detections.length !== 1 ? 's' : ''}
        </Text>

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

  // Loading / Error
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

  // Permission
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

  // Top overlay
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
  detectionList: {
    marginTop: 8,
    alignItems: 'center',
  },
  detectionItem: {
    color: COLORS.text,
    fontSize: 13,
    opacity: 0.85,
  },
});
