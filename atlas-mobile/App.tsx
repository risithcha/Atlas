import { StatusBar } from 'expo-status-bar';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity,
  SafeAreaView,
  Platform,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
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

export default function App() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();

  // Camera permissions are still loading
  if (!permission) {
    return (
      <View style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </View>
    );
  }

  // Camera permissions are not granted yet
  if (!permission.granted) {
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
              <Text style={styles.cameraIcon}>📷</Text>
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

  // Toggle camera facing (front/back)
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  // Main camera view
  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      
      {/* Camera View - Full Screen */}
      <CameraView 
        style={styles.camera} 
        facing={facing}
      >
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

        {/* Bottom Overlay with Controls */}
        <View style={styles.bottomOverlay}>
          {/* Detect Objects Button */}
          <TouchableOpacity 
            style={styles.detectButton}
            activeOpacity={0.8}
            onPress={() => {
              // Placeholder - will connect to backend later
              console.log('Detect Objects pressed');
            }}
          >
            <Text style={styles.detectButtonText}>Detect Objects</Text>
          </TouchableOpacity>

          {/* Status Text */}
          <Text style={styles.statusText}>
            Point camera at objects • Tap to detect
          </Text>
        </View>
      </CameraView>
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
  },
  loadingText: {
    color: COLORS.text,
    fontSize: 18,
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
  cameraIcon: {
    fontSize: 48,
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

  // Camera View
  camera: {
    flex: 1,
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
  detectButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 30,
    paddingHorizontal: 32,
    paddingVertical: 16,
    marginBottom: 16,
    // Shadow for iOS
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    // Shadow for Android
    elevation: 8,
  },
  detectButtonText: {
    color: COLORS.text,
    fontSize: 20,
    fontWeight: 'bold',
  },
  statusText: {
    color: COLORS.textMuted,
    fontSize: 14,
    textAlign: 'center',
  },
});
