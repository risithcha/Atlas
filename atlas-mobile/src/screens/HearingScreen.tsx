/**
 * HearingScreen – Live speech-to-text captioning for accessibility.
 *
 * Integrates the `useSpeechRecognition` hook to provide continuous live
 * captions.  Design mirrors the desktop Hearing Assist mode: dark
 * background, large scrolling caption area, green-accented controls.
 *
 * Lifecycle:
 *   • Stops listening when the screen loses focus (tab switch) and does
 *     NOT auto-resume. The user is always in control via the toggle.
 */
import { StatusBar } from 'expo-status-bar';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  Platform,
} from 'react-native';
import { useCallback, useEffect, useRef } from 'react';
import { useIsFocused } from '@react-navigation/native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  cancelAnimation,
  Easing,
} from 'react-native-reanimated';
import Ionicons from '@expo/vector-icons/Ionicons';

import { useSpeechRecognition, useAppState } from '../hooks';
import { triggerHaptic } from '../utils/haptics';
import { COLORS, RADII, SPACING, TYPOGRAPHY, SIZES } from '../theme';
import { AtlasHeader, ActionButton } from '../components';

// ---------------------------------------------------------------------------
// HearingScreen
// ---------------------------------------------------------------------------
export default function HearingScreen() {
  const isFocused = useIsFocused();
  const appState = useAppState();
  const scrollRef = useRef<ScrollView>(null);

  const {
    text,
    isListening,
    error,
    isAvailable,
    startListening,
    stopListening,
    resetTranscript,
  } = useSpeechRecognition({ lang: 'en-US', continuous: true });

  // --- Stop listening when navigating away or app backgrounds ---
  useEffect(() => {
    if ((!isFocused || appState !== 'active') && isListening) {
      stopListening();
    }
  }, [isFocused, appState, isListening, stopListening]);

  // --- Pulsing dot animation (Reanimated) ---
  const pulseScale = useSharedValue(1);
  const pulseOpacity = useSharedValue(1);

  useEffect(() => {
    if (isListening) {
      pulseScale.value = withRepeat(
        withTiming(1.4, { duration: 800, easing: Easing.inOut(Easing.ease) }),
        -1,
        true,
      );
      pulseOpacity.value = withRepeat(
        withTiming(0.4, { duration: 800, easing: Easing.inOut(Easing.ease) }),
        -1,
        true,
      );
    } else {
      cancelAnimation(pulseScale);
      cancelAnimation(pulseOpacity);
      pulseScale.value = withTiming(1, { duration: 200 });
      pulseOpacity.value = withTiming(1, { duration: 200 });
    }
  }, [isListening, pulseScale, pulseOpacity]);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
    opacity: pulseOpacity.value,
  }));

  // --- Auto-scroll captions to bottom ---
  useEffect(() => {
    if (text) {
      // Small delay so layout has time to update
      const t = setTimeout(() => {
        scrollRef.current?.scrollToEnd({ animated: true });
      }, 100);
      return () => clearTimeout(t);
    }
  }, [text]);

  // --- Toggle handler ---
  const handleToggle = useCallback(async () => {
    triggerHaptic('toggle');
    if (isListening) {
      await stopListening();
    } else {
      await startListening();
    }
  }, [isListening, startListening, stopListening]);

  // --- Clear handler with haptic ---
  const handleClear = useCallback(() => {
    triggerHaptic('selection');
    resetTranscript();
  }, [resetTranscript]);

  // --- Render ---
  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Unified header */}
      <AtlasHeader
        subtitle="Hearing Assist"
        accentColor={COLORS.primary}
        rightContent={
          <View style={styles.statusRow}>
            <Animated.View
              style={[
                styles.statusDot,
                isListening ? styles.statusDotActive : styles.statusDotIdle,
                isListening && pulseStyle,
              ]}
            />
            <Text
              style={[
                styles.statusLabel,
                isListening ? styles.statusLabelActive : styles.statusLabelIdle,
              ]}
            >
              {isListening ? 'Listening...' : 'Ready'}
            </Text>
          </View>
        }
      />

      <View style={styles.content}>

        {/* Instruction text */}
        <Text style={styles.instructions}>
          {isListening
            ? 'Speak clearly  live captions will appear below.'
            : 'Tap "Start Listening" to begin live captioning.'}
        </Text>

        {/* Error banner */}
        {error && (
          <View style={styles.errorBanner}>
            <Ionicons
              name="warning-outline"
              size={18}
              color={COLORS.warning}
            />
            <Text style={styles.errorText}>
              {error === 'not-allowed'
                ? 'Microphone permission denied. Please enable it in Settings.'
                : error === 'service-not-allowed'
                  ? 'Speech recognition is not available on this device.'
                  : `Error: ${error}`}
            </Text>
          </View>
        )}

        {/* Caption area */}
        <View style={styles.captionContainer}>
          <Text style={styles.captionHeader}>Live Captions</Text>
          <ScrollView
            ref={scrollRef}
            style={styles.captionScroll}
            contentContainerStyle={styles.captionContent}
            showsVerticalScrollIndicator
          >
            {text ? (
              <Text style={styles.captionText}>{text}</Text>
            ) : (
              <Text style={styles.placeholderText}>
                {isListening
                  ? 'Waiting for speech...'
                  : 'Your transcribed speech will appear here...\n\nTips:\n• Speak clearly and at a normal pace\n• Reduce background noise for best results\n• Each chunk of speech will be transcribed in real time'}
              </Text>
            )}
          </ScrollView>
        </View>

        {/* Controls */}
        <View style={styles.controlsRow}>
          <ActionButton
            label="Clear"
            icon="trash-outline"
            color={COLORS.secondary}
            variant="outlined"
            onPress={handleClear}
            disabled={!text}
          />

          <ActionButton
            label={isListening ? 'Stop Listening' : 'Start Listening'}
            icon={isListening ? 'mic-off' : 'mic'}
            iconSize={28}
            color={isListening ? COLORS.danger : COLORS.primary}
            variant="filled"
            onPress={handleToggle}
            disabled={!isAvailable}
            fullWidth
          />
        </View>
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
  content: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
    paddingBottom: Platform.OS === 'ios' ? SPACING.md : SPACING.lg,
  },

  // Status indicator (in header rightContent)
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: SIZES.statusDot,
    height: SIZES.statusDot,
    borderRadius: SIZES.statusDot / 2,
    marginRight: SPACING.sm,
  },
  statusDotActive: {
    backgroundColor: COLORS.primary,
  },
  statusDotIdle: {
    backgroundColor: COLORS.textMuted,
  },
  statusLabel: {
    fontSize: TYPOGRAPHY.body.fontSize,
    fontWeight: 'bold',
  },
  statusLabelActive: {
    color: COLORS.primary,
  },
  statusLabelIdle: {
    color: COLORS.textMuted,
  },

  // Instructions
  instructions: {
    fontSize: TYPOGRAPHY.caption.fontSize,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginBottom: SPACING.md,
    marginTop: SPACING.sm,
  },

  // Error
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(244, 67, 54, 0.15)',
    borderRadius: RADII.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    marginBottom: SPACING.md,
  },
  errorText: {
    color: COLORS.danger,
    fontSize: TYPOGRAPHY.caption.fontSize,
    marginLeft: SPACING.sm,
    flex: 1,
  },

  // Caption area
  captionContainer: {
    flex: 1,
    borderWidth: 2,
    borderColor: COLORS.primary,
    borderRadius: RADII.lg,
    overflow: 'hidden',
    marginBottom: SPACING.md,
  },
  captionHeader: {
    fontSize: TYPOGRAPHY.body.fontSize,
    fontWeight: 'bold',
    color: COLORS.primary,
    paddingHorizontal: SPACING.md,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.xs,
  },
  captionScroll: {
    flex: 1,
  },
  captionContent: {
    paddingHorizontal: SPACING.md,
    paddingBottom: SPACING.lg,
  },
  captionText: {
    fontSize: TYPOGRAPHY.bodyLarge.fontSize,
    color: COLORS.text,
    lineHeight: 34,
  },
  placeholderText: {
    fontSize: TYPOGRAPHY.body.fontSize,
    color: COLORS.textMuted,
    lineHeight: 26,
    fontStyle: 'italic',
  },

  // Controls
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
    paddingBottom: SPACING.sm,
  },
});
