/**
 * Atlas – Hearing Assist Mode (Voice Processing Layer)
 *
 * A reusable React hook that wraps `expo-speech-recognition` into a clean,
 * plug-and-play abstraction for continuous live captioning.
 *
 * Some things we added on top of the raw module:
 *  1. On-device-first recognition with automatic network fallback.
 *  2. Auto-restart on silence / timeout for seamless continuous captioning.
 *  3. iOS audio session save / restore to avoid conflicts with
 *     react-native-vision-camera.
 *  4. Graceful permission handling — denied permissions never crash the app.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import { Platform } from 'react-native';
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
  type ExpoSpeechRecognitionErrorCode,
} from 'expo-speech-recognition';

// Types

/** Options accepted by the hook consumer. */
export interface UseSpeechRecognitionOptions {
  /** BCP-47 language tag. @default "en-US" */
  lang?: string;
  /**
   * When true, the hook will automatically restart recognition after a
   * silence / timeout event so captioning feels continuous.
   * @default true
   */
  continuous?: boolean;
  /**
   * Delay (ms) before auto-restarting after an unexpected stop.
   * Prevents tight restart loops.
   * @default 300
   */
  restartDelayMs?: number;
}

/** The public surface returned by the hook. */
export interface SpeechRecognitionState {
  /** Live transcript: accumulated final segments + current interim text. */
  text: string;
  /** Whether the recogniser is actively listening. */
  isListening: boolean;
  /** Last error code, or `null` when everything is fine. */
  error: string | null;
  /** Whether speech recognition is available on this device at all. */
  isAvailable: boolean;
  /** Request permissions & begin listening. */
  startListening: () => Promise<void>;
  /** Gracefully stop listening (processes a final result). */
  stopListening: () => Promise<void>;
  /** Clear the accumulated transcript. */
  resetTranscript: () => void;
}

// Error codes that should trigger an automatic restart rather than being
// surfaced to the consumer as hard errors.
const RESTARTABLE_ERRORS: Set<ExpoSpeechRecognitionErrorCode> = new Set([
  'no-speech',
  'speech-timeout',
  'network', // transient – worth retrying
  'client', // generic Android client error
]);

// Hook implementation

export function useSpeechRecognition(
  options: UseSpeechRecognitionOptions = {},
): SpeechRecognitionState {
  const {
    lang = 'en-US',
    continuous = true,
    restartDelayMs = 300,
  } = options;

  // State
  const [text, setText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs (mutable across renders, no re-render cost)

  /** Whether the user intends recognition to be active. */
  const shouldBeListening = useRef(false);

  /** Accumulated finalised transcript segments. */
  const finalTranscript = useRef('');

  /** Timer handle for the auto-restart debounce. */
  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Whether we already fell back to network-based recognition. */
  const fellBackToNetwork = useRef(false);

  /** Saved iOS audio session so we can restore it after recognition. */
  const savedIOSAudioSession = useRef<{
    category: string;
    categoryOptions: string[];
    mode: string;
  } | null>(null);

  // Derived
  const isAvailable = ExpoSpeechRecognitionModule.isRecognitionAvailable();

  // Helpers
  /** Cancel any pending restart timer. */
  const clearRestart = useCallback(() => {
    if (restartTimer.current) {
      clearTimeout(restartTimer.current);
      restartTimer.current = null;
    }
  }, []);

  /**
   * Save the current iOS audio session state so we can restore it when
   * speech recognition ends. This prevents conflicts with the camera.
   */
  const saveIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios') return;
    try {
      const session =
        ExpoSpeechRecognitionModule.getAudioSessionCategoryAndOptionsIOS();
      savedIOSAudioSession.current = session;
    } catch {
      // Non-critical – worst case we don't restore.
      savedIOSAudioSession.current = null;
    }
  }, []);

  /**
   * Restore the iOS audio session to whatever it was before we started
   * speech recognition.
   */
  const restoreIOSAudioSession = useCallback(() => {
    if (Platform.OS !== 'ios' || !savedIOSAudioSession.current) return;
    try {
      ExpoSpeechRecognitionModule.setCategoryIOS(
        savedIOSAudioSession.current as Parameters<
          typeof ExpoSpeechRecognitionModule.setCategoryIOS
        >[0],
      );
    } catch {
      // Best-effort restore.
    }
    savedIOSAudioSession.current = null;
  }, []);

  /** Internal: actually kick off the native recogniser. */
  const beginRecognition = useCallback(() => {
    // Determine whether to use on-device recognition.
    const supportsOnDevice =
      ExpoSpeechRecognitionModule.supportsOnDeviceRecognition();
    const useOnDevice = supportsOnDevice && !fellBackToNetwork.current;

    ExpoSpeechRecognitionModule.start({
      lang,
      interimResults: true,
      continuous,
      requiresOnDeviceRecognition: useOnDevice,
      addsPunctuation: true,
      // Extend Android's silence window so it doesn't cut off recognition
      // early in a quiet room.
      androidIntentOptions: {
        EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS: 10000,
        EXTRA_MASK_OFFENSIVE_WORDS: false,
      },
      // Tell iOS to use a reasonable audio session configuration that coexists
      // with camera audio.
      iosCategory: {
        category: 'playAndRecord',
        categoryOptions: ['defaultToSpeaker', 'allowBluetooth'],
        mode: 'measurement',
      },
      iosTaskHint: 'dictation',
    });
  }, [lang, continuous]);

  // Public methods

  const startListening = useCallback(async () => {
    if (!isAvailable) {
      setError('service-not-allowed');
      return;
    }

    // Request permissions – gracefully bail if denied.
    try {
      const result =
        await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!result.granted) {
        setError('not-allowed');
        return;
      }
    } catch {
      setError('not-allowed');
      return;
    }

    // Clear previous state.
    setError(null);
    fellBackToNetwork.current = false;
    shouldBeListening.current = true;

    // Save the iOS audio session before we mutate it.
    saveIOSAudioSession();

    beginRecognition();
  }, [isAvailable, saveIOSAudioSession, beginRecognition]);

  const stopListening = useCallback(async () => {
    shouldBeListening.current = false;
    clearRestart();

    try {
      ExpoSpeechRecognitionModule.stop();
    } catch {
      // Already stopped or destroyed – safe to ignore.
    }
  }, [clearRestart]);

  const resetTranscript = useCallback(() => {
    finalTranscript.current = '';
    setText('');
  }, []);

  // Event listeners

  useSpeechRecognitionEvent('start', () => {
    setIsListening(true);
    setError(null);
  });

  useSpeechRecognitionEvent('end', () => {
    setIsListening(false);
    restoreIOSAudioSession();

    // Auto-restart if the user hasn't explicitly stopped.
    if (shouldBeListening.current && continuous) {
      clearRestart();
      restartTimer.current = setTimeout(() => {
        if (shouldBeListening.current) {
          // Re-save audio session before restarting.
          saveIOSAudioSession();
          beginRecognition();
        }
      }, restartDelayMs);
    }
  });

  useSpeechRecognitionEvent('result', (event) => {
    const bestResult = event.results[0];
    if (!bestResult) return;

    if (event.isFinal) {
      // Append the final segment to our running transcript.
      const segment = bestResult.transcript.trim();
      if (segment) {
        finalTranscript.current = finalTranscript.current
          ? `${finalTranscript.current} ${segment}`
          : segment;
      }
      // Show the accumulated final transcript.
      setText(finalTranscript.current);
    } else {
      // Show the final transcript + the current partial (interim) result.
      const interim = bestResult.transcript.trim();
      const combined = finalTranscript.current
        ? `${finalTranscript.current} ${interim}`
        : interim;
      setText(combined);
    }
  });

  useSpeechRecognitionEvent('error', (event) => {
    const code = event.error;

    // If on-device recognition failed, fall back to network-based.
    if (
      (code === 'service-not-allowed' || code === 'language-not-supported') &&
      !fellBackToNetwork.current &&
      shouldBeListening.current
    ) {
      fellBackToNetwork.current = true;
      clearRestart();
      restartTimer.current = setTimeout(() => {
        if (shouldBeListening.current) {
          saveIOSAudioSession();
          beginRecognition();
        }
      }, restartDelayMs);
      return; // Don't surface this transient error.
    }

    // Restartable errors: swallow and let the `end` handler auto-restart.
    if (RESTARTABLE_ERRORS.has(code as ExpoSpeechRecognitionErrorCode)) {
      return;
    }

    // Non-recoverable errors get surfaced to the consumer.
    setError(code);

    // If permission was revoked mid-session, stop trying.
    if (code === 'not-allowed') {
      shouldBeListening.current = false;
      clearRestart();
    }
  });

  // Cleanup on unmount

  useEffect(() => {
    return () => {
      shouldBeListening.current = false;
      clearRestart();
      try {
        ExpoSpeechRecognitionModule.abort();
      } catch {
        // Component unmounting — swallow.
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Return the public API

  return {
    text,
    isListening,
    error,
    isAvailable,
    startListening,
    stopListening,
    resetTranscript,
  };
}
