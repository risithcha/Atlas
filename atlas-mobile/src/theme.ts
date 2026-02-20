/**
 * Atlas Mobile – Shared Design Tokens
 *
 * Single source of truth for colours, typography, spacing, and model
 * constants.  Mirrors the desktop app's palette.
 */

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------
export const COLORS = {
  /** App-wide dark background */
  background: '#1a1a1a',
  /** Card / elevated surface */
  surface: '#2d2d2d',
  /** Atlas green – primary action, Hearing Assist, success states */
  primary: '#4CAF50',
  primaryDark: '#2E7D32',
  /** Atlas blue – secondary action, Vision Assist */
  secondary: '#2196F3',
  secondaryDark: '#1976D2',
  /** Primary text on dark */
  text: '#ffffff',
  /** Secondary / muted text */
  textMuted: '#888888',
  textSecondary: '#aaaaaa',
  /** Semi-transparent overlays */
  overlay: 'rgba(0, 0, 0, 0.6)',
  /** Danger / error / stop states */
  danger: '#f44336',
  dangerDark: '#D32F2F',
  /** Warning / alerts */
  warning: '#FFC107',
  warningBright: '#FFEB3B',
} as const;

// ---------------------------------------------------------------------------
// Bounding-box colours
// ---------------------------------------------------------------------------
export const BOX_COLORS = [
  '#4CAF50',
  '#2196F3',
  '#FF9800',
  '#E91E63',
  '#9C27B0',
  '#00BCD4',
  '#FFEB3B',
  '#FF5722',
  '#795548',
  '#607D8B',
] as const;

// ---------------------------------------------------------------------------
// Model / inference constants
// ---------------------------------------------------------------------------
export const MODEL_INPUT_SIZE = 300;
export const CONFIDENCE_THRESHOLD = 0.45;
export const MAX_DETECTIONS = 10;
export const INFERENCE_FPS = 5;
export const MIN_BOX_AREA = 0.005;

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------
export const TYPOGRAPHY = {
  /** Logo / brand text */
  logo: { fontSize: 48, fontWeight: 'bold' as const, letterSpacing: 8 },
  logoSmall: { fontSize: 24, fontWeight: 'bold' as const, letterSpacing: 4 },
  /** Screen titles */
  title: { fontSize: 28, fontWeight: 'bold' as const },
  /** Section headers */
  heading: { fontSize: 22, fontWeight: '600' as const },
  /** Body / caption text */
  body: { fontSize: 16 },
  bodyLarge: { fontSize: 22 },
  /** Small / meta text */
  caption: { fontSize: 13 },
  small: { fontSize: 12 },
  label: { fontSize: 11, fontWeight: '700' as const },
} as const;

// ---------------------------------------------------------------------------
// Spacing & radii
// ---------------------------------------------------------------------------
export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 40,
} as const;

export const RADII = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  round: 999,
} as const;
