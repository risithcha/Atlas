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
  /** Faded / translucent variants */
  primaryFaded: 'rgba(76, 175, 80, 0.2)',
  secondaryFaded: 'rgba(33, 150, 243, 0.2)',
  whiteFaded: 'rgba(255, 255, 255, 0.2)',
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
  /** Button label */
  button: { fontSize: 18, fontWeight: 'bold' as const },
  /** Subtitle / tagline */
  subtitle: { fontSize: 16, fontWeight: '600' as const, letterSpacing: 2 },
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

// ---------------------------------------------------------------------------
// Component-level size tokens
// ---------------------------------------------------------------------------
export const SIZES = {
  /** Tab bar total height (icon + label + padding) */
  tabBarHeight: 90,
  tabBarPaddingBottom: 28,
  /** Unified header height (excl. safe-area insets) */
  headerHeight: 56,
  /** Standard action-button min height */
  buttonMinHeight: 60,
  /** Round icon-button dimensions */
  iconButton: 44,
  iconButtonRadius: 22,
  /** Status dot */
  statusDot: 12,
  statusDotSmall: 10,
} as const;

// ---------------------------------------------------------------------------
// Per-tab accent colours  (Vision = blue, Hearing = green)
// ---------------------------------------------------------------------------
export const TAB_ACCENT = {
  Vision: COLORS.secondary,
  Hearing: COLORS.primary,
} as const;
