/**
 * Atlas Mobile – Navigation Router
 *
 * This is the app's entry component.  It sets up a bottom-tab navigator
 * with two modes that mirror the desktop app:
 *   • Vision Assist  (camera + object detection)
 *   • Hearing Assist (live speech-to-text captioning)
 *
 * Each screen manages its own lifecycle (Camera `isActive`, speech
 * recognition start/stop) via the `useIsFocused` hook so resources are
 * released when the user switches tabs.
 */
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from '@expo/vector-icons/Ionicons';

import { VisionScreen, HearingScreen } from './src/screens';
import { COLORS } from './src/theme';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={{
            headerShown: false,
            animation: 'fade',
            lazy: true,
            freezeOnBlur: true,
            tabBarStyle: {
              backgroundColor: COLORS.background,
              borderTopColor: COLORS.surface,
              borderTopWidth: 1,
              height: 90,
              paddingBottom: 28,
              paddingTop: 8,
            },
            tabBarActiveTintColor: COLORS.primary,
            tabBarInactiveTintColor: COLORS.textMuted,
            tabBarLabelStyle: {
              fontSize: 12,
              fontWeight: '600',
            },
          }}
        >
          <Tab.Screen
            name="Vision"
            component={VisionScreen}
            options={{
              tabBarLabel: 'Vision',
              tabBarIcon: ({ color, size, focused }) => (
                <Ionicons
                  name={focused ? 'eye' : 'eye-outline'}
                  size={size}
                  color={color}
                />
              ),
              tabBarAccessibilityLabel: 'Vision Assist Mode',
            }}
          />
          <Tab.Screen
            name="Hearing"
            component={HearingScreen}
            options={{
              tabBarLabel: 'Hearing',
              tabBarIcon: ({ color, size, focused }) => (
                <Ionicons
                  name={focused ? 'ear' : 'ear-outline'}
                  size={size}
                  color={color}
                />
              ),
              tabBarAccessibilityLabel: 'Hearing Assist Mode',
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </>
  );
}
