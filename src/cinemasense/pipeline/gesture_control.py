"""
Gesture Control Mode - Hand gesture recognition using MediaPipe

This module provides hand gesture recognition for hands-free control of the
CinemaSense application. It uses MediaPipe for hand detection and landmark
tracking, with robust error handling for when MediaPipe is unavailable.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger("cinemasense.pipeline.gesture_control")

# Flag to track MediaPipe availability
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_ERROR_MESSAGE = ""

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_ERROR_MESSAGE = f"MediaPipe is not installed. Please install it with: pip install mediapipe. Error: {e}"
    logger.warning(MEDIAPIPE_ERROR_MESSAGE)
except Exception as e:
    MEDIAPIPE_ERROR_MESSAGE = f"Failed to import MediaPipe: {e}"
    logger.warning(MEDIAPIPE_ERROR_MESSAGE)


class GestureType(Enum):
    """Supported gesture types as per Requirement 8.2"""
    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PEACE = "peace"
    POINTING = "pointing"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    PINCH = "pinch"
    SPREAD = "spread"
    
    @classmethod
    def get_valid_types(cls) -> List[str]:
        """Return list of valid gesture type values"""
        return [g.value for g in cls]
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid gesture type"""
        return value in cls.get_valid_types()


@dataclass
class GestureResult:
    """Result of gesture detection"""
    gesture: GestureType
    confidence: float
    hand_landmarks: Optional[List]
    timestamp: float
    position: Tuple[float, float]  # Normalized position (0-1)
    cooldown_active: bool = False  # Whether cooldown prevented this gesture from triggering
    
    def is_valid(self) -> bool:
        """Check if this is a valid gesture result"""
        return (
            isinstance(self.gesture, GestureType) and
            0.0 <= self.confidence <= 1.0 and
            isinstance(self.timestamp, (int, float)) and
            len(self.position) == 2 and
            all(0.0 <= p <= 1.0 for p in self.position)
        )


@dataclass
class GestureAction:
    """Action to perform for a gesture"""
    gesture: GestureType
    action_name: str
    callback: Optional[Callable]
    description: str


# Default gesture-action mappings
DEFAULT_GESTURE_ACTIONS = {
    GestureType.THUMBS_UP: GestureAction(
        GestureType.THUMBS_UP,
        "approve",
        None,
        "Approve / Like"
    ),
    GestureType.THUMBS_DOWN: GestureAction(
        GestureType.THUMBS_DOWN,
        "reject",
        None,
        "Reject / Dislike"
    ),
    GestureType.OPEN_PALM: GestureAction(
        GestureType.OPEN_PALM,
        "pause",
        None,
        "Pause / Stop"
    ),
    GestureType.FIST: GestureAction(
        GestureType.FIST,
        "play",
        None,
        "Play / Start"
    ),
    GestureType.PEACE: GestureAction(
        GestureType.PEACE,
        "next",
        None,
        "Next Item"
    ),
    GestureType.POINTING: GestureAction(
        GestureType.POINTING,
        "select",
        None,
        "Select / Click"
    ),
    GestureType.SWIPE_LEFT: GestureAction(
        GestureType.SWIPE_LEFT,
        "previous",
        None,
        "Previous"
    ),
    GestureType.SWIPE_RIGHT: GestureAction(
        GestureType.SWIPE_RIGHT,
        "next",
        None,
        "Next"
    ),
    GestureType.PINCH: GestureAction(
        GestureType.PINCH,
        "zoom_out",
        None,
        "Zoom Out"
    ),
    GestureType.SPREAD: GestureAction(
        GestureType.SPREAD,
        "zoom_in",
        None,
        "Zoom In"
    )
}


class GestureController:
    """
    Hand gesture recognition and control.
    
    Provides robust gesture detection with:
    - Graceful handling when MediaPipe is unavailable (Requirement 8.5)
    - Cooldown enforcement to prevent repeated triggers (Requirement 8.3)
    - Valid gesture type classification (Requirement 8.2)
    - Proper resource cleanup
    """
    
    # Valid gesture types that can be returned by classification
    VALID_GESTURE_TYPES = set(GestureType)
    
    def __init__(self, cooldown_seconds: float = 0.5):
        """
        Initialize the gesture controller.
        
        Args:
            cooldown_seconds: Minimum time between gesture triggers (default 0.5s)
        """
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        self.initialized = False
        self.gesture_actions = DEFAULT_GESTURE_ACTIONS.copy()
        self.last_gesture = GestureType.NONE
        self.last_gesture_time = 0.0
        self.gesture_cooldown = max(0.0, cooldown_seconds)  # Ensure non-negative
        self.position_history: List[Tuple[float, float]] = []
        self.history_length = 10
        self._error_message: Optional[str] = None
        self._mediapipe_available = MEDIAPIPE_AVAILABLE
        
        # Track gesture timestamps for cooldown per gesture type
        self._gesture_timestamps: Dict[GestureType, float] = {
            g: 0.0 for g in GestureType
        }
    
    @property
    def error_message(self) -> Optional[str]:
        """Get the last error message, if any"""
        return self._error_message
    
    @property
    def is_mediapipe_available(self) -> bool:
        """Check if MediaPipe is available for use"""
        return self._mediapipe_available
        
    def initialize(self) -> bool:
        """
        Initialize MediaPipe hands detection.
        
        Returns:
            True if initialization successful, False otherwise.
            
        Requirement 8.1: Initialize MediaPipe hand detection when enabled
        Requirement 8.5: Return clear error message if MediaPipe unavailable
        """
        # Check if MediaPipe is available
        if not self._mediapipe_available:
            self._error_message = MEDIAPIPE_ERROR_MESSAGE
            logger.error(f"Cannot initialize gesture controller: {self._error_message}")
            self.initialized = False
            return False
        
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.initialized = True
            self._error_message = None
            logger.info("Gesture controller initialized successfully")
            return True
        except ImportError as e:
            self._error_message = f"MediaPipe not available: {e}. Please install with: pip install mediapipe"
            self._mediapipe_available = False
            logger.error(self._error_message)
            return False
        except Exception as e:
            self._error_message = f"Failed to initialize gesture controller: {e}"
            logger.error(self._error_message, exc_info=True)
            return False
    
    def _is_cooldown_active(self, gesture: GestureType, current_time: float) -> bool:
        """
        Check if cooldown is active for a specific gesture.
        
        Requirement 8.3: Apply cooldown period to prevent repeated triggers
        
        Args:
            gesture: The gesture type to check
            current_time: Current timestamp
            
        Returns:
            True if cooldown is active (gesture should be suppressed)
        """
        if gesture == GestureType.NONE:
            return False
            
        last_time = self._gesture_timestamps.get(gesture, 0.0)
        return (current_time - last_time) < self.gesture_cooldown
    
    def _update_gesture_timestamp(self, gesture: GestureType, timestamp: float) -> None:
        """Update the timestamp for a gesture trigger"""
        if gesture != GestureType.NONE:
            self._gesture_timestamps[gesture] = timestamp
            self.last_gesture = gesture
            self.last_gesture_time = timestamp
    
    def process_frame(self, frame: np.ndarray) -> Tuple[GestureResult, np.ndarray]:
        """
        Process a frame and detect gestures.
        
        Args:
            frame: Input BGR frame from video/camera
            
        Returns:
            Tuple of (GestureResult, annotated_frame)
            
        Requirement 8.4: Draw hand landmarks and gesture labels on video feed
        Requirement 8.5: Return clear error if MediaPipe unavailable
        """
        current_time = time.time()
        default_result = GestureResult(
            gesture=GestureType.NONE,
            confidence=0.0,
            hand_landmarks=None,
            timestamp=current_time,
            position=(0.5, 0.5),
            cooldown_active=False
        )
        
        # Check if MediaPipe is available
        if not self._mediapipe_available:
            self._error_message = MEDIAPIPE_ERROR_MESSAGE
            logger.debug("MediaPipe not available, returning default result")
            return default_result, frame
        
        # Try to initialize if not already done
        if not self.initialized:
            if not self.initialize():
                return default_result, frame
        
        # Validate input frame
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to process_frame")
            return default_result, frame
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            annotated_frame = frame.copy()
            gesture_result = default_result
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks (Requirement 8.4)
                    self.mp_draw.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture (Requirement 8.2)
                    gesture, confidence = self._classify_gesture(hand_landmarks)
                    
                    # Validate gesture type
                    if not isinstance(gesture, GestureType):
                        logger.warning(f"Invalid gesture type returned: {gesture}")
                        gesture = GestureType.NONE
                        confidence = 0.0
                    
                    # Get hand position
                    wrist = hand_landmarks.landmark[0]
                    position = (
                        max(0.0, min(1.0, wrist.x)),
                        max(0.0, min(1.0, wrist.y))
                    )
                    
                    # Update position history for swipe detection
                    self.position_history.append(position)
                    if len(self.position_history) > self.history_length:
                        self.position_history.pop(0)
                    
                    # Check for swipe gestures
                    swipe_gesture = self._detect_swipe()
                    if swipe_gesture != GestureType.NONE:
                        gesture = swipe_gesture
                        confidence = 0.8
                    
                    # Check cooldown (Requirement 8.3)
                    cooldown_active = self._is_cooldown_active(gesture, current_time)
                    
                    if gesture != GestureType.NONE:
                        if not cooldown_active:
                            # Update timestamp for this gesture
                            self._update_gesture_timestamp(gesture, current_time)
                            
                            gesture_result = GestureResult(
                                gesture=gesture,
                                confidence=confidence,
                                hand_landmarks=hand_landmarks,
                                timestamp=current_time,
                                position=position,
                                cooldown_active=False
                            )
                        else:
                            # Gesture detected but cooldown active
                            gesture_result = GestureResult(
                                gesture=gesture,
                                confidence=confidence,
                                hand_landmarks=hand_landmarks,
                                timestamp=current_time,
                                position=position,
                                cooldown_active=True
                            )
                    
                    # Draw gesture label (Requirement 8.4)
                    h, w = frame.shape[:2]
                    label = f"{gesture.value} ({confidence:.2f})"
                    if cooldown_active:
                        label += " [cooldown]"
                    cv2.putText(
                        annotated_frame, label,
                        (int(position[0] * w), int(position[1] * h) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
            
            return gesture_result, annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing frame for gestures: {e}", exc_info=True)
            self._error_message = f"Error processing frame: {e}"
            return default_result, frame
    
    def _classify_gesture(self, landmarks) -> Tuple[GestureType, float]:
        """
        Classify hand gesture from landmarks.
        
        Requirement 8.2: Recognize gestures: thumbs_up, thumbs_down, open_palm,
                        fist, peace, pointing, swipe_left, swipe_right, pinch
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Tuple of (GestureType, confidence) - always returns valid GestureType
        """
        try:
            # Extract landmark positions
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]
            
            thumb_ip = landmarks.landmark[3]
            index_pip = landmarks.landmark[6]
            middle_pip = landmarks.landmark[10]
            ring_pip = landmarks.landmark[14]
            pinky_pip = landmarks.landmark[18]
            
            wrist = landmarks.landmark[0]
            
            # Calculate finger states (extended or not)
            thumb_extended = thumb_tip.x < thumb_ip.x  # For right hand
            index_extended = index_tip.y < index_pip.y
            middle_extended = middle_tip.y < middle_pip.y
            ring_extended = ring_tip.y < ring_pip.y
            pinky_extended = pinky_tip.y < pinky_pip.y
            
            fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
            num_extended = sum(fingers_extended)
            
            # Default confidence for recognized gestures
            confidence = 0.8
            
            # Classify gestures (Requirement 8.2)
            
            # Open palm (all fingers extended)
            if num_extended >= 4:
                return GestureType.OPEN_PALM, confidence
            
            # Fist (no fingers extended)
            if num_extended == 0:
                return GestureType.FIST, confidence
            
            # Thumbs up (only thumb extended, thumb above wrist)
            if thumb_extended and not any(fingers_extended[1:]) and thumb_tip.y < wrist.y:
                return GestureType.THUMBS_UP, confidence
            
            # Thumbs down (only thumb extended, thumb below wrist)
            if thumb_extended and not any(fingers_extended[1:]) and thumb_tip.y > wrist.y:
                return GestureType.THUMBS_DOWN, confidence
            
            # Peace sign (index and middle extended)
            if index_extended and middle_extended and not ring_extended and not pinky_extended:
                return GestureType.PEACE, confidence
            
            # Pointing (only index extended)
            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return GestureType.POINTING, confidence
            
            # Pinch (thumb and index close together)
            thumb_index_dist = np.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 + 
                (thumb_tip.y - index_tip.y) ** 2
            )
            if thumb_index_dist < 0.05:
                return GestureType.PINCH, confidence
            
            # No recognized gesture
            return GestureType.NONE, 0.0
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error classifying gesture from landmarks: {e}")
            return GestureType.NONE, 0.0
        except Exception as e:
            logger.error(f"Unexpected error in gesture classification: {e}", exc_info=True)
            return GestureType.NONE, 0.0
    
    def _detect_swipe(self) -> GestureType:
        """Detect swipe gestures from position history"""
        if len(self.position_history) < self.history_length:
            return GestureType.NONE
        
        # Calculate horizontal movement
        start_x = self.position_history[0][0]
        end_x = self.position_history[-1][0]
        x_diff = end_x - start_x
        
        # Calculate vertical movement
        start_y = self.position_history[0][1]
        end_y = self.position_history[-1][1]
        y_diff = end_y - start_y
        
        # Swipe threshold
        threshold = 0.2
        
        if abs(x_diff) > threshold and abs(x_diff) > abs(y_diff):
            self.position_history.clear()
            if x_diff > 0:
                return GestureType.SWIPE_RIGHT
            else:
                return GestureType.SWIPE_LEFT
        
        return GestureType.NONE
    
    def register_action(self, gesture: GestureType, action_name: str, 
                       callback: Callable, description: str):
        """Register a custom action for a gesture"""
        self.gesture_actions[gesture] = GestureAction(
            gesture, action_name, callback, description
        )
    
    def execute_gesture_action(self, gesture_result: GestureResult) -> Optional[str]:
        """Execute the action associated with a gesture"""
        if gesture_result.gesture == GestureType.NONE:
            return None
        
        action = self.gesture_actions.get(gesture_result.gesture)
        if action:
            if action.callback:
                action.callback()
            return action.action_name
        
        return None
    
    def get_gesture_guide(self) -> List[Dict]:
        """Get guide of available gestures and their actions"""
        return [
            {
                "gesture": action.gesture.value,
                "action": action.action_name,
                "description": action.description
            }
            for action in self.gesture_actions.values()
        ]
    
    def cleanup(self) -> None:
        """
        Clean up resources properly.
        
        This method should be called when the gesture controller is no longer needed
        to ensure proper release of MediaPipe resources.
        """
        try:
            if self.hands is not None:
                self.hands.close()
                self.hands = None
            self.mp_hands = None
            self.mp_draw = None
            self.initialized = False
            self.position_history.clear()
            self._gesture_timestamps = {g: 0.0 for g in GestureType}
            self.last_gesture = GestureType.NONE
            self.last_gesture_time = 0.0
            logger.info("Gesture controller cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during gesture controller cleanup: {e}", exc_info=True)
    
    def reset_cooldowns(self) -> None:
        """Reset all gesture cooldown timers"""
        self._gesture_timestamps = {g: 0.0 for g in GestureType}
        self.last_gesture = GestureType.NONE
        self.last_gesture_time = 0.0
    
    def set_cooldown(self, seconds: float) -> None:
        """
        Set the cooldown period for gesture triggers.
        
        Args:
            seconds: Cooldown period in seconds (must be non-negative)
        """
        self.gesture_cooldown = max(0.0, seconds)
    
    def get_status(self) -> Dict:
        """
        Get the current status of the gesture controller.
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "mediapipe_available": self._mediapipe_available,
            "error_message": self._error_message,
            "cooldown_seconds": self.gesture_cooldown,
            "last_gesture": self.last_gesture.value if self.last_gesture else None,
            "last_gesture_time": self.last_gesture_time
        }


def is_mediapipe_available() -> Tuple[bool, str]:
    """
    Check if MediaPipe is available for gesture control.
    
    Returns:
        Tuple of (is_available, error_message)
        
    Requirement 8.5: Return clear error message if MediaPipe unavailable
    """
    if MEDIAPIPE_AVAILABLE:
        return True, ""
    return False, MEDIAPIPE_ERROR_MESSAGE


def create_gesture_overlay(frame: np.ndarray, gesture: GestureType, 
                          action: str = None) -> np.ndarray:
    """Create a visual overlay showing the detected gesture"""
    result = frame.copy()
    h, w = result.shape[:2]
    
    # Create semi-transparent overlay
    overlay = result.copy()
    
    # Draw gesture indicator
    indicator_size = 100
    indicator_x = w - indicator_size - 20
    indicator_y = 20
    
    # Background circle
    cv2.circle(
        overlay,
        (indicator_x + indicator_size // 2, indicator_y + indicator_size // 2),
        indicator_size // 2,
        (0, 0, 0),
        -1
    )
    
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Draw gesture icon (simplified)
    icon_color = (0, 255, 0) if gesture != GestureType.NONE else (100, 100, 100)
    
    # Draw gesture name
    gesture_text = gesture.value.replace("_", " ").title()
    cv2.putText(
        result, gesture_text,
        (indicator_x, indicator_y + indicator_size + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, icon_color, 2
    )
    
    # Draw action if provided
    if action:
        cv2.putText(
            result, f"Action: {action}",
            (indicator_x, indicator_y + indicator_size + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return result