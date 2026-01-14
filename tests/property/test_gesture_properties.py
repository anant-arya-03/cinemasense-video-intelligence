"""
Property-based tests for Gesture Control Mode.

Feature: cinemasense-stabilization
Property 16: Gesture Cooldown Enforcement
Validates: Requirements 8.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import time

from src.cinemasense.pipeline.gesture_control import (
    GestureController,
    GestureType,
)


class TestGestureCooldownEnforcement:
    """
    Property 16: Gesture Cooldown Enforcement

    For any sequence of identical gestures detected within the cooldown period,
    only the first gesture SHALL trigger an action.

    **Validates: Requirements 8.3**
    """

    @given(
        cooldown_seconds=st.floats(min_value=0.1, max_value=2.0),
        gesture_type=st.sampled_from([g for g in GestureType if g != GestureType.NONE]),
        num_rapid_gestures=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_cooldown_blocks_rapid_identical_gestures(
        self,
        cooldown_seconds,
        gesture_type,
        num_rapid_gestures
    ):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        For any sequence of identical gestures detected within the cooldown period,
        only the first gesture SHALL trigger an action.
        """
        controller = GestureController(cooldown_seconds=cooldown_seconds)

        base_time = time.time()

        # First gesture should NOT be blocked by cooldown
        first_cooldown_active = controller._is_cooldown_active(gesture_type, base_time)
        assert not first_cooldown_active, "First gesture should not be blocked by cooldown"

        # Update timestamp to simulate first gesture being triggered
        controller._update_gesture_timestamp(gesture_type, base_time)

        # Subsequent gestures within cooldown period SHOULD be blocked
        for i in range(1, num_rapid_gestures):
            within_cooldown_time = base_time + (cooldown_seconds * 0.5)

            cooldown_active = controller._is_cooldown_active(gesture_type, within_cooldown_time)
            assert cooldown_active, (
                f"Gesture {i} at time {within_cooldown_time - base_time:.3f}s "
                f"should be blocked by cooldown (cooldown={cooldown_seconds}s)"
            )

    @given(
        cooldown_seconds=st.floats(min_value=0.1, max_value=1.0),
        gesture_type=st.sampled_from([g for g in GestureType if g != GestureType.NONE])
    )
    @settings(max_examples=100, deadline=None)
    def test_cooldown_expires_after_period(self, cooldown_seconds, gesture_type):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        After the cooldown period expires, the same gesture SHALL be allowed to trigger again.
        """
        controller = GestureController(cooldown_seconds=cooldown_seconds)

        base_time = time.time()

        # First gesture triggers
        controller._update_gesture_timestamp(gesture_type, base_time)

        # After cooldown period, gesture should be allowed again
        after_cooldown_time = base_time + cooldown_seconds + 0.01

        cooldown_active = controller._is_cooldown_active(gesture_type, after_cooldown_time)
        assert not cooldown_active, (
            f"Gesture should be allowed after cooldown period expires "
            f"(time elapsed: {after_cooldown_time - base_time:.3f}s, cooldown: {cooldown_seconds}s)"
        )

    @given(
        cooldown_seconds=st.floats(min_value=0.1, max_value=1.0),
        gesture_a=st.sampled_from([g for g in GestureType if g != GestureType.NONE]),
        gesture_b=st.sampled_from([g for g in GestureType if g != GestureType.NONE])
    )
    @settings(max_examples=100, deadline=None)
    def test_cooldown_is_per_gesture_type(self, cooldown_seconds, gesture_a, gesture_b):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        Cooldown for one gesture type SHALL NOT block different gesture types.
        """
        assume(gesture_a != gesture_b)

        controller = GestureController(cooldown_seconds=cooldown_seconds)

        base_time = time.time()

        # Trigger gesture A
        controller._update_gesture_timestamp(gesture_a, base_time)

        # Gesture A should be blocked within cooldown
        within_cooldown_time = base_time + (cooldown_seconds * 0.5)
        assert controller._is_cooldown_active(gesture_a, within_cooldown_time), \
            "Gesture A should be blocked by its own cooldown"

        # Gesture B should NOT be blocked by gesture A's cooldown
        assert not controller._is_cooldown_active(gesture_b, within_cooldown_time), \
            "Gesture B should not be blocked by gesture A's cooldown"

    @given(cooldown_seconds=st.floats(min_value=0.0, max_value=0.0))
    @settings(max_examples=10, deadline=None)
    def test_zero_cooldown_allows_all_gestures(self, cooldown_seconds):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        With zero cooldown, all gestures SHALL be allowed immediately.
        """
        controller = GestureController(cooldown_seconds=cooldown_seconds)

        base_time = time.time()
        gesture = GestureType.THUMBS_UP

        # Trigger gesture
        controller._update_gesture_timestamp(gesture, base_time)

        # Even immediately after, should not be blocked with zero cooldown
        assert not controller._is_cooldown_active(gesture, base_time), \
            "With zero cooldown, gestures should never be blocked"

    @given(gesture_type=st.sampled_from([g for g in GestureType if g != GestureType.NONE]))
    @settings(max_examples=50, deadline=None)
    def test_none_gesture_never_has_cooldown(self, gesture_type):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        GestureType.NONE SHALL never be subject to cooldown.
        """
        controller = GestureController(cooldown_seconds=1.0)

        base_time = time.time()

        # NONE gesture should never have cooldown active
        assert not controller._is_cooldown_active(GestureType.NONE, base_time), \
            "NONE gesture should never have cooldown"

        # Even after triggering other gestures
        controller._update_gesture_timestamp(gesture_type, base_time)
        assert not controller._is_cooldown_active(GestureType.NONE, base_time + 0.1), \
            "NONE gesture should never have cooldown even after other gestures"

    @given(
        cooldown_seconds=st.floats(min_value=0.1, max_value=2.0),
        gesture_type=st.sampled_from([g for g in GestureType if g != GestureType.NONE])
    )
    @settings(max_examples=50, deadline=None)
    def test_reset_cooldowns_clears_all_timers(self, cooldown_seconds, gesture_type):
        """
        Feature: cinemasense-stabilization, Property 16: Gesture Cooldown Enforcement

        After reset_cooldowns() is called, all gestures SHALL be allowed immediately.
        """
        controller = GestureController(cooldown_seconds=cooldown_seconds)

        base_time = time.time()

        # Trigger gesture
        controller._update_gesture_timestamp(gesture_type, base_time)

        # Verify cooldown is active
        within_cooldown_time = base_time + (cooldown_seconds * 0.5)
        assert controller._is_cooldown_active(gesture_type, within_cooldown_time), \
            "Cooldown should be active before reset"

        # Reset cooldowns
        controller.reset_cooldowns()

        # After reset, gesture should be allowed
        assert not controller._is_cooldown_active(gesture_type, within_cooldown_time), \
            "After reset_cooldowns(), gesture should be allowed"
