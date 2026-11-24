#!/usr/bin/env python3
"""Jetson Nano classroom demo: two DIFFERENT people's poses trigger a GPIO relay.

Person A: one arm stretched horizontally, a simulated punch/strike pose.
Person B: BOTH wrists raised just above the shoulders, a help/signal pose.
The relay turns on only when there are at least two people and A != B.
This sensitivity-focused version relaxes thresholds for easier classroom demonstrations.

This is gesture detection for an instructed demonstration, NOT a model of
violence, intent, or personal safety.  Never use it as a real emergency alarm.

Files: place this script at ~/jetson-inference/python/examples/
Test without GPIO: python3 posenet_gpio_two_person_alert.py /dev/video0 --dry-run
Actual relay IN2:  python3 posenet_gpio_two_person_alert.py /dev/video0
Active LOW board:  python3 posenet_gpio_two_person_alert.py /dev/video0 --active-low

Physical Jetson pin BOARD 13 -> relay IN2, plus compatible power and GND.
Power the Jetson OFF before connecting or adjusting wiring.  Only test the
relay module's own indicator LED; do not wire mains/high-power loads.
"""

import argparse
import math
import sys

import jetson.inference
import jetson.utils

RELAY_PIN = 13  # GPIO.BOARD physical pin 13 -> relay IN2 (BCM 27)


def keypoint(pose, name):
    """Return a named detected keypoint or None (Keypoints is sparse)."""
    index = pose.FindKeypoint(name)
    if index < 0:
        return None
    return pose.Keypoints[index]


def angle_degrees(a, b, c):
    """Angle ABC in image-space degrees, or None for degenerate vectors."""
    vx, vy = a.x - b.x, a.y - b.y
    wx, wy = c.x - b.x, c.y - b.y
    length = math.hypot(vx, vy) * math.hypot(wx, wy)
    if length <= 1e-6:
        return None
    cosine = (vx * wx + vy * wy) / length
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def strike_pose(pose, straight_angle=125.0, horizontal_ratio=0.65, vertical_ratio=0.90):
    """One arm is straight and *horizontally* extended to the side.

    This is a deliberately simplified, staged 'punch-like' gesture. PoseNet
    cannot see a clenched fist or infer that someone intends to hurt another.
    """
    left_shoulder = keypoint(pose, 'left_shoulder')
    right_shoulder = keypoint(pose, 'right_shoulder')
    if left_shoulder is None or right_shoulder is None:
        return False

    shoulder_width = math.hypot(
        left_shoulder.x - right_shoulder.x,
        left_shoulder.y - right_shoulder.y
    )
    if shoulder_width < 8:
        return False

    for side in ('left', 'right'):
        shoulder = keypoint(pose, side + '_shoulder')
        elbow = keypoint(pose, side + '_elbow')
        wrist = keypoint(pose, side + '_wrist')
        if shoulder is None or elbow is None or wrist is None:
            continue
        elbow_angle = angle_degrees(shoulder, elbow, wrist)
        if elbow_angle is None or elbow_angle < straight_angle:
            continue

        horizontal = abs(wrist.x - shoulder.x)
        vertical = abs(wrist.y - shoulder.y)
        if (horizontal > horizontal_ratio * shoulder_width
                and vertical < vertical_ratio * shoulder_width
                and horizontal > 0.8 * vertical):
            return True
    return False


def hands_up_signal(pose, hand_above=0.05):
    """Both wrists clearly higher than their respective shoulders."""
    left_shoulder = keypoint(pose, 'left_shoulder')
    right_shoulder = keypoint(pose, 'right_shoulder')
    left_wrist = keypoint(pose, 'left_wrist')
    right_wrist = keypoint(pose, 'right_wrist')
    if any(p is None for p in
           (left_shoulder, right_shoulder, left_wrist, right_wrist)):
        return False

    # Body-relative threshold works across camera resolution and distance.
    shoulder_width = math.hypot(
        left_shoulder.x - right_shoulder.x,
        left_shoulder.y - right_shoulder.y
    )
    if shoulder_width < 8:
        return False
    height_above = hand_above * shoulder_width
    return (left_shoulder.y - left_wrist.y > height_above
            and right_shoulder.y - right_wrist.y > height_above)


def distinct_person_pair(poses, strike_angle=125.0, horizontal_ratio=0.65,
                         vertical_ratio=0.90, hand_above=0.05):
    """Return (attacker_index, signaller_index), always different, or None."""
    if len(poses) < 2:
        return None
    strike = [strike_pose(p, strike_angle, horizontal_ratio, vertical_ratio) for p in poses]
    signal = [hands_up_signal(p, hand_above) for p in poses]
    for i, is_strike in enumerate(strike):
        if is_strike:
            for j, is_signal in enumerate(signal):
                if is_signal and i != j:
                    return (i, j)
    return None


class DebouncedOutput:
    """Consecutive-frame confirmation prevents momentary GPIO flicker."""
    def __init__(self, on_frames, off_frames):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.state = False
        self.counter = 0

    def update(self, target):
        if bool(target) == self.state:
            self.counter = 0
            return False
        self.counter += 1
        threshold = self.on_frames if target else self.off_frames
        if self.counter >= threshold:
            self.state = bool(target)
            self.counter = 0
            return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Two-person staged gesture + Jetson Nano GPIO alert'
    )
    parser.add_argument('input_URI', nargs='?', default='/dev/video0')
    parser.add_argument('output_URI', nargs='?', default='display://0')
    parser.add_argument('--network', default='resnet18-body')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='PoseNet confidence threshold (default 0.10)')
    parser.add_argument('--strike-angle', type=float, default=125.0,
                        help='minimum elbow angle for the staged strike (default 125)')
    parser.add_argument('--strike-horizontal', type=float, default=0.65,
                        help='minimum horizontal wrist reach, in shoulder widths (default 0.65)')
    parser.add_argument('--strike-vertical', type=float, default=0.90,
                        help='maximum vertical wrist offset, in shoulder widths (default 0.90)')
    parser.add_argument('--hand-above', type=float, default=0.05,
                        help='wrists above shoulders, in shoulder widths (default 0.05)')
    parser.add_argument('--confirm', type=int, default=2,
                        help='consecutive frames to turn the relay ON (default 2)')
    parser.add_argument('--clear', type=int, default=4,
                        help='consecutive frames to turn it OFF (default 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='show detection and simulated relay state; do not open GPIO')
    parser.add_argument('--active-low', action='store_true',
                        help='relay turns ON when GPIO level is LOW')
    args = parser.parse_args()
    if not (0.0 < args.threshold < 1.0):
        parser.error('--threshold must be between 0 and 1')
    if not (0.0 <= args.strike_angle <= 180.0):
        parser.error('--strike-angle must be between 0 and 180')
    if not (0.0 <= args.strike_horizontal and 0.0 <= args.strike_vertical):
        parser.error('--strike-horizontal and --strike-vertical cannot be negative')
    if not (-0.25 <= args.hand_above <= 1.0):
        parser.error('--hand-above must be between -0.25 and 1')
    if args.confirm < 1 or args.clear < 1:
        parser.error('--confirm and --clear must be positive')

    gpio = None
    if not args.dry_run:
        import Jetson.GPIO as GPIO
        gpio = GPIO
        gpio.setmode(gpio.BOARD)
        active_level = gpio.LOW if args.active_low else gpio.HIGH
        inactive_level = gpio.HIGH if args.active_low else gpio.LOW

    def relay_write(is_on):
        if gpio is not None:
            gpio.output(RELAY_PIN, active_level if is_on else inactive_level)

    relay = DebouncedOutput(args.confirm, args.clear)
    try:
        if gpio is not None:
            gpio.setup(RELAY_PIN, gpio.OUT, initial=inactive_level)

        # Avoid handing the custom GPIO flags to jetson-inference parsers.
        backend_argv = [sys.argv[0]]
        net = jetson.inference.poseNet(args.network, backend_argv, args.threshold)
        camera = jetson.utils.videoSource(args.input_URI, argv=backend_argv)
        output = jetson.utils.videoOutput(args.output_URI, argv=backend_argv)
        font = jetson.utils.cudaFont(size=22)

        print('Scene: A extends one arm sideways (staged strike), B raises both hands.')
        print('Only DIFFERENT people may satisfy the two roles; at least 2 poses required.')
        print('GPIO: BOARD 13 -> relay IN2 | mode: {}'.format(
            'DRY RUN' if args.dry_run else 'active LOW' if args.active_low else 'active HIGH'))
        print('Press Ctrl+C to stop.')

        while output.IsStreaming():
            img = camera.Capture()
            if img is None:
                if not camera.IsStreaming():
                    break
                continue
            poses = net.Process(img, overlay='links,keypoints')
            strikes = [strike_pose(p, args.strike_angle,
                                   args.strike_horizontal, args.strike_vertical)
                       for p in poses]
            signals = [hands_up_signal(p, args.hand_above) for p in poses]
            pair = next(((i, j) for i, a in enumerate(strikes) if a
                         for j, b in enumerate(signals) if b and i != j), None)
            target = pair is not None

            if relay.update(target):
                relay_write(relay.state)
                print('ALERT {} | {} pose(s), pair={}'.format(
                    'ON' if relay.state else 'OFF', len(poses), pair))

            label_pair = 'none' if pair is None else 'A=#{} B=#{}'.format(
                pair[0] + 1, pair[1] + 1)
            font.OverlayText(img,
                             text='People: {} | Two-person gesture: {}'.format(
                                 len(poses), 'YES' if target else 'NO'),
                             x=8, y=8, color=font.White, background=font.Gray40)
            font.OverlayText(img,
                             text='Strike + hands-up: {} | Relay IN2: {}'.format(
                                 label_pair, 'ON' if relay.state else 'OFF'),
                             x=8, y=40, color=font.White, background=font.Gray40)
            strike_ids = ','.join(str(i + 1) for i, ok in enumerate(strikes) if ok) or '-'
            signal_ids = ','.join(str(i + 1) for i, ok in enumerate(signals) if ok) or '-'
            font.OverlayText(img,
                             text='Strike persons: {} | Signal persons: {}'.format(
                                 strike_ids, signal_ids),
                             x=8, y=72, color=font.White, background=font.Gray40)
            output.Render(img)
            output.SetStatus('Two-person PoseNet GPIO | {:.0f} FPS'.format(
                net.GetNetworkFPS()))
            if not camera.IsStreaming():
                break
    except KeyboardInterrupt:
        print('Stopped by user')
    finally:
        if gpio is not None:
            try:
                relay_write(False)
            finally:
                gpio.cleanup()
            print('Relay OFF; GPIO cleaned up')


if __name__ == '__main__':
    main()
