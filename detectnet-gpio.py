#!/usr/bin/env python3
"""Lecture 6: AI + two-channel relay on Jetson Nano.

Use with the existing SSD-MobileNet-v2 model (no training required):

    # Still image: save annotated result; blink one relay indicator twice
    python3 detectnet_gpio_assignment.py person.jpg person_result.jpg
    python3 detectnet_gpio_assignment.py no_person.jpg no_person_result.jpg

    # Live webcam: switch channel based on person presence
    python3 detectnet_gpio_assignment.py /dev/video0

WIRING (physical 40-pin header numbers; BOARD mode):
    Pin 11 -> relay IN1  (no person detected)
    Pin 13 -> relay IN2  (person detected)
    Header 5V -> relay VCC (as instructed by the course)
    Header GND -> relay GND

IMPORTANT:
    - Make connections with Jetson power completely OFF.
    - Disconnect all loads from relay COM/NO/NC terminals for this demo.
    - GPIO signals are 3.3V; ensure relay input module is compatible.
    - Most course examples use HIGH=active. If yours is active-low,
      pass --active-low after the paths.
"""

import argparse
import time
from pathlib import Path

try:
    from jetson_inference import detectNet
    from jetson_utils import (
        videoSource, videoOutput, loadImage, saveImage,
        cudaDeviceSynchronize,
    )
except ImportError:
    from jetson.inference import detectNet
    from jetson.utils import (
        videoSource, videoOutput, loadImage, saveImage,
        cudaDeviceSynchronize,
    )

import Jetson.GPIO as GPIO

# GPIO.BOARD means *physical* header pin numbers, not BCM numbers.
PIN_NO_PERSON = 11  # relay IN1, physical header pin 11 = BCM 17
PIN_PERSON = 13     # relay IN2, physical header pin 13 = BCM 27
PERSON_CLASS_ID = 1  # COCO person, same as teacher's slide


def set_relay(person_found, active, inactive):
    """Make exactly one relay channel active."""
    GPIO.output(PIN_NO_PERSON, inactive if person_found else active)
    GPIO.output(PIN_PERSON, active if person_found else inactive)


def turn_off_both(inactive):
    GPIO.output(PIN_NO_PERSON, inactive)
    GPIO.output(PIN_PERSON, inactive)


def blink_selected(person_found, active, inactive, repeats=2):
    pin = PIN_PERSON if person_found else PIN_NO_PERSON
    channel = "IN2 (PERSON)" if person_found else "IN1 (NO PERSON)"
    print("Blinking {} twice".format(channel))
    turn_off_both(inactive)
    for _ in range(repeats):
        GPIO.output(pin, active)
        time.sleep(0.5)
        GPIO.output(pin, inactive)
        time.sleep(0.5)


def person_count(detections):
    return sum(int(det.ClassID) == PERSON_CLASS_ID for det in detections)


def run_image(net, input_path, output_path, active, inactive):
    """Read a photo, save detection overlay, then pulse the relay twice."""
    img = loadImage(str(input_path))
    if img is None:
        raise RuntimeError("Could not read image: {}".format(input_path))

    detections = net.Detect(img, overlay="box,labels,conf")
    people = person_count(detections)
    cudaDeviceSynchronize()
    saveImage(str(output_path), img)

    print("Detected objects: {}".format(len(detections)))
    print("Detected persons: {}".format(people))
    print("Saved annotated image: {}".format(output_path))
    blink_selected(people > 0, active, inactive)


def run_camera(net, uri, active, inactive):
    """Use webcam frames to update a relay continuously."""
    camera = videoSource(uri)
    display = videoOutput("display://0")
    previous_state = None
    candidate_state = None
    candidate_frames = 0

    while display.IsStreaming():
        img = camera.Capture()
        if img is None:
            if not camera.IsStreaming():
                break
            continue

        detections = net.Detect(img, overlay="box,labels,conf")
        people = person_count(detections)
        found = people > 0

        # Three-frame confirmation helps avoid relay flickering on a noisy frame.
        if found == candidate_state:
            candidate_frames += 1
        else:
            candidate_state = found
            candidate_frames = 1

        if candidate_frames >= 3 and found != previous_state:
            set_relay(found, active, inactive)
            previous_state = found
            print("Person count: {} | {}".format(
                people, "IN2 ON (PERSON)" if found else "IN1 ON (NO PERSON)"
            ))

        display.Render(img)
        display.SetStatus("AI + GPIO | persons={} | {:.0f} FPS".format(
            people, net.GetNetworkFPS()
        ))
        if not camera.IsStreaming():
            break


def main():
    parser = argparse.ArgumentParser(
        description="SSD-MobileNet-v2 + Jetson Nano GPIO relay indicators"
    )
    parser.add_argument("input", nargs="?", default="/dev/video0",
                        help="a JPG/PNG image or camera URI (default /dev/video0)")
    parser.add_argument("output", nargs="?", default="result.jpg",
                        help="output JPG for image mode")
    parser.add_argument("--network", default="ssd-mobilenet-v2",
                        help="pretrained object detection model")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="person/object confidence threshold")
    parser.add_argument("--active-low", action="store_true",
                        help="use LOW to energize relay if board is active-low")
    args = parser.parse_args()

    active = GPIO.LOW if args.active_low else GPIO.HIGH
    inactive = GPIO.HIGH if args.active_low else GPIO.LOW

    input_path = Path(args.input).expanduser()
    is_image = input_path.suffix.lower() in (".jpg", ".jpeg", ".png")
    if is_image and not input_path.is_file():
        parser.error("image not found: {}".format(input_path))

    # No GPIO state is changed until after an explicit setup below.
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    try:
        GPIO.setup(PIN_NO_PERSON, GPIO.OUT, initial=inactive)
        GPIO.setup(PIN_PERSON, GPIO.OUT, initial=inactive)

        print("Loading network: {}".format(args.network))
        net = detectNet(args.network, threshold=args.threshold)
        if is_image:
            run_image(net, input_path, Path(args.output).expanduser(),
                      active, inactive)
        else:
            print("Webcam mode: person -> IN2; no person -> IN1; Ctrl+C to stop")
            run_camera(net, args.input, active, inactive)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        turn_off_both(inactive)
        GPIO.cleanup()
        print("GPIO cleaned up; both relay outputs released.")


if __name__ == "__main__":
    main()
