"""Recognize faces from the default webcam using Sagar.jpeg as the reference."""

from pathlib import Path

import cv2
import face_recognition
import numpy as np


def main():
    reference_path = Path(__file__).with_name("Sagar.jpeg")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    reference = face_recognition.load_image_file(str(reference_path))
    encodings = face_recognition.face_encodings(reference)
    if not encodings:
        raise ValueError(f"No face found in reference image: {reference_path.name}")
    known_encoding = encodings[0]

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera 0. Check that a webcam is connected and available.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            small = cv2.resize(frame, None, fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for location, encoding in zip(locations, encodings):
                top, right, bottom, left = (coordinate * 4 for coordinate in location)
                distance = face_recognition.face_distance([known_encoding], encoding)[0]
                name = "Sagar" if distance < 0.6 else "Unknown"
                color = (0, 200, 0) if name == "Sagar" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
