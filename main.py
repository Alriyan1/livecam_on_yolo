import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([[0, 0],  [0.5, 0],[0.5, 1],[0, 1]])  # Define the polygon for the zone

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        '--webcam-resolution',
        default=[1280, 720],
        nargs=2,
        type=int,
    )
    args=parser.parse_args()
    return args


def main():
    args=parse_arguments()
    frame_width,frame_height = args.webcam_resolution

    cap= cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)  # Set the width of the frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)  # Set the height of the frame

    model=YOLO('yolov8n.pt')  # Load the YOLOv8 model

    box_annotator=sv.BoxAnnotator(
        thickness=2,

    )

    label_annotator=sv.LabelAnnotator(
        text_scale=1,
        text_thickness=2
    )

    zone_polygon=(ZONE_POLYGON*np.array(args.webcam_resolution)).astype(int)  # Scale the polygon to the webcam resolution
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2, text_scale=2, text_thickness=4)

    while True:
        ret,frame = cap.read()

        results=model(frame,agnostic_nms=True)[0]
        detections=sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id != 0]  # Filter out for class ID 0 (person)
        labels=[
            f'{model.model.names[class_id]} {confidence:0.2f}'
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame,detections=detections)
        frame = label_annotator.annotate(scene=frame,detections=detections,labels=labels)
        zone.trigger(detections=detections)  # Check if the zone is triggered
        frame = zone_annotator.annotate(scene=frame)
        cv2.imshow('yolov8',frame)  # Display the results

        
        if (cv2.waitKey(30)==27):
            break

if __name__ == "__main__":
    main()