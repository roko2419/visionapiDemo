import os
from google.cloud import videointelligence_v1 as videointelligence

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'demo_key.json'


gs_URI = 'gs://demotwo/demo_vid.mp4'
gs_URI_two = 'gs://demotwo/demo_vid_2.mp4'
gs_URI_three = 'gs://demotwo/demo_vid_three.mp4'

def detect_faces(gcs_uri=""):

    client = videointelligence.VideoIntelligenceServiceClient()

    config = videointelligence.FaceDetectionConfig(
        include_bounding_boxes=True, include_attributes=True
    )
    context = videointelligence.VideoContext(face_detection_config=config)

    operation = client.annotate_video(
        request={
            "features": [videointelligence.Feature.FACE_DETECTION],
            "input_uri": gcs_uri,
            "video_context": context,
        }
    )

    print("\nProcessing video")
    result = operation.result(timeout=300)

    print("\nFinished processing.\n")

    annotation_result = result.annotation_results[0]

    for annotation in annotation_result.face_detection_annotations:
        print("Face detected:")
        for track in annotation.tracks:
            print(
                "Segment: {}s to {}s".format(
                    track.segment.start_time_offset.seconds
                    + track.segment.start_time_offset.microseconds / 1e6,
                    track.segment.end_time_offset.seconds
                    + track.segment.end_time_offset.microseconds / 1e6,
                )
            )

            timestamped_object = track.timestamped_objects[0]

            print("Attributes:")
            for attribute in timestamped_object.attributes:
                if attribute.name in ['looking_at_camera', 'smiling']:
                    print(
                        "\t{}:{} {}".format(
                            attribute.name, attribute.value, attribute.confidence
                        )
                    )

detect_faces(gs_URI_three)