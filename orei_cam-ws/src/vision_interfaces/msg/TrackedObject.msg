int32 track_id              # Unique ID for each track
int32 class_id              # Predicted class ID for the track (most likely based on history)
string class_name           # Predicted class name string
float32 confidence          # Confidence score of the predicted class
int32 frames_missing        # Number of consecutive frames the track has been missing
int32 class_history_len

# Current Bounding Box (image coordinates, pixels)
float32 bbox_x          # Top-left x
float32 bbox_y          # Top-left y
float32 bbox_width      # Width of the box
float32 bbox_height     # Height of the box

# Current Estimated Center Position (image coordinates, pixels)
float32 center_x
float32 center_y