from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    class_id: int                            # Class ID from the model (or adjusted by matcher)
    confidence: float                        # Confidence score for the detected class_id

    class_confidences: Dict[int, float] = field(default_factory=dict)

    track_id: Optional[int] = None           # Assigned by the object tracker
    associated_psez: Optional['Detection'] = field(default=None, repr=False) # Linked by PsezCropMatcher
    is_synthetic: bool = field(default=False, repr=False) # Marked by PsezCropMatcher if crop is synthetic

    @property
    def center(self) -> Tuple[float, float]:
        x_center = (self.bbox[0] + self.bbox[2]) / 2.0
        y_center = (self.bbox[1] + self.bbox[3]) / 2.0
        return (x_center, y_center)

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class TrackedObject:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    frames_missing: int = 0

    class_history: Dict[int, List[Tuple[float, bool]]] = field(default_factory=dict)
    last_vertical_displacement: float = 0.0
    last_horizontal_displacement: float = 0.0
    is_locked_to_crop: bool = False 
    predicted_position: Optional[Tuple[float, float]] = None

    @property
    def current_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None

    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        if self.frames_missing > 0 and self.predicted_position is not None:
            return self.predicted_position
        current_det = self.current_detection
        return current_det.center if current_det else None

    @property
    def current_center_x(self) -> Optional[float]:
        pos = self.current_position
        return pos[0] if pos else None

    @property
    def current_center_y(self) -> Optional[float]:
        pos = self.current_position
        return pos[1] if pos else None