import time
from typing import List, Tuple, Dict, Any, Optional
from ..common.definitions import Detection

class PsezCropMatcher:
    def __init__(self, crop_class_id: int, psez_class_id: int, logger):
        if not isinstance(crop_class_id, int) or crop_class_id < 0:
             raise ValueError("crop_class_id must be a non-negative integer")
        if not isinstance(psez_class_id, int) or psez_class_id < 0:
             raise ValueError("psez_class_id must be a non-negative integer")

        self.crop_class_id = crop_class_id
        self.psez_class_id = psez_class_id
        self.logger = logger
        self.logger.info(f"[{self.__class__.__name__}] Initialized (PSEZ-Driven). Crop ID: {crop_class_id}, PSEZ ID: {psez_class_id}")

    def match(self, detections: List[Detection]) -> List[Detection]:
        start_time = time.monotonic()

        initial_crops: List[Detection] = []
        psezs: List[Detection] = []
        others: List[Detection] = []
        for d in detections:
            if d.class_id == self.crop_class_id:
                initial_crops.append(d)
            elif d.class_id == self.psez_class_id:
                psezs.append(d)
            else:
                others.append(d)

        #self.logger.debug(f"[{self.__class__.__name__}] Input counts - Crops: {len(initial_crops)}, PSEZs: {len(psezs)}, Others: {len(others)}")

        processed_detections: List[Detection] = list(others) # Start with non-crop/psez
        available_crops: List[Detection] = list(initial_crops)

        for psez in psezs:
            overlapping_available_crops: List[Tuple[Detection, float]] = [] 
            for crop in available_crops:
                if self._do_bboxes_overlap(psez.bbox, crop.bbox):
                     overlapping_available_crops.append(crop) 

            best_matching_crop: Optional[Detection] = None
            if overlapping_available_crops:
                # Find the best crop based on HIGHEST CROP confidence
                highest_crop_confidence = -1.0
                for crop in overlapping_available_crops:
                    if crop.confidence > highest_crop_confidence:
                        highest_crop_confidence = crop.confidence
                        best_matching_crop = crop
                #self.logger.debug(f"  > PSEZ (Conf {psez.confidence:.2f}) overlaps with {len(overlapping_available_crops)} available crops. Best crop conf: {highest_crop_confidence:.2f}")

            if best_matching_crop:
                combined_crop = self._create_combined_detection(best_matching_crop, psez)
                processed_detections.append(combined_crop)
                # CRITICAL: Remove the matched crop so it cannot be matched by another PSEZ
                try:
                    available_crops.remove(best_matching_crop)
                    #self.logger.debug(f"  > Matched PSEZ (Conf {psez.confidence:.2f}) with Crop (bbox {best_matching_crop.bbox}, Conf {best_matching_crop.confidence:.2f}). Crop removed from pool.")
                except ValueError:
                    # shouldnt happen if logic is sound
                    self.logger.error(f"  > Failed to remove already matched crop (bbox {best_matching_crop.bbox}) from available pool. This indicates a potential logic error.")

            else:
                synthetic_crop = self._create_synthetic_crop(psez)
                processed_detections.append(synthetic_crop)

        # Any crops left in available_crops were not matched by any PSEZ
        if available_crops:
             #self.logger.debug(f"  > Adding {len(available_crops)} unmatched original crops to results.")
             processed_detections.extend(available_crops)

        """
        match_time_ms = (time.monotonic() - start_time) * 1000
        final_combined = sum(1 for d in processed_detections if d.class_id == self.crop_class_id and d.associated_psez is not None and not d.is_synthetic)
        final_synthetic = sum(1 for d in processed_detections if d.class_id == self.crop_class_id and d.is_synthetic)
        final_unmatched_crops = sum(1 for d in processed_detections if d.class_id == self.crop_class_id and d.associated_psez is None and not d.is_synthetic)
        final_others = sum(1 for d in processed_detections if d.class_id != self.crop_class_id and d.class_id != self.psez_class_id) # Exclude PSEZ class too

        self.logger.info(
             f"[{self.__class__.__name__}] Matching finished in {match_time_ms:.1f} ms. "
             f"Output: {len(processed_detections)} detections "
             f"(Combined: {final_combined}, Synthetic: {final_synthetic}, UnmatchedCrops: {final_unmatched_crops}, Others: {final_others})"
        )
        """

        return processed_detections

    def _create_combined_detection(self, crop: Detection, psez: Detection) -> Detection:
        combined_detection = Detection(
            bbox=crop.bbox,                  # Use CROP's bbox
            class_id=self.crop_class_id,     # It's a CROP detection
            confidence=max(crop.confidence, psez.confidence), # Max confidence
            track_id=crop.track_id,          # Preserve track ID if crop was already tracked
            associated_psez=psez,            # Link to the PSEZ detection object
            is_synthetic=False               # This is based on a real crop
        )
        return combined_detection

    # should be moved to utils????
    def _do_bboxes_overlap(self, bbox1: Tuple[float, float, float, float],
                           bbox2: Tuple[float, float, float, float]) -> bool:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
            return False
        return True

    # rethink this
    def _create_synthetic_crop(self, psez: Detection) -> Detection:
        expand_factor = 0.5
        psez_w = psez.width
        psez_h = psez.height

        crop_x1 = psez.bbox[0] - psez_w * expand_factor
        crop_y1 = psez.bbox[1] - psez_h * expand_factor
        crop_x2 = psez.bbox[2] + psez_w * expand_factor
        crop_y2 = psez.bbox[3] + psez_h * expand_factor

        crop_x1 = max(0.0, crop_x1)
        crop_y1 = max(0.0, crop_y1)

        synthetic_crop = Detection(
            bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
            class_id=self.crop_class_id,     # Assign CROP class ID
            confidence=psez.confidence * 0.8, # Reduced confidence 
            track_id=None,                   # No track ID initially
            associated_psez=psez,            # Link to the PSEZ that generated it
            is_synthetic=True                # Mark as synthetic
        )
        return synthetic_crop