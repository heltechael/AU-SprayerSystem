import time
import traceback
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Dict, Any, Optional, Tuple

from ..common.definitions import Detection

class InferenceEngine:
    def __init__(self, config: Dict[str, Any], logger): 
        self.logger = logger
        self._config = config
        self._model_path = self._config.get('model_path')
        self._device = self._config.get('device', 'cuda:0')
        self._imgsz_height = self._config.get('resize_height', 1600)
        self._imgsz_width = self._config.get('resize_width', 2464)
        self._conf_threshold = self._config.get('confidence_threshold', 0.25)
        self._iou_threshold = self._config.get('iou_threshold', 0.4)
        self._max_detections = self._config.get('max_detections', 2000)
        self._use_half = self._config.get('use_half_precision', False)
        self._agnostic_nms = self._config.get('agnostic_nms', False)
        self._verbose = self._config.get('verbose', False) # Use config verbose flag
        self.is_engine = True
 
        if not self._model_path:
            self.logger.fatal("[InferenceEngine] 'model_path' not specified in config.")
            raise ValueError("Model path is required for InferenceEngine.")

        try:
            self._model = self._load_model()
            self._model_names = self._model.names if hasattr(self._model, 'names') else {}
            self.logger.info(f"[InferenceEngine] Model loaded successfully. Found {len(self._model_names)} classes.")
            self.logger.debug(f"  Model Class Names: {self._model_names}")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.fatal(f"[InferenceEngine] Failed to load model from {self._model_path}: {e}\n{error_trace}")
            raise

        self.is_engine = self._model_path.lower().endswith('.engine')
        self.logger.info(f"[InferenceEngine] Detected model type: {'TensorRT Engine' if self.is_engine else 'PyTorch Model'}")

    def _load_model(self) -> YOLO:
        start_load_time = time.monotonic()
        model = YOLO(self._model_path, task='detect') # ensure task is 'detect' and not 'segment'
        
        if not self.is_engine:
            model.to(self._device)
            self.logger.info(f"[InferenceEngine] Moved PyTorch model to device: {self._device}")
        else:
             self.logger.info(f"[InferenceEngine] TensorRT engine loaded. Device assignment handled by Ultralytics/TensorRT.")
        
        load_time = (time.monotonic() - start_load_time) * 1000
        self.logger.info(f"[InferenceEngine] Model loading took {load_time:.2f} ms.")
        return model

    def run_inference(self, processed_image: np.ndarray, image_identifier: str = "image") -> Tuple[List[Detection], Optional[Results]]:
        if self._model is None:
            self.logger.error("[InferenceEngine] Model not loaded. Cannot run inference.")
            return [], None

        start_inf_time = time.monotonic()
        results: Optional[Results] = None
        detections: List[Detection] = []

        #self.logger.warn(f"[InferenceEngine] PROCESSED IMG: {processed_image.shape}")
        #self.logger.warn(f"[InferenceEngine] _imgsz_height: {self._imgsz_height}, _imgsz_width: {self._imgsz_width}")

        try:
            results_list = self._model.predict(
                source=processed_image,
                # Provide exact image size tuple (H, W) expected by engine/model - yes you read rihgt: (height,width) and NOT (width,height)
                imgsz=(self._imgsz_height, self._imgsz_width),
                device=self._device, #have to psas device directly for engine
                half=self._use_half, # Pass half precision flag, should match engine build (True for FP16)
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                max_det=self._max_detections,
                agnostic_nms=self._agnostic_nms,
                verbose=self._verbose # ultralytics output - useful when debugging
            )

            results = results_list[0] if results_list else None
            inf_time_ms = (time.monotonic() - start_inf_time) * 1000

            if results:
                log_string = results.verbose().strip()
                self.logger.info(f"[InferenceEngine] {log_string} ({inf_time_ms:.1f}ms)")
                detections = self._parse_results_to_detections(results)
                self.logger.debug(f"[InferenceEngine] Processed {len(detections)} detections.")
            else:
                 self.logger.info(f"[InferenceEngine] No results returned by model. ({inf_time_ms:.1f}ms)")

        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"[InferenceEngine] Error during prediction: {e}\n{error_trace}")
            return [], None 

        total_time = (time.monotonic() - start_inf_time) * 1000
        self.logger.debug(f"[InferenceEngine] Inference step finished. Total time: {total_time:.1f} ms")
        return detections, results

    def _parse_results_to_detections(self, results: Results) -> List[Detection]:
        """Parses YOLO detection results into a list of Detection objects, including class confidences."""
        detections = []
        if results.boxes is None:
            self.logger.warning("[InferenceEngine] results.boxes is None. No detections to parse.")
            return detections

        try:
            boxes = results.boxes.cpu().numpy() 
            probs_data = None
            if results.probs is not None:
                 probs_data = results.probs.cpu().numpy().data # Get probabilities tensor data
                 if len(probs_data) != len(boxes.data):
                     self.logger.warning(f"Number of probability vectors ({len(probs_data)}) doesn't match number of boxes ({len(boxes.data)}). Class confidences might be incorrect.")
                     probs_data = None # Disable if mismatch

            for i, box_data in enumerate(boxes.data):
                if len(box_data) < 6:
                     self.logger.warning(f"Skipping invalid box data (length {len(box_data)}): {box_data}")
                     continue

                x_min, y_min, x_max, y_max = map(float, box_data[:4])
                confidence = float(box_data[4]) 
                class_id = int(box_data[5])     

                if confidence < 0.0 or confidence > 1.0:
                    self.logger.warning(f"Skipping detection with invalid confidence {confidence}")
                    continue
                if class_id < 0 or class_id >= len(self._model_names):
                    self.logger.warning(f"Skipping detection with invalid class ID {class_id}")
                    continue

                class_confidences_dict: Dict[int, float] = {}
                if probs_data is not None and i < len(probs_data):
                    prob_vector = probs_data[i]
                    if len(prob_vector) == len(self._model_names): 
                         for c_id, c_name in self._model_names.items():
                             class_confidences_dict[c_id] = float(prob_vector[c_id])
                    else:
                         self.logger.warning(f"Probability vector length ({len(prob_vector)}) doesn't match model class count ({len(self._model_names)}) for box {i}. Falling back.")
                         class_confidences_dict[class_id] = confidence
                else:
                    class_confidences_dict[class_id] = confidence

                detection = Detection(
                    bbox=(x_min, y_min, x_max, y_max),
                    class_id=class_id,
                    confidence=confidence, 
                    class_confidences=class_confidences_dict 
                )
                detections.append(detection)

        except Exception as parse_e:
             self.logger.error(f"[InferenceEngine] Error parsing results: {parse_e}\n{traceback.format_exc()}")
             return [] 

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Gets the class name string from the ID using the loaded model's names."""
        return self._model_names.get(class_id, f"ID:{class_id}")