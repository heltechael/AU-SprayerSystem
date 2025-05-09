# ENTRY POINT AND MAIN FILE

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterException

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from orei_cam_interfaces.msg import ImageWithMetadata
from vision_interfaces.msg import TrackedObject as TrackedObjectMsg
from vision_interfaces.msg import TrackingResults

from .processing.image_processor import ImageProcessor
from .inference.inference_engine import InferenceEngine
from .matching.psez_crop_matcher import PsezCropMatcher
from .motion_estimation import create_motion_estimator, BaseMotionEstimator
from .tracking.object_tracker import ObjectTracker

from .common.definitions import Detection, TrackedObject as TrackDataClass 

import os
import time
import threading
import numpy as np
import cv2
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, List, Optional, Tuple
from ultralytics.engine.results import Results

class VisionNode(Node):
    """
    ROS2 Node for processing images, performing motion estimation, detecting objects, matching PSEZ/Crops, tracking objects, visualizing results, and publishing tracking results
    """
    def __init__(self):
        super().__init__('vision_processing_node')
        self.get_logger().info(f'Initializing {self.get_name()}...')

        self.config: Dict[str, Any] = {}
        self._log_parallelism = False
        self._save_images = False
        self._save_path = ""

        # 1. Parameter loading 
        try:
            self._declare_parameters()
            self.config = self._load_parameters()
            debug_config = self.config.get('debug', {})
            self._log_parallelism = debug_config.get('log_parallelism_details', False)
            self._save_images = debug_config.get('save_annotated_images', False)
            self._save_path = debug_config.get('save_image_path', '')

            self.get_logger().info('Configuration parameters loaded successfully.')
            self._log_essential_parameters()
            self._ensure_save_directory()

        except ParameterException as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Failed during parameter declaration/loading: {e}\n{error_trace}")
            rclpy.try_shutdown()
            raise SystemExit(f"Parameter Error: {e}")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Unexpected error during parameter setup: {e}\n{error_trace}")
            rclpy.try_shutdown()
            raise SystemExit(f"Initialization Error: {e}")

        # 2. component initialization
        try:
            ip_config = self.config.get('image_processing', {})
            self.image_processor = ImageProcessor(ip_config, self.get_logger())
            self._initialize_motion_estimator()
            inf_config = self.config.get('inference', {})
            self.inference_engine = InferenceEngine(inf_config, self.get_logger())
            self._update_class_ids_from_model(self.config) 
            meta_config = self.config.get('model_metadata', {})
            self.psez_matcher = PsezCropMatcher(
                crop_class_id=meta_config.get('beet_class_id', -1),
                psez_class_id=meta_config.get('psez_class_id', -1),
                logger=self.get_logger()
            )
            track_config = self.config.get('tracking', {})
            img_dims = (ip_config.get('resize_width', 0), ip_config.get('resize_height', 0))
            if img_dims[0] == 0 or img_dims[1] == 0:
                 raise ValueError("Image dimensions are zero, check image_processing config.")
            self.object_tracker = ObjectTracker(
                config=track_config, image_dims=img_dims, logger=self.get_logger()
            )
            self.object_tracker.set_class_ids(
                crop_id=meta_config.get('beet_class_id', -1),
                psez_id=meta_config.get('psez_class_id', -1)
            )
            self.get_logger().info('Core components initialized.')
        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Failed to initialize core components: {e}\n{error_trace}")
            rclpy.try_shutdown()
            raise SystemExit(f"Component Initialization Error: {e}")


        # 3. ROS Communication Setup
        try:
            input_topic = self.config.get('input_topic', 'images')
            qos_profile = rclpy.qos.qos_profile_sensor_data
            self.subscription = self.create_subscription(
                ImageWithMetadata, input_topic, self.image_callback, qos_profile=qos_profile
            )
            self.get_logger().info(f"Subscribed to '{input_topic}' with QoS: {qos_profile.reliability.name}/{qos_profile.durability.name}") 

            output_topic = self.config.get('output_topic', '/vision_system/tracking_results') 
            self.publisher_ = self.create_publisher(TrackingResults, output_topic, 10) 
            self.get_logger().info(f"Will publish TrackingResults to '{output_topic}'.")

        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().fatal(f"Failed to create subscription or publisher: {e}\n{error_trace}")
            rclpy.try_shutdown()
            raise SystemExit(f"ROS Communication Setup Error: {e}")

        # 4. Threading/Processing setup
        self.processing_lock = threading.Lock()
        max_workers_config = self.config.get('threading', {}).get('max_workers', os.cpu_count() or 4)
        self.pipeline_executor = ThreadPoolExecutor(
            max_workers=max_workers_config,
            thread_name_prefix='pipeline_worker'
        )
        self.get_logger().info(f"Initialized ThreadPoolExecutor with max_workers={max_workers_config}.")


        self.get_logger().info(f'{self.get_name()} initialization complete.')

    # Parameter Handling (Declarations, Loading, Validation, Logging)
    def _declare_parameters(self):
        self.get_logger().debug('Declaring parameters...')

        self.declare_parameter('threading.max_workers', 8)

        self.declare_parameter('image_processing.resize_width', 2464)
        self.declare_parameter('image_processing.resize_height', 1408)

        self.declare_parameter('motion_estimation.estimator_type', 'orb_homography')

        self.declare_parameter('motion_estimation.orb_homography.num_features', 1000)
        self.declare_parameter('motion_estimation.orb_homography.knn_k', 2)
        self.declare_parameter('motion_estimation.orb_homography.lowe_ratio', 0.75)
        self.declare_parameter('motion_estimation.orb_homography.ransac_threshold', 5.0)
        self.declare_parameter('motion_estimation.orb_homography.min_good_matches', 10)

        self.declare_parameter('motion_estimation.sparse_lk.max_corners', 500)
        self.declare_parameter('motion_estimation.sparse_lk.quality_level', 0.05)
        self.declare_parameter('motion_estimation.sparse_lk.min_distance', 10)
        self.declare_parameter('motion_estimation.sparse_lk.block_size', 7)
        self.declare_parameter('motion_estimation.sparse_lk.lk_window_size', 21)
        self.declare_parameter('motion_estimation.sparse_lk.lk_pyramid_levels', 3)
        self.declare_parameter('motion_estimation.sparse_lk.lk_max_iterations', 50)
        self.declare_parameter('motion_estimation.sparse_lk.lk_epsilon', 0.01)
        self.declare_parameter('motion_estimation.sparse_lk.min_tracked_points', 50)

        self.declare_parameter('inference.model_path', Parameter.Type.STRING)
        self.declare_parameter('inference.device', 'cuda:0')
        self.declare_parameter('inference.confidence_threshold', 0.25)
        self.declare_parameter('inference.iou_threshold', 0.4)
        self.declare_parameter('inference.max_detections', 1500)
        self.declare_parameter('inference.use_half_precision', False)
        self.declare_parameter('inference.agnostic_nms', False)
        self.declare_parameter('inference.verbose', False)

        self.declare_parameter('model_metadata.beet_class_name', 'BEAVA')
        self.declare_parameter('model_metadata.psez_class_name', 'PSEZ')

        self.declare_parameter('tracking.enabled', True)
        self.declare_parameter('tracking.max_age', 5)
        self.declare_parameter('tracking.vertical_match_tolerance_px', 80.0)
        self.declare_parameter('tracking.horizontal_match_tolerance_px', 100.0)
        self.declare_parameter('tracking.validation_tolerance_factor', 1.5)
        self.declare_parameter('tracking.distance_penalty_threshold_factor', 2.0)
        self.declare_parameter('tracking.close_to_edge_threshold_px', 10)
        self.declare_parameter('tracking.bbox_touching_edge_weight', 0.25)
        self.declare_parameter('tracking.lock_beets_with_psez', True) 

        self.declare_parameter('tracking.weight_costs.position', 1.0)
        self.declare_parameter('tracking.weight_costs.area', 0.5)
        self.declare_parameter('tracking.weight_costs.aspect', 0.5)
        self.declare_parameter('tracking.weight_costs.class_consistency', 0.8) 
        self.declare_parameter('tracking.weight_costs.horizontal', 0.3)

        self.declare_parameter('tracking.min_confidence_for_new_track', 0.30)

        self.declare_parameter('input_topic', 'images')
        self.declare_parameter('output_topic', '/vision_system/tracking_results') 

        self.declare_parameter('debug.save_annotated_images', True)
        self.declare_parameter('debug.save_image_path', '/media/OREI-cam/yolo_debug_output')
        self.declare_parameter('debug.log_parallelism_details', False)

        self.declare_parameter('debug.visualization.draw_track_ids', True)
        self.declare_parameter('debug.visualization.draw_track_trails', True)
        self.declare_parameter('debug.visualization.trail_length', 15)
        self.declare_parameter('debug.visualization.trail_color', [255, 0, 255])
        self.declare_parameter('debug.visualization.trail_thickness', 2)
        self.declare_parameter('debug.visualization.track_id_color', [0, 255, 255])
        self.declare_parameter('debug.visualization.track_id_font_scale', 0.7)
        self.declare_parameter('debug.visualization.track_id_font_thickness', 2)
        self.declare_parameter('debug.visualization.track_id_offset_x', 5)
        self.declare_parameter('debug.visualization.track_id_offset_y', -10)
        self.get_logger().debug('Parameters declared.')

    def _load_parameters(self) -> dict:
        self.get_logger().debug('Loading parameters into config dictionary...')

        all_params = self.get_parameters_by_prefix('')
        config = {}
        for param_name, param_obj in all_params.items():
            keys = param_name.split('.')
            d = config
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = param_obj.value
        self.get_logger().debug('Raw parameters loaded.')

        # validation
        if not config.get('inference', {}).get('model_path'):
             raise ParameterException("Parameter 'inference.model_path' cannot be empty.")
        model_p = config.get('inference', {}).get('model_path')
        if not os.path.exists(model_p):
             self.get_logger().warning(f"Model path does not exist: {model_p}")
        if config.get('debug',{}).get('save_annotated_images') and not config.get('debug',{}).get('save_image_path'):
            raise ParameterException("Parameter 'debug.save_image_path' needed if 'debug.save_annotated_images' is true.")

        if 'model_metadata' not in config: config['model_metadata'] = {}
        meta = config['model_metadata']
        meta.setdefault('class_names', [])
        meta.setdefault('nc', 0)
        meta.setdefault('name_to_id', {})
        meta.setdefault('id_to_name', {})
        meta.setdefault('beet_class_id', -1) 
        meta.setdefault('psez_class_id', -1)
        
        # ensure crop/psez names are actually present
        if not meta.get('beet_class_name'): raise ParameterException("Parameter 'model_metadata.beet_class_name' required.")
        if not meta.get('psez_class_name'): raise ParameterException("Parameter 'model_metadata.psez_class_name' required.")

        self.get_logger().debug('Parameters loaded and basic validation passed.')
        return config

    def _log_essential_parameters(self):
        self.get_logger().info("--- Key Configuration")
        self.get_logger().info(f"  Input Topic: {self.config.get('input_topic', 'N/A')}")
        self.get_logger().info(f"  Output Topic: {self.config.get('output_topic', 'N/A')}")
        img_proc_cfg = self.config.get('image_processing', {})
        self.get_logger().info(f"  Image Size: {img_proc_cfg.get('resize_width')}x{img_proc_cfg.get('resize_height')}")
        motion_cfg = self.config.get('motion_estimation', {})
        self.get_logger().info(f"  Motion Estimator: {motion_cfg.get('estimator_type', 'N/A')}")
        inf_cfg = self.config.get('inference', {})
        self.get_logger().info(f"  Inference Device: {inf_cfg.get('device', 'N/A')}")
        self.get_logger().info(f"  Model Path: {inf_cfg.get('model_path', 'N/A')}")
        self.get_logger().info(f"  Confidence Thresh: {inf_cfg.get('confidence_threshold')}")
        meta_cfg = self.config.get('model_metadata', {})
        self.get_logger().info(f"  Crop Class Name: {meta_cfg.get('beet_class_name')}") 
        self.get_logger().info(f"  PSEZ Class Name: {meta_cfg.get('psez_class_name')}")
        track_cfg = self.config.get('tracking', {})
        self.get_logger().info(f"  Tracking Enabled: {track_cfg.get('enabled')}")
        self.get_logger().info(f"  Max Track Age: {track_cfg.get('max_age')}")
        self.get_logger().info(f"  Lock Crop/PSEZ: {track_cfg.get('lock_beets_with_psez')}")
        self.get_logger().info(f"  Save Annotated Images: {self._save_images}")
        self.get_logger().info(f"  Save Image Path: {self._save_path if self._save_images else 'N/A'}")
        self.get_logger().info(f"  Log Parallelism Details: {self._log_parallelism}")
        self.get_logger().info("-------------------------")

    def _initialize_motion_estimator(self):
        """Initializes the motion estimator using the factory."""
        motion_config = self.config.get('motion_estimation', {})
        estimator_type = motion_config.get('estimator_type', 'orb_homography')
        specific_config = motion_config.get(estimator_type, {})
        if not specific_config:
             self.get_logger().warning(f"Config 'motion_estimation.{estimator_type}' not found. Estimator might use defaults.")

        self.motion_estimator: Optional[BaseMotionEstimator] = create_motion_estimator(
            estimator_type=estimator_type,
            config=specific_config, 
            logger=self.get_logger()
        )
        if self.motion_estimator is None:
            self.get_logger().error(f"Motion estimation type '{estimator_type}' is configured but failed to initialize. Tracking quality will be degraded.")
            self.get_logger().error("Proceeding without motion estimation.")

    def _update_class_ids_from_model(self, config: Dict[str, Any]):
        if not hasattr(self, 'inference_engine') or not self.inference_engine._model_names:
            self.get_logger().error("Cannot update class IDs: InferenceEngine/model names not ready.")
            raise RuntimeError("Model names required for class ID mapping.")

        model_id_to_name = self.inference_engine._model_names
        model_name_to_id = {name: id for id, name in model_id_to_name.items()}
        class_names_list = [model_id_to_name[i] for i in sorted(model_id_to_name.keys())]

        meta = config.setdefault('model_metadata', {})
        meta['class_names'] = class_names_list
        meta['nc'] = len(class_names_list)
        meta['name_to_id'] = model_name_to_id
        meta['id_to_name'] = model_id_to_name

        crop_name = meta.get('beet_class_name')
        psez_name = meta.get('psez_class_name')

        meta['beet_class_id'] = model_name_to_id.get(crop_name, -1)
        meta['psez_class_id'] = model_name_to_id.get(psez_name, -1)

        if meta['beet_class_id'] == -1:
            raise ParameterException(f"Crop class '{crop_name}' (from 'beet_class_name' param) not found in model: {list(model_name_to_id.keys())}")
        if meta['psez_class_id'] == -1:
            self.get_logger().warning(f"PSEZ class '{psez_name}' not found in model: {list(model_name_to_id.keys())}. PSEZ matching will not function.")
        elif crop_name == psez_name:
            self.get_logger().warning(f"Crop class name and PSEZ class name are identical ('{crop_name}'). Check config.")

        self.get_logger().info(f"Updated Class IDs - Crop ('{crop_name}'): {meta['beet_class_id']}, PSEZ ('{psez_name}'): {meta['psez_class_id']}")

    def _ensure_save_directory(self):
        if self._save_images and self._save_path:
            try:
                os.makedirs(self._save_path, exist_ok=True)
                self.get_logger().info(f"Ensured save directory exists: {self._save_path}")
            except OSError as e:
                self.get_logger().error(f"Failed to create save directory '{self._save_path}': {e}. Disabling image saving.")
                self._save_images = False
        elif self._save_images and not self._save_path:
            self.get_logger().error("Image saving enabled but save path is empty. Disabling image saving.")
            self._save_images = False

    # lightweight img callback
    def image_callback(self, msg: ImageWithMetadata):

        # LOG HEADER
        #self.get_logger().info(f"msg.header: {msg.header}")
        #self.get_logger().info(f"msg.image.header: {msg.image.header}")
          
        image_num_str = f"{msg.image_number:06d}"
        if self.processing_lock.acquire(blocking=False):
            try:
                submit_time = self.get_clock().now()
                #self.get_logger().info(f'[Img {image_num_str}] Submitting for processing | Msg TS: {msg.header.stamp.sec}.{msg.header.stamp.nanosec} | Submit TS: {submit_time.nanoseconds}')
                future = self.pipeline_executor.submit(self._process_pipeline_async, msg, submit_time)
            except Exception as submit_e:
                self.get_logger().error(f"[Img {image_num_str}] Error submitting pipeline task: {submit_e}\n{traceback.format_exc()}")
                self.processing_lock.release()
        else:
            self.get_logger().warn(f'[Img {image_num_str}] Pipeline busy, skipping frame.')

    # Async pipeline processing
    def _process_pipeline_async(self, msg: ImageWithMetadata, submit_time_ros):
        image_num_str = f"{msg.image_number:06d}"
        pipeline_start_mono = time.monotonic()
        msg_stamp_sec = msg.header.stamp.sec
        msg_stamp_nanosec = msg.header.stamp.nanosec
        
        #self.get_logger().warning(f"msg.header.stamp.sec: {msg.header.stamp.sec}, msg.header.stamp.nanosec: {msg.header.stamp.nanosec}")
        
        submit_stamp_sec, submit_stamp_nanosec = submit_time_ros.seconds_nanoseconds()

        # Approximate entry latency (difference between submit time and worker start time)
        entry_latency_ns = (pipeline_start_mono * 1e9) - (submit_stamp_sec * 1e9 + submit_stamp_nanosec)
        entry_latency_ms = entry_latency_ns / 1e6

        processed_image: Optional[np.ndarray] = None
        estimated_dx: Optional[float] = None
        estimated_dy: Optional[float] = None
        raw_detections: List[Detection] = []
        raw_results: Optional[Results] = None
        matched_detections: List[Detection] = []
        final_tracked_objects: List[TrackDataClass] = [] # Use alias for clarity

        proc_time, motion_exec_time, infer_exec_time = 0, 0, 0
        parallel_wait_time, match_time, track_time, pub_time, save_time = 0, 0, 0, 0, 0

        try:
            # 1. image processing
            
            #self.get_logger().warning(f"msg.header.stamp.nanosec: {msg.header.stamp.nanosec}")

            start_proc_time = time.monotonic()
            processed_image = self.image_processor.process_image(msg.image)
            proc_time = (time.monotonic() - start_proc_time) * 1000
            if processed_image is None:
                self.get_logger().error(f"[Img {image_num_str}] Image processing failed.")
                return # exit pipe early

            #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: ImgProc Done ({proc_time:.1f}ms)")

            # 2. submit prallel tasks (motion Est. and inference)
            motion_future: Optional[Future] = None
            inference_future: Optional[Future] = None

            if self.motion_estimator:
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Submitting Motion Est...")
                start_submit_time = time.monotonic()
                # pass a copy if estimator might modify, otherwise reference is fine
                motion_future = self.pipeline_executor.submit(
                    self.motion_estimator.estimate_displacement, processed_image
                )
                submit_duration = (time.monotonic() - start_submit_time) * 1000
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Motion Submitted ({submit_duration:.1f}ms)")
            #else:
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Skipping Motion Est submission (disabled).")


            if self.inference_engine:
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Submitting Inference...")
                start_submit_time = time.monotonic()
                inference_future = self.pipeline_executor.submit(
                    self.inference_engine.run_inference,
                    processed_image=processed_image,
                    image_identifier=image_num_str
                )
                submit_duration = (time.monotonic() - start_submit_time) * 1000
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Inference Submitted ({submit_duration:.1f}ms)")

            # 3. wait for parallel tasks and get results
            #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Waiting for parallel tasks...")
            start_parallel_wait_time = time.monotonic()

            # Wait for motion est. (if submitted)
            if motion_future:
                start_motion_wait = time.monotonic()
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: --> Waiting for Motion result...")
                try:
                    motion_result = motion_future.result() # Timeout?
                    estimated_dx, estimated_dy = motion_result if motion_result else (None, None)
                    motion_exec_time = (time.monotonic() - start_motion_wait) * 1000
                    #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: <-- Motion DONE (Wait: {motion_exec_time:.1f}ms, dx={estimated_dx}, dy={estimated_dy})")
                except Exception as motion_e:
                    motion_exec_time = (time.monotonic() - start_motion_wait) * 1000
                    self.get_logger().error(f"[Img {image_num_str}] Worker: <-- Motion FAILED (Wait: {motion_exec_time:.1f}ms): {motion_e}\n{traceback.format_exc()}")
                    estimated_dx, estimated_dy = None, None
            else:
                 estimated_dx, estimated_dy = 0.0, 0.0 # Use zero if motion estimation is disabled
                 #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Using default motion (0,0) as estimator is disabled.")

            #self.get_logger().warning(f"estimated_dx: {estimated_dx}, estimated_dy: {estimated_dy}")

            # wait for inference (if submitted)
            if inference_future:
                start_infer_wait = time.monotonic()
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: --> Waiting for Inference result...")
                try:
                    infer_result = inference_future.result() # Timeout?
                    raw_detections, raw_results = infer_result if infer_result else ([], None)
                    infer_exec_time = (time.monotonic() - start_infer_wait) * 1000
                    #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: <-- Inference DONE (Wait: {infer_exec_time:.1f}ms, Dets: {len(raw_detections)})")
                except Exception as infer_e:
                    infer_exec_time = (time.monotonic() - start_infer_wait) * 1000
                    self.get_logger().error(f"[Img {image_num_str}] Worker: <-- Inference FAILED (Wait: {infer_exec_time:.1f}ms): {infer_e}\n{traceback.format_exc()}")
                    raw_detections, raw_results = [], None
            else:
                 raw_detections, raw_results = [], None

            parallel_wait_time = (time.monotonic() - start_parallel_wait_time) * 1000
            #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Parallel tasks finished (Wait time: {parallel_wait_time:.1f}ms)")

            # 4. PSEZ/crop matching
            if self.psez_matcher:
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Starting PSEZ/Crop Matching...")
                start_match_time = time.monotonic()
                matched_detections = self.psez_matcher.match(raw_detections)
                match_time = (time.monotonic() - start_match_time) * 1000
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Matching Done ({match_time:.1f}ms, Output Dets: {len(matched_detections)})")
            else:
                matched_detections = raw_detections

            # 5. object tracking
            if self.object_tracker:
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Starting Object Tracking...")
                start_track_time = time.monotonic()
                # Ensure motion estimates are float or 0.0
                current_dx = float(estimated_dx) if estimated_dx is not None else 0.0
                current_dy = float(estimated_dy) if estimated_dy is not None else 0.0
                final_tracked_objects = self.object_tracker.update(
                    matched_detections, current_dx, current_dy
                )
                track_time = (time.monotonic() - start_track_time) * 1000
                #if self._log_parallelism: self.get_logger().info(f"[Img {image_num_str}] Worker: Tracking Done ({track_time:.1f}ms, Tracks: {len(final_tracked_objects)})")
            else:
                 final_tracked_objects = []

            # 6. Save annotated DEBUG image
            start_save_time = time.monotonic()
            if self._save_images and processed_image is not None:
                self._save_annotated_image(
                    base_image=processed_image.copy(), # Draw on a copy
                    tracks=final_tracked_objects,
                    detections_for_bbox=matched_detections,
                    image_identifier=image_num_str,
                    estimated_dx=estimated_dx,
                    estimated_dy=estimated_dy
                )
            save_time = (time.monotonic() - start_save_time) * 1000

            # 7. convert and pub
            start_pub_time = time.monotonic()

            if self.publisher_:
                msg_dx = float(estimated_dx) if estimated_dx is not None else 0.0
                msg_dy = float(estimated_dy) if estimated_dy is not None else 0.0

                output_header = Header()
                output_stamp_msg = None 

                #self.get_logger().warning(f"{msg.gpsfix}")

                #self.get_logger().warning(f"msg.header.stamp.sec: {msg.header.stamp.sec}")
                
                if msg.gpsfix.time > 0: 
                    try:
                        rclpy_time_from_gps = rclpy.time.Time(seconds=msg.gpsfix.time)
                        output_stamp_msg = rclpy_time_from_gps.to_msg()
                        # self.get_logger().debug(f"[Img {image_num_str}] Using valid GPSFix time for output header.")
                    except Exception as e:
                        self.get_logger().error(f"[Img {image_num_str}] Error converting valid GPS time {msg.gpsfix.time} to stamp: {e}")
                        output_stamp_msg = None 
                else:
                    self.get_logger().warning(f"[Img {image_num_str}] Invalid gpsfix.time ({msg.gpsfix.time}).")
                    output_stamp_msg = None

                if output_stamp_msg is None:
                    if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
                        output_stamp_msg = msg.header.stamp
                        self.get_logger().warning(f"[Img {image_num_str}] Using non-zero input header stamp as fallback.")
                    else:
                        output_stamp_msg = self.get_clock().now().to_msg()
                        self.get_logger().error(f"[Img {image_num_str}] CRITICAL: Both GPSFix time and input header stamp are invalid/zero! Using current time.")

                output_header.stamp = output_stamp_msg
                output_header.frame_id = msg.header.frame_id 

                results_msg = self._create_tracking_results_msg(
                    final_tracked_objects,
                    output_header,
                    msg_dx,
                    msg_dy
                )
                self.publisher_.publish(results_msg)

            

            pub_time = (time.monotonic() - start_pub_time) * 1000
            final_output_count = len(final_tracked_objects)

            total_pipeline_time = (time.monotonic() - pipeline_start_mono) * 1000

            # OUTPUT FOR DEBUGGING
            timing_parts = [
                f'Total: {total_pipeline_time:.1f}ms',
                f'EntryLat: {entry_latency_ms:.1f}ms',
                f'Proc: {proc_time:.1f}ms',
                f'MotionWait: {motion_exec_time:.1f}ms',
                f'InferWait: {infer_exec_time:.1f}ms ({len(raw_detections)} raw)',
                f'Match: {match_time:.1f}ms ({len(matched_detections)} matched)',
                f'Track: {track_time:.1f}ms ({final_output_count} tracked)',
                f'SaveImg: {save_time:.1f}ms',
                f'Pub: {pub_time:.1f}ms'
            ]
            self.get_logger().info(f'[Img {image_num_str}] Pipeline Finished | ' + ' | '.join(timing_parts))
            

        except Exception as e:
            error_trace = traceback.format_exc()
            self.get_logger().error(f"[Img {image_num_str}] Worker: Unhandled error in pipeline: {e}\n{error_trace}")
        finally:
            if self.processing_lock.locked():
                 self.processing_lock.release()
                 if self._log_parallelism: self.get_logger().debug(f"[Img {image_num_str}] Worker: Processing lock released.")

    def _create_tracking_results_msg(self,
                                    tracks: List[TrackDataClass],
                                    header: Header,
                                    estimated_dx: float,
                                    estimated_dy: float) -> TrackingResults:
        results_msg = TrackingResults()
        results_msg.header = header 
        results_msg.estimated_motion_dx = estimated_dx
        results_msg.estimated_motion_dy = estimated_dy

        meta_config = self.config.get('model_metadata', {})
        id_to_name_map = meta_config.get('id_to_name', {})

        for track_data in tracks:
            if track_data.frames_missing == 0 and track_data.current_detection:
                current_det = track_data.current_detection
                track_msg = TrackedObjectMsg()

                track_msg.track_id = track_data.track_id
                predicted_class_id, predicted_confidence = self.object_tracker.get_predicted_class(track_data)

                if predicted_class_id == -1:
                    self.get_logger().warning(f"Track {track_data.track_id} has no valid predicted class. Skipping population.")
                    continue

                track_msg.class_id = predicted_class_id
                track_msg.class_name = id_to_name_map.get(predicted_class_id, f"UNKNOWN_ID_{predicted_class_id}")
                track_msg.confidence = float(predicted_confidence)

                bbox = current_det.bbox
                track_msg.bbox_x = float(bbox[0])
                track_msg.bbox_y = float(bbox[1])
                track_msg.bbox_width = float(bbox[2] - bbox[0])
                track_msg.bbox_height = float(bbox[3] - bbox[1])

                center = current_det.center
                track_msg.center_x = float(center[0])
                track_msg.center_y = float(center[1])

                track_msg.frames_missing = track_data.frames_missing
                track_msg.class_history_len = sum(len(value) for value in track_data.class_history.values())

                results_msg.tracked_objects.append(track_msg)

        return results_msg

    def _save_annotated_image(self,
                              base_image: np.ndarray,
                              tracks: List[TrackDataClass], 
                              detections_for_bbox: List[Detection],
                              image_identifier: str,
                              estimated_dx: Optional[float],
                              estimated_dy: Optional[float]):
        if not self._save_images or not self._save_path or base_image is None:
            return

        start_save_time = time.monotonic()
        try:
            img_h, img_w = base_image.shape[:2]

            # Draw Global Motion Vector
            motion_text_color = (0, 255, 255) # Cyan
            motion_text_scale = 0.6
            motion_text_thickness = 1
            motion_text_y_pos = img_h - 10
            if estimated_dx is not None and estimated_dy is not None:
                center_x, center_y = img_w // 2, img_h // 2
                arrow_color = (0, 255, 255) # Cyan
                cv2.arrowedLine(base_image, (center_x, center_y), (int(center_x + estimated_dx), int(center_y + estimated_dy)), arrow_color, 1, tipLength=0.1)
                motion_text_top = f"MotionEst: dx={estimated_dx:.1f}, dy={estimated_dy:.1f} px"
                cv2.putText(base_image, motion_text_top, (10, motion_text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, motion_text_scale, motion_text_color, motion_text_thickness, cv2.LINE_AA)
            else:
                cv2.putText(base_image, "MotionEst: N/A", (10, motion_text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, motion_text_scale, (0, 0, 255), motion_text_thickness, cv2.LINE_AA) # Red text

            # Draw Tracks / Detections
            vis_config = self.config.get('debug', {}).get('visualization', {})
            draw_ids = vis_config.get('draw_track_ids', True)
            draw_trails = vis_config.get('draw_track_trails', True)
            trail_len = vis_config.get('trail_length', 15)
            trail_color = tuple(vis_config.get('trail_color', [255, 0, 255]))
            trail_thickness = vis_config.get('trail_thickness', 2)
            id_color = tuple(vis_config.get('track_id_color', [0, 255, 255]))
            id_scale = vis_config.get('track_id_font_scale', 0.7)
            id_thick = vis_config.get('track_id_font_thickness', 2)
            id_offset_x = vis_config.get('track_id_offset_x', 5)
            id_offset_y = vis_config.get('track_id_offset_y', -10)

            if tracks:
                for track in tracks:
                    if track.frames_missing == 0 and track.current_detection:
                        det = track.current_detection
                        bbox = det.bbox
                        x1, y1, x2, y2 = map(int, bbox)

                        pred_class_id, pred_conf = self.object_tracker.get_predicted_class(track)
                        # Check if prediction is valid before getting name
                        if pred_class_id != -1:
                            class_name = self.inference_engine.get_class_name(pred_class_id)
                            label = f"{track.track_id}:{class_name}:{pred_conf:.2f}"
                        else:
                            label = f"{track.track_id}:INVALID_CLS:{pred_conf:.2f}"

                        color_seed = track.track_id * 30
                        color = (color_seed % 255, (color_seed * 2) % 255, (color_seed * 3) % 255)

                        cv2.rectangle(base_image, (x1, y1), (x2, y2), color, id_thick)

                        if draw_ids:
                            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, id_scale, id_thick)
                            label_pos = (x1 + id_offset_x, y1 + id_offset_y)
                            cv2.putText(base_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, id_scale, id_color, id_thick)

                            dx = track.last_horizontal_displacement
                            dy = track.last_vertical_displacement
                            displacement_text = f"dx: {dx:.1f}, dy: {dy:.1f}" 
                            disp_color = (255, 255, 0) # Yellow
                            disp_pos_y = label_pos[1] + label_height + 5
                            disp_pos = (label_pos[0], disp_pos_y)
                            cv2.putText(base_image, displacement_text, disp_pos, cv2.FONT_HERSHEY_SIMPLEX, id_scale * 0.8, disp_color, id_thick)

                        if draw_trails and len(track.detections) > 1:
                            points = np.array([d.center for d in track.detections[-trail_len:]], dtype=np.int32)
                            if points.size > 0:
                                cv2.polylines(base_image, [points], isClosed=False, color=trail_color, thickness=trail_thickness)

            elif detections_for_bbox: 
                 for det in detections_for_bbox:
                      bbox = det.bbox
                      x1, y1, x2, y2 = map(int, bbox)
                      class_name = self.inference_engine.get_class_name(det.class_id)
                      label = f"Det:{class_name}:{det.confidence:.2f}"
                      color_seed = det.class_id * 50
                      color = ((color_seed * 3) % 255, color_seed % 255, (color_seed * 2) % 255)
                      cv2.rectangle(base_image, (x1, y1), (x2, y2), color, 1)
                      cv2.putText(base_image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save img
            filename = f"{image_identifier}_tracked_annotated.jpg"
            filepath = os.path.join(self._save_path, filename)

            success = cv2.imwrite(filepath, base_image)
            save_time_ms = (time.monotonic() - start_save_time) * 1000

            if success:
                self.get_logger().debug(f"Saved annotated image to {filepath} ({save_time_ms:.1f}ms)")
            else:
                self.get_logger().error(f"Failed to save annotated image to {filepath} using cv2.imwrite")
        except Exception as plot_save_e:
            error_trace = traceback.format_exc()
            self.get_logger().error(f"Error during image annotation/saving: {plot_save_e}\n{error_trace}")


    def on_shutdown(self):
        """Cleans up resources on node shutdown."""
        self.get_logger().info("Shutting down pipeline executor...")
        self.pipeline_executor.shutdown(wait=True)
        self.get_logger().info("Pipeline executor shut down.")
        if hasattr(self.motion_estimator, 'cleanup'):
            self.motion_estimator.cleanup()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        log_func = node.get_logger().info if node else print
        log_func('Keyboard interrupt received, shutting down.')
    except SystemExit as e:
        print(f"Node exited during initialization or processing: {e}")
    except Exception as e:
        error_trace = traceback.format_exc()
        log_func = node.get_logger().fatal if node else print
        log_func(f"Unhandled exception during spin: {e}\n{error_trace}")
    finally:
        if node:
            try:
                node.on_shutdown()
            except Exception as shutdown_e:
                 print(f"Error during node shutdown method: {shutdown_e}\n{traceback.format_exc()}")
            finally:
                if rclpy.ok():
                    if node.context and node.context.ok():
                         node.destroy_node()
                         print("Vision node destroyed.")
                    else:
                         print("Node context invalid, cannot destroy node.")
                    rclpy.shutdown()
                    print("ROS2 shutdown complete.")
                else:
                     print("ROS2 context invalid before final shutdown sequence.")

if __name__ == '__main__':
    main()