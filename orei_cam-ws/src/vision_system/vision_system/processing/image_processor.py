import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time 
import traceback 

class ImageProcessor:
    _DEMOSAIC_CODE = cv2.COLOR_BayerBG2RGB
    _GAMMA_INVERSE = 1 / 1.7
    _GAMMA_FORWARD = 2.8
    _COLOR_MATRIX = np.array([
        [ 1.54526611, -0.44373111,  0.02583314],
        [-0.59687119,  1.26119455, -0.35844165],
        [-0.14311284, -0.23684594,  0.90426895]
    ], dtype=np.float64)

    def __init__(self, config: dict, logger):
        self._bridge = CvBridge()
        self.logger = logger
        self._target_width = config.get('resize_width', 4160)
        self._target_height = config.get('resize_height', 2368)

        if not isinstance(self._target_width, int) or self._target_width <= 0:
             raise ValueError("Invalid 'resize_width' in config")
        if not isinstance(self._target_height, int) or self._target_height <= 0:
             raise ValueError("Invalid 'resize_height' in config")

        try:
            self._gamma_lut_inv = self._create_gamma_lut(self._GAMMA_INVERSE)
            self._gamma_lut_fwd = self._create_gamma_lut(self._GAMMA_FORWARD)
            self.logger.info("Gamma LUTs created successfully.")
        except Exception as e:
            self.logger.error(f"[ImageProcessor] Error creating LUTs: {e}")
            raise

        self.logger.info(f"ImageProcessor initialized: Target size {self._target_width}x{self._target_height}")
        self.logger.info(f"  Using Demosaic Code: {self._DEMOSAIC_CODE}")


    def _create_gamma_lut(self, gamma_value: float) -> np.ndarray:
        if gamma_value <= 0:
            raise ValueError("Gamma value must be positive.")
        inv_gamma = 1.0 / gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)], dtype=np.uint8)
        return table

    def _apply_gamma(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            self.logger.warning("[ImageProcessor] Input to _apply_gamma was not uint8, converting.")
            image = image.astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
             return cv2.LUT(image, lut)
        elif len(image.shape) == 2: 
             return cv2.LUT(image, lut)
        else:
             self.logger.error(f"[ImageProcessor] Unexpected image shape for LUT application: {image.shape}")
             return image 


    def _apply_color_matrix(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0 
        original_shape = img_float.shape

        if len(original_shape) != 3 or original_shape[2] != 3:
             self.logger.error(f"[ImageProcessor] Error: Image for color matrix is not 3-channel RGB. Shape: {original_shape}")
             return (image * 255.0).astype(np.uint8) if image.dtype==np.float32 else image.astype(np.uint8)

        img_reshaped = img_float.reshape(-1, 3)
        color_matrix_32 = self._COLOR_MATRIX.astype(np.float32)
        color_corrected_reshaped = img_reshaped @ color_matrix_32.T
        color_corrected_img = color_corrected_reshaped.reshape(original_shape)

        color_corrected_img *= 255.0
        np.clip(color_corrected_img, 0, 255, out=color_corrected_img)

        return color_corrected_img.astype(np.uint8)

    def process_image(self, image_msg: Image) -> np.ndarray | None:
        start_total = time.monotonic()
        timings = {}

        processed_image = None

        try:
            # --- Step 1: ROS Msg to OpenCV ---
            start_step = time.monotonic()
            raw_cv_image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
            timings['1_to_cv2'] = (time.monotonic() - start_step) * 1000
            self.logger.debug(f"Raw image shape: {raw_cv_image.shape}, dtype: {raw_cv_image.dtype}")

            if len(raw_cv_image.shape) != 2:
                self.logger.error(f"Expected single-channel raw Bayer image from 'passthrough', but got shape {raw_cv_image.shape}. Check camera driver or CvBridge behavior.")
                return None

            # --- Step 2: Demosaic ---
            start_step = time.monotonic()
            rgb_image = cv2.cvtColor(raw_cv_image, self._DEMOSAIC_CODE)
            timings['2_demosaic'] = (time.monotonic() - start_step) * 1000
            self.logger.debug(f"Demosaiced image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

            """
            # --- Step 3: First Gamma Correction ---
            start_step = time.monotonic()
            gamma_corrected_1 = self._apply_gamma(rgb_image, self._gamma_lut_inv)
            timings['3_gamma1'] = (time.monotonic() - start_step) * 1000

            # --- Step 4: Color Matrix Correction ---
            start_step = time.monotonic()
            color_corrected = self._apply_color_matrix(gamma_corrected_1)
            timings['4_color_matrix'] = (time.monotonic() - start_step) * 1000

            # --- Step 5: Second Gamma Correction ---
            start_step = time.monotonic()
            gamma_corrected_2 = self._apply_gamma(color_corrected, self._gamma_lut_fwd)
            timings['5_gamma2'] = (time.monotonic() - start_step) * 1000
            """

            # --- Step 6: Resize ---
            start_step = time.monotonic()
            resized_image = cv2.resize(
                rgb_image,
                (self._target_width, 2144),
                interpolation=cv2.INTER_NEAREST
            )
            timings['6_resize'] = (time.monotonic() - start_step) * 1000

            # --- Step 6.5: Crop Image ---
            cropped_image = resized_image[0:1600,:,:]

            # --- Step 7: Final Color Conversion (RGB to BGR) ---
            start_step = time.monotonic()
            bgr_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            timings['7_to_bgr'] = (time.monotonic() - start_step) * 1000

            #self.logger.warning(f"IMAGE DIM: {bgr_image.shape}")
            processed_image = bgr_image

        except CvBridgeError as e:
            self.logger.error(f"[ImageProcessor] CvBridge Error: {e}")
            self.logger.error(f"  > Image message encoding was: {image_msg.encoding}")
        except Exception as e:
            self.logger.error(f"[ImageProcessor] Unexpected error during processing: {e}")
            self.logger.error(traceback.format_exc())

        
        total_time = (time.monotonic() - start_total) * 1000
        
        timing_log_parts = [f"{step}: {ms:.1f}ms" for step, ms in timings.items()]
        timing_log_str = " | ".join(timing_log_parts)

        if processed_image is not None:
             self.logger.info(f"[ImageProcessor] Timings: {timing_log_str} | Total: {total_time:.1f}ms") 
        else:
             self.logger.error(f"[ImageProcessor] Processing failed. Timings recorded: {timing_log_str}")
        
        return processed_image