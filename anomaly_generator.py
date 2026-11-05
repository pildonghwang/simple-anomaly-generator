import numpy as np
import random
from typing import Tuple

class AnomalyGenerator:
    """
    A programmatic 2D image generator that creates controlled synthetic datasets.
    Generates baseline noise arrays (backgrounds) and injects parameterized 
    anomalies ("stains") with configurable boundaries and scattering.
    """
    
    def __init__(self, image_size: int = 256, background_intensity: Tuple[int, int] = (0, 40)):
        """
        Args:
            image_size (int): The width and height of the 2D image (default: 256x256).
            background_intensity (tuple): Min and max pixel intensity for the background noise.
        """
        self.image_size = image_size
        self.flat_size = image_size * image_size
        self.background_intensity = background_intensity
        self.stain_intensity = (55, 110)

    def generate_dataset(
        self, 
        num_classes: int, 
        num_samples: int, 
        noise_ratio: float = 0.03, 
        fuzzy_borders: bool = False, 
        include_complex: bool = False
    ) -> np.ndarray:
        """
        Generates a synthetic dataset of flattened 2D images.
        
        Class Mapping:
        - Class 0: Simple Anomaly (Standard Stain)
        - Class 1: Clean Background
        - Class 2: Complex Anomaly (Scattered/Gradient Stain)
        
        Args:
            num_classes (int): Total number of target classes to generate.
            num_samples (int): Number of image samples per class.
            noise_ratio (float): Base ratio of the image covered by the anomaly.
            fuzzy_borders (bool): If True, applies structural zig-zag distortion to edges.
            include_complex (bool): If True, enables Class 2 with aggressive gradient scattering.
            
        Returns:
            np.ndarray: Dataset of shape (num_classes, num_samples, image_size^2)
        """
        dataset = np.ndarray((num_classes, num_samples, self.flat_size))

        for class_idx in range(num_classes):
            for sample_idx in range(num_samples):
                # 1. Generate clean background noise
                dataset[class_idx, sample_idx] = np.random.randint(
                    self.background_intensity[0], 
                    self.background_intensity[1], 
                    self.flat_size
                )
                
                # 2. Inject Simple Anomaly (Class 0)
                if class_idx == 0:
                    self._inject_anomaly(
                        dataset[class_idx, sample_idx], 
                        noise_ratio, 
                        fuzzy_borders, 
                        is_complex=False
                    )

                # 3. Inject Complex Anomaly (Class 2)
                elif include_complex and class_idx == 2:
                    self._inject_anomaly(
                        dataset[class_idx, sample_idx], 
                        noise_ratio, 
                        fuzzy_borders, 
                        is_complex=True
                    )
                    
        return dataset

    def _inject_anomaly(self, image_array: np.ndarray, base_noise: float, fuzzy: bool, is_complex: bool):
        """Internal method to mutate the array with a structural anomaly (stain)."""
        transparency = np.random.uniform(0.4, 1.0, 1).item() if is_complex else 0.0
        effective_noise = base_noise * (1 + transparency) if is_complex else base_noise
        
        # Calculate anomaly dimensions
        fingerprint = np.random.uniform(effective_noise * 0.5, effective_noise * 1.5, 1).item() if fuzzy else effective_noise
        total_pixels = int(fingerprint * self.flat_size)
        
        width = int(np.ceil(np.sqrt(total_pixels)))
        height = int(total_pixels / width) + (1 if total_pixels % width != 0 else 0)

        # Determine spatial anchor points
        pos_ratios = np.random.uniform(0.2, 0.4, 2)
        anchor_x, anchor_y = int(pos_ratios[0] * self.image_size), int(pos_ratios[1] * self.image_size)
        
        # Boundary collision constraints
        if self.image_size - anchor_y < height:
            anchor_y = self.image_size - height - 1
        if anchor_x + width > self.image_size:
            anchor_x = self.image_size - width - 1

        pixels_remaining = total_pixels
        zig_zag = 0
        zig_range = [int(width * -0.3), int(width * 0.3)] if is_complex and fuzzy else \
                    [int(width * -0.05), int(width * 0.05)] if fuzzy else [0, 0]

        # Apply structural mutation row by row
        for row in range(height):
            if fuzzy:
                shift_prob = np.random.choice([True, False], 1, p=[0.55, 0.45] if is_complex else [0.2, 0.8])
                if shift_prob:
                    direction = -1 if np.random.choice([True, False]) else 1
                    if not (zig_range[0] <= zig_zag <= zig_range[1]):
                        zig_zag += direction

            start_idx = (self.image_size * (anchor_y + row)) + anchor_x + zig_zag
            
            if not is_complex:
                draw_width = width if pixels_remaining > width else pixels_remaining
                image_array[start_idx : start_idx + draw_width] = np.random.randint(
                    self.stain_intensity[0], self.stain_intensity[1], draw_width
                )
                pixels_remaining -= width
            else:
                # Complex specific gradient and scattering logic
                image_array[start_idx : start_idx + width] = np.random.randint(
                    self.stain_intensity[0], self.stain_intensity[1], width
                )
                # Apply peripheral decay simulating scattered margins
                decay_width = int(width * (transparency / 4))
                if decay_width > 0:
                    image_array[start_idx : start_idx + decay_width] -= int(width * 0.1)
