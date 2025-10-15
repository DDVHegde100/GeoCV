"""
Core Computer Vision Pipeline for GeoCV
Heavy emphasis on OpenCV-based feature extraction and analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DetectionResult:
    """Structure for individual detection results"""
    feature_type: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    metadata: Dict[str, Any]
    geographic_hints: List[str]

@dataclass
class CVAnalysisResult:
    """Complete CV analysis result structure"""
    detections: List[DetectionResult]
    overall_confidence: float
    processing_time: float
    suggested_regions: List[str]
    confidence_level: ConfidenceLevel
    feature_summary: Dict[str, int]

class CVPipeline:
    """
    Main Computer Vision Pipeline using OpenCV
    Focuses on traditional CV methods with strategic ML integration
    """
    
    def __init__(self):
        self.initialized = False
        self.cascade_classifiers = {}
        self.feature_extractors = {}
        self.background_subtractor = None
        
        # Initialize detection thresholds
        self.thresholds = {
            'vehicle_confidence': 0.3,
            'sign_confidence': 0.4, 
            'architecture_confidence': 0.35,
            'vegetation_confidence': 0.25,
            'road_confidence': 0.3
        }
        
    async def initialize(self):
        """Initialize CV pipeline components"""
        try:
            # Load cascade classifiers for object detection
            await self._load_cascade_classifiers()
            
            # Initialize feature extractors
            await self._initialize_feature_extractors()
            
            # Setup background subtraction for motion analysis
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True
            )
            
            self.initialized = True
            logger.info("CV Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CV pipeline: {e}")
            raise
    
    async def _load_cascade_classifiers(self):
        """Load OpenCV cascade classifiers for object detection"""
        cascade_paths = {
            'car': cv2.data.haarcascades + 'haarcascade_car.xml',
            'frontal_face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            # Add more cascades as needed
        }
        
        for name, path in cascade_paths.items():
            if os.path.exists(path):
                self.cascade_classifiers[name] = cv2.CascadeClassifier(path)
                logger.info(f"Loaded cascade classifier: {name}")
    
    async def _initialize_feature_extractors(self):
        """Initialize feature extraction methods"""
        # SIFT for keypoint detection
        self.feature_extractors['sift'] = cv2.SIFT_create()
        
        # ORB for fast feature detection
        self.feature_extractors['orb'] = cv2.ORB_create()
        
        # HOG descriptor for pedestrian/object detection
        self.feature_extractors['hog'] = cv2.HOGDescriptor()
        self.feature_extractors['hog'].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        logger.info("Feature extractors initialized")
    
    async def analyze_image(self, image_path: str) -> CVAnalysisResult:
        """
        Main analysis function - processes image through CV pipeline
        """
        start_time = cv2.getTickCount()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for consistent processing (while maintaining aspect ratio)
        image = self._resize_image(image, max_size=1024)
        
        detections = []
        
        # Run all detection modules
        detections.extend(await self._detect_vehicles(image))
        detections.extend(await self._analyze_roads_and_infrastructure(image))
        detections.extend(await self._detect_vegetation_and_terrain(image))
        detections.extend(await self._analyze_architecture(image))
        detections.extend(await self._analyze_sky_and_lighting(image))
        detections.extend(await self._detect_text_and_signs(image))
        
        # Calculate processing time
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        
        # Aggregate results
        result = self._aggregate_results(detections, processing_time)
        
        return result
    
    async def _detect_vehicles(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect and analyze vehicles using OpenCV methods"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use cascade classifier if available
        if 'car' in self.cascade_classifiers:
            cars = self.cascade_classifiers['car'].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in cars:
                # Extract vehicle region for further analysis
                vehicle_roi = image[y:y+h, x:x+w]
                vehicle_features = self._analyze_vehicle_features(vehicle_roi)
                
                detection = DetectionResult(
                    feature_type="vehicle",
                    confidence=0.7,  # Base confidence from cascade
                    bounding_box=(x, y, w, h),
                    metadata=vehicle_features,
                    geographic_hints=self._get_vehicle_geographic_hints(vehicle_features)
                )
                detections.append(detection)
        
        # Additional vehicle detection using contour analysis
        detections.extend(self._detect_vehicles_by_contour(image))
        
        return detections
    
    def _analyze_vehicle_features(self, vehicle_roi: np.ndarray) -> Dict[str, Any]:
        """Analyze specific features of detected vehicles"""
        features = {}
        
        # Color analysis
        hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        dominant_color = self._get_dominant_color(hsv)
        features['dominant_color'] = dominant_color
        
        # Shape analysis
        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        aspect_ratio = width / height
        features['aspect_ratio'] = aspect_ratio
        
        # Determine vehicle type based on aspect ratio and size
        if aspect_ratio > 2.5:
            features['vehicle_type'] = 'truck_or_bus'
        elif aspect_ratio > 1.8:
            features['vehicle_type'] = 'car'
        else:
            features['vehicle_type'] = 'compact_car'
        
        return features
    
    def _get_vehicle_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on vehicle features"""
        hints = []
        
        vehicle_type = features.get('vehicle_type', '')
        
        if vehicle_type == 'truck_or_bus':
            hints.extend(['commercial_area', 'developed_country', 'highway_system'])
        elif vehicle_type == 'compact_car':
            hints.extend(['urban_area', 'europe_or_asia', 'city_center'])
        
        # Color-based hints (very basic heuristics)
        color = features.get('dominant_color', '')
        if color == 'white':
            hints.extend(['hot_climate', 'middle_east_or_australia'])
        elif color == 'yellow':
            hints.extend(['taxi', 'urban_area', 'new_york_or_mumbai'])
        
        return hints
    
    async def _analyze_roads_and_infrastructure(self, image: np.ndarray) -> List[DetectionResult]:
        """Analyze road markings, surface, and infrastructure"""
        detections = []
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for road markings
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform for lane detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            road_features = self._analyze_road_lines(lines, image.shape)
            
            detection = DetectionResult(
                feature_type="road_infrastructure",
                confidence=0.6,
                bounding_box=(0, 0, image.shape[1], image.shape[0]),
                metadata=road_features,
                geographic_hints=self._get_road_geographic_hints(road_features)
            )
            detections.append(detection)
        
        # Road surface analysis
        road_surface = self._analyze_road_surface(image)
        if road_surface:
            detection = DetectionResult(
                feature_type="road_surface",
                confidence=road_surface['confidence'],
                bounding_box=(0, image.shape[0]//2, image.shape[1], image.shape[0]//2),
                metadata=road_surface,
                geographic_hints=self._get_surface_geographic_hints(road_surface)
            )
            detections.append(detection)
        
        return detections
    
    def _analyze_road_lines(self, lines: np.ndarray, image_shape: tuple) -> Dict[str, Any]:
        """Analyze detected road lines for geographic clues"""
        features = {
            'line_count': len(lines),
            'line_colors': [],
            'line_patterns': [],
            'road_width_estimate': None
        }
        
        # Analyze line orientations
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 30 or abs(angle) > 150:
                horizontal_lines += 1
            elif 60 < abs(angle) < 120:
                vertical_lines += 1
        
        features['horizontal_lines'] = horizontal_lines
        features['vertical_lines'] = vertical_lines
        
        # Estimate road type based on line patterns
        if horizontal_lines > vertical_lines * 2:
            features['road_type'] = 'highway'
        elif vertical_lines > horizontal_lines:
            features['road_type'] = 'city_street'
        else:
            features['road_type'] = 'mixed'
        
        return features
    
    def _get_road_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on road features"""
        hints = []
        
        road_type = features.get('road_type', '')
        line_count = features.get('line_count', 0)
        
        if road_type == 'highway':
            hints.extend(['developed_infrastructure', 'highway_system', 'modern_country'])
        elif road_type == 'city_street':
            hints.extend(['urban_area', 'city_center', 'pedestrian_area'])
        
        if line_count > 10:
            hints.extend(['well_maintained_roads', 'developed_country'])
        elif line_count < 3:
            hints.extend(['rural_area', 'developing_region', 'countryside'])
        
        return hints
    
    async def _detect_vegetation_and_terrain(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect and analyze vegetation and terrain features"""
        detections = []
        
        # Convert to HSV for better vegetation detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define vegetation color ranges (green hues)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find vegetation contours
        contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vegetation_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                vegetation_areas.append((x, y, w, h, area))
        
        if vegetation_areas:
            # Analyze vegetation characteristics
            vegetation_features = self._analyze_vegetation(image, vegetation_areas)
            
            detection = DetectionResult(
                feature_type="vegetation",
                confidence=0.5,
                bounding_box=(0, 0, image.shape[1], image.shape[0]),
                metadata=vegetation_features,
                geographic_hints=self._get_vegetation_geographic_hints(vegetation_features)
            )
            detections.append(detection)
        
        return detections
    
    def _analyze_vegetation(self, image: np.ndarray, areas: List[tuple]) -> Dict[str, Any]:
        """Analyze vegetation characteristics"""
        total_area = sum([area[4] for area in areas])
        image_area = image.shape[0] * image.shape[1]
        vegetation_coverage = total_area / image_area
        
        features = {
            'coverage_percentage': vegetation_coverage * 100,
            'vegetation_density': len(areas),
            'vegetation_type': 'unknown'
        }
        
        # Classify vegetation type based on coverage and density
        if vegetation_coverage > 0.6:
            features['vegetation_type'] = 'forest_or_jungle'
        elif vegetation_coverage > 0.3:
            features['vegetation_type'] = 'parkland_or_suburbs'
        elif vegetation_coverage > 0.1:
            features['vegetation_type'] = 'scattered_trees'
        else:
            features['vegetation_type'] = 'minimal_vegetation'
        
        return features
    
    def _get_vegetation_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on vegetation"""
        hints = []
        
        vegetation_type = features.get('vegetation_type', '')
        coverage = features.get('coverage_percentage', 0)
        
        if vegetation_type == 'forest_or_jungle':
            hints.extend(['tropical_region', 'temperate_forest', 'rural_area', 'canada_or_russia'])
        elif vegetation_type == 'minimal_vegetation':
            hints.extend(['arid_climate', 'desert_region', 'urban_center', 'middle_east'])
        elif coverage > 40:
            hints.extend(['temperate_climate', 'suburban_area', 'parks_system'])
        
        return hints
    
    async def _analyze_architecture(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect and analyze architectural features"""
        detections = []
        
        # Edge detection for building outlines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Find rectangular structures (buildings)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buildings = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular and large enough
            if len(approx) >= 4 and cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                buildings.append((x, y, w, h))
        
        if buildings:
            arch_features = self._analyze_architectural_style(image, buildings)
            
            detection = DetectionResult(
                feature_type="architecture",
                confidence=0.4,
                bounding_box=(0, 0, image.shape[1], image.shape[0]//2),
                metadata=arch_features,
                geographic_hints=self._get_architecture_geographic_hints(arch_features)
            )
            detections.append(detection)
        
        return detections
    
    def _analyze_architectural_style(self, image: np.ndarray, buildings: List[tuple]) -> Dict[str, Any]:
        """Analyze architectural characteristics"""
        features = {
            'building_count': len(buildings),
            'building_density': 'low',
            'architectural_style': 'modern',
            'building_heights': []
        }
        
        # Analyze building density
        total_building_area = sum([w * h for x, y, w, h in buildings])
        image_area = image.shape[0] * image.shape[1]
        density_ratio = total_building_area / image_area
        
        if density_ratio > 0.5:
            features['building_density'] = 'high'
        elif density_ratio > 0.2:
            features['building_density'] = 'medium'
        
        # Analyze building heights (rough estimate from bounding boxes)
        heights = [h for x, y, w, h in buildings]
        if heights:
            avg_height = np.mean(heights)
            features['average_building_height'] = avg_height
            
            if avg_height > 200:
                features['building_type'] = 'high_rise'
            elif avg_height > 100:
                features['building_type'] = 'mid_rise'
            else:
                features['building_type'] = 'low_rise'
        
        return features
    
    def _get_architecture_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on architecture"""
        hints = []
        
        building_type = features.get('building_type', '')
        density = features.get('building_density', '')
        
        if building_type == 'high_rise':
            hints.extend(['major_city', 'developed_country', 'urban_center'])
        elif building_type == 'low_rise' and density == 'high':
            hints.extend(['suburban_area', 'residential_zone', 'developed_suburbs'])
        elif density == 'low':
            hints.extend(['rural_area', 'countryside', 'small_town'])
        
        return hints
    
    async def _analyze_sky_and_lighting(self, image: np.ndarray) -> List[DetectionResult]:
        """Analyze sky conditions and lighting for geographic clues"""
        detections = []
        
        # Extract sky region (top portion of image)
        height, width = image.shape[:2]
        sky_region = image[0:height//3, :]
        
        # Analyze sky color and brightness
        sky_features = self._analyze_sky_characteristics(sky_region)
        
        if sky_features:
            detection = DetectionResult(
                feature_type="sky_lighting",
                confidence=0.3,
                bounding_box=(0, 0, width, height//3),
                metadata=sky_features,
                geographic_hints=self._get_sky_geographic_hints(sky_features)
            )
            detections.append(detection)
        
        return detections
    
    def _analyze_sky_characteristics(self, sky_region: np.ndarray) -> Dict[str, Any]:
        """Analyze sky color and lighting characteristics"""
        features = {}
        
        # Convert to HSV for better color analysis
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness
        brightness = np.mean(hsv_sky[:, :, 2])
        features['brightness'] = brightness
        
        # Determine sky condition based on brightness and color
        if brightness > 200:
            features['sky_condition'] = 'bright_sunny'
        elif brightness > 150:
            features['sky_condition'] = 'partly_cloudy'
        elif brightness > 100:
            features['sky_condition'] = 'overcast'
        else:
            features['sky_condition'] = 'dark_or_night'
        
        # Analyze color distribution
        blue_pixels = cv2.inRange(hsv_sky, np.array([100, 50, 50]), np.array([130, 255, 255]))
        blue_ratio = np.sum(blue_pixels > 0) / (sky_region.shape[0] * sky_region.shape[1])
        features['blue_sky_ratio'] = blue_ratio
        
        return features
    
    def _get_sky_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on sky characteristics"""
        hints = []
        
        condition = features.get('sky_condition', '')
        blue_ratio = features.get('blue_sky_ratio', 0)
        
        if condition == 'bright_sunny' and blue_ratio > 0.5:
            hints.extend(['clear_weather', 'dry_climate', 'mediterranean_or_desert'])
        elif condition == 'overcast':
            hints.extend(['temperate_climate', 'northern_regions', 'rainy_season'])
        elif blue_ratio < 0.2:
            hints.extend(['cloudy_region', 'tropical_climate', 'monsoon_area'])
        
        return hints
    
    async def _detect_text_and_signs(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect text and signs using OpenCV preprocessing + OCR"""
        detections = []
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold for better text detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find text regions using contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter potential text regions by aspect ratio and size
            aspect_ratio = w / h
            if 1 < aspect_ratio < 10 and w > 20 and h > 10:
                text_regions.append((x, y, w, h))
        
        if text_regions:
            text_features = {
                'text_regions_count': len(text_regions),
                'detected_languages': [],  # Would integrate OCR here
                'sign_types': []
            }
            
            detection = DetectionResult(
                feature_type="text_and_signs",
                confidence=0.35,
                bounding_box=(0, 0, image.shape[1], image.shape[0]),
                metadata=text_features,
                geographic_hints=['text_detected', 'signage_present']
            )
            detections.append(detection)
        
        return detections
    
    def _detect_vehicles_by_contour(self, image: np.ndarray) -> List[DetectionResult]:
        """Additional vehicle detection using contour analysis"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter potential vehicles by size and aspect ratio
            if 500 < area < 10000 and 1.2 < aspect_ratio < 4.0:
                # Additional validation could be added here
                detection = DetectionResult(
                    feature_type="vehicle",
                    confidence=0.4,  # Lower confidence for contour-based detection
                    bounding_box=(x, y, w, h),
                    metadata={'detection_method': 'contour', 'area': area, 'aspect_ratio': aspect_ratio},
                    geographic_hints=['vehicle_traffic', 'road_system']
                )
                detections.append(detection)
        
        return detections
    
    def _analyze_road_surface(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze road surface characteristics"""
        # Extract road region (bottom portion of image)
        height, width = image.shape[:2]
        road_region = image[height//2:, :]
        
        # Convert to grayscale for texture analysis
        gray_road = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using standard deviation of intensity
        texture_std = np.std(gray_road)
        
        # Analyze surface color
        hsv_road = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(hsv_road, axis=(0, 1))
        
        surface_features = {
            'texture_roughness': texture_std,
            'average_brightness': avg_color[2],
            'surface_type': 'unknown',
            'confidence': 0.3
        }
        
        # Classify surface type based on texture and color
        if texture_std > 30:
            surface_features['surface_type'] = 'rough_or_gravel'
            surface_features['confidence'] = 0.5
        elif texture_std < 15 and avg_color[2] > 100:
            surface_features['surface_type'] = 'smooth_asphalt'
            surface_features['confidence'] = 0.6
        elif avg_color[2] < 80:
            surface_features['surface_type'] = 'dark_asphalt'
            surface_features['confidence'] = 0.4
        
        return surface_features
    
    def _get_surface_geographic_hints(self, features: Dict[str, Any]) -> List[str]:
        """Generate geographic hints based on road surface"""
        hints = []
        
        surface_type = features.get('surface_type', '')
        
        if surface_type == 'smooth_asphalt':
            hints.extend(['developed_infrastructure', 'modern_roads', 'wealthy_region'])
        elif surface_type == 'rough_or_gravel':
            hints.extend(['rural_area', 'developing_region', 'mountain_roads'])
        elif surface_type == 'dark_asphalt':
            hints.extend(['recent_construction', 'highway_system', 'urban_area'])
        
        return hints
    
    def _resize_image(self, image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image
        
        if height > width:
            new_height = max_size
            new_width = int((width * max_size) / height)
        else:
            new_width = max_size
            new_height = int((height * max_size) / width)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _get_dominant_color(self, hsv_image: np.ndarray) -> str:
        """Get dominant color from HSV image"""
        # Simple hue-based color classification
        hue_channel = hsv_image[:, :, 0]
        avg_hue = np.mean(hue_channel)
        
        if avg_hue < 15 or avg_hue > 165:
            return 'red'
        elif avg_hue < 45:
            return 'yellow'
        elif avg_hue < 75:
            return 'green'
        elif avg_hue < 105:
            return 'cyan'
        elif avg_hue < 135:
            return 'blue'
        else:
            return 'purple'
    
    def _aggregate_results(self, detections: List[DetectionResult], processing_time: float) -> CVAnalysisResult:
        """Aggregate all detection results into final analysis"""
        
        # Calculate overall confidence
        if detections:
            confidences = [d.confidence for d in detections]
            overall_confidence = np.mean(confidences)
        else:
            overall_confidence = 0.0
        
        # Determine confidence level
        if overall_confidence > 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence > 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        # Collect all geographic hints
        all_hints = []
        for detection in detections:
            all_hints.extend(detection.geographic_hints)
        
        # Count unique hints to suggest regions
        from collections import Counter
        hint_counts = Counter(all_hints)
        suggested_regions = [hint for hint, count in hint_counts.most_common(5)]
        
        # Create feature summary
        feature_summary = {}
        for detection in detections:
            feature_type = detection.feature_type
            feature_summary[feature_type] = feature_summary.get(feature_type, 0) + 1
        
        return CVAnalysisResult(
            detections=detections,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            suggested_regions=suggested_regions,
            confidence_level=confidence_level,
            feature_summary=feature_summary
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        self.cascade_classifiers.clear()
        self.feature_extractors.clear()
        if self.background_subtractor:
            self.background_subtractor = None
        
        logger.info("CV Pipeline cleaned up")
