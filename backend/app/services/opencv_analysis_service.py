"""
OpenCV computer vision analysis service for GeoGuessr
Provides real-time image analysis with object detection and geographic feature identification
"""

import cv2
import numpy as np
import requests
from typing import List, Dict, Any, Tuple
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OpenCVAnalysisService:
    """Real-time computer vision analysis for Street View images"""
    
    def __init__(self):
        self.object_classifiers = self._initialize_classifiers()
        self.geographic_patterns = self._load_geographic_patterns()
        
    def _initialize_classifiers(self) -> Dict[str, Any]:
        """Initialize OpenCV classifiers and detectors"""
        try:
            # Initialize basic cascades (you can expand this with YOLO/other models)
            cascades = {}
            
            # Car detection (simplified - in production use YOLO)
            cascades['vehicle'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
            
            # Text detection
            cascades['text'] = cv2.EAST('frozen_east_text_detection.pb') if self._model_exists('frozen_east_text_detection.pb') else None
            
            return cascades
        except Exception as e:
            logger.warning(f"Could not initialize all classifiers: {e}")
            return {}
    
    def _model_exists(self, model_path: str) -> bool:
        """Check if model file exists"""
        try:
            import os
            return os.path.exists(model_path)
        except:
            return False
    
    def _load_geographic_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate specific geographic regions"""
        return {
            'architecture': {
                'european': ['red_tile_roof', 'balcony', 'shutters', 'stone_building'],
                'asian': ['pagoda_style', 'curved_roof', 'bamboo', 'rice_fields'],
                'american': ['wooden_house', 'fire_hydrant', 'stop_sign', 'yellow_lines'],
                'tropical': ['palm_trees', 'thatched_roof', 'bright_colors']
            },
            'infrastructure': {
                'driving_side': ['left_driving', 'right_driving'],
                'signs': ['english_signs', 'cyrillic_signs', 'arabic_signs', 'asian_scripts'],
                'utilities': ['power_lines', 'phone_poles', 'street_lamps']
            },
            'vegetation': {
                'climate': ['desert', 'tropical', 'temperate', 'arctic'],
                'types': ['palm_trees', 'pine_trees', 'deciduous', 'cacti', 'rice_fields']
            }
        }
    
    async def analyze_street_view(self, image_url: str, location: Dict[str, float], view: Dict[str, float]) -> Dict[str, Any]:
        """Perform comprehensive CV analysis on Street View image"""
        try:
            # Download and process image
            image = await self._download_image(image_url)
            if image is None:
                return self._empty_analysis()
            
            # Perform various analyses
            objects = self._detect_objects(image)
            features = self._extract_features(image)
            geographic_hints = self._analyze_geographic_features(image, objects)
            color_analysis = self._analyze_colors(image)
            
            return {
                'objects': objects,
                'features': features,
                'geographical_hints': geographic_hints,
                'color_analysis': color_analysis,
                'timestamp': np.datetime64('now').astype(int),
                'location': location,
                'view': view
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._empty_analysis()
    
    async def _download_image(self, image_url: str) -> np.ndarray:
        """Download and convert image to OpenCV format"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL then to OpenCV
            pil_image = Image.open(BytesIO(response.content))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return cv_image
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return None
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and classify objects in the image"""
        objects = []
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect vehicles (simplified)
            vehicles = self._detect_vehicles(gray)
            objects.extend(vehicles)
            
            # Detect buildings/structures
            buildings = self._detect_buildings(image)
            objects.extend(buildings)
            
            # Detect vegetation
            vegetation = self._detect_vegetation(hsv)
            objects.extend(vegetation)
            
            # Detect signs and text
            signs = self._detect_signs(image)
            objects.extend(signs)
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _detect_vehicles(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicles using edge detection and contours"""
        vehicles = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Vehicle-like size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Vehicle-like aspect ratio
                    if 1.2 < aspect_ratio < 4.0:
                        vehicles.append({
                            'type': 'vehicle',
                            'confidence': min(0.7, area / 20000),
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'color': '#4ECDC4'
                        })
                        
            return vehicles[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Vehicle detection failed: {e}")
            return []
    
    def _detect_buildings(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect building structures"""
        buildings = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect straight lines (building edges)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Group vertical and horizontal lines
                vertical_lines = []
                horizontal_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if abs(angle) < 15 or abs(angle) > 165:  # Horizontal-ish
                        horizontal_lines.append(line)
                    elif 75 < abs(angle) < 105:  # Vertical-ish
                        vertical_lines.append(line)
                
                # If we have both vertical and horizontal lines, likely buildings
                if len(vertical_lines) > 2 and len(horizontal_lines) > 2:
                    buildings.append({
                        'type': 'building',
                        'confidence': min(0.8, (len(vertical_lines) + len(horizontal_lines)) / 20),
                        'bbox': [0, 0, image.shape[1]//3, image.shape[0]//2],
                        'color': '#FF6B6B'
                    })
            
            return buildings
            
        except Exception as e:
            logger.error(f"Building detection failed: {e}")
            return []
    
    def _detect_vegetation(self, hsv_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vegetation using color analysis"""
        vegetation = []
        
        try:
            # Green color range for vegetation
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Create mask for green areas
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            
            # Find contours of green areas
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Significant vegetation area
                    x, y, w, h = cv2.boundingRect(contour)
                    vegetation.append({
                        'type': 'vegetation',
                        'confidence': min(0.9, area / 50000),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'color': '#95E1D3'
                    })
            
            return vegetation[:3]  # Limit to top 3
            
        except Exception as e:
            logger.error(f"Vegetation detection failed: {e}")
            return []
    
    def _detect_signs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect signs and text areas"""
        signs = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for rectangular shapes (sign-like)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if 500 < area < 10000:  # Sign-like size
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Sign-like aspect ratio
                        if 0.5 < aspect_ratio < 3.0:
                            signs.append({
                                'type': 'sign',
                                'confidence': 0.6,
                                'bbox': [int(x), int(y), int(w), int(h)],
                                'color': '#FF9F43'
                            })
            
            return signs[:3]  # Limit to top 3
            
        except Exception as e:
            logger.error(f"Sign detection failed: {e}")
            return []
    
    def _extract_features(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract distinctive features using SIFT/ORB"""
        features = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use ORB feature detector
            orb = cv2.ORB_create(nfeatures=50)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if keypoints:
                # Convert keypoints to feature format
                points = [[int(kp.pt[0]), int(kp.pt[1])] for kp in keypoints[:20]]
                
                features.append({
                    'type': 'keypoints',
                    'points': points,
                    'confidence': min(0.8, len(keypoints) / 100)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return []
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze dominant colors in the image"""
        try:
            # Convert to RGB for color analysis
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = rgb_image.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            return {
                'dominant_colors': [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in colors],
                'color_distribution': kmeans.labels_.tolist()[:100]  # Sample
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {'dominant_colors': [], 'color_distribution': []}
    
    def _analyze_geographic_features(self, image: np.ndarray, objects: List[Dict]) -> List[str]:
        """Analyze features that provide geographic clues"""
        hints = []
        
        try:
            # Analyze based on detected objects
            object_types = [obj['type'] for obj in objects]
            
            if 'vegetation' in object_types:
                hints.append("🌳 Vegetation detected - likely temperate or tropical climate")
            
            if 'vehicle' in object_types:
                hints.append("🚗 Vehicles present - urban or suburban area")
            
            if 'building' in object_types:
                hints.append("🏢 Buildings detected - developed area")
            
            if 'sign' in object_types:
                hints.append("🚏 Signage visible - may contain language/script clues")
            
            # Analyze image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 150:
                hints.append("☀️ Bright lighting - sunny climate or good weather")
            elif brightness < 80:
                hints.append("🌧️ Low lighting - overcast or northern latitude")
            
            # Color analysis for climate hints
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            blue_pixels = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            blue_ratio = np.sum(blue_pixels > 0) / blue_pixels.size
            
            if blue_ratio > 0.3:
                hints.append("💧 Significant blue areas - water nearby or clear sky")
            
            return hints[:5]  # Limit to top 5 hints
            
        except Exception as e:
            logger.error(f"Geographic analysis failed: {e}")
            return ["🤖 Analysis in progress..."]
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'objects': [],
            'features': [],
            'geographical_hints': ["🤖 Analysis unavailable"],
            'color_analysis': {'dominant_colors': [], 'color_distribution': []},
            'timestamp': 0,
            'location': {},
            'view': {}
        }
