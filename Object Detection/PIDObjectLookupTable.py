from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from transformers import SamModel, SamImageProcessor


class ObjectLookupTable():
    
    def __init__(self, labels, gt_image_paths, encoder, img_size=224):
        """
            Parameters:
                labels (json): JSON mapping class names & ids
                gt_image_paths (glob): Paths to Ground Truth (gt) images
                encoder (torch): Image encoder 
                inference_tile_size (int): SAHI tile size
        """
        self.device = self._get_device()
        self.encoder = encoder.eval().to(self.device)
        self.class_labels = labels
        self.n_classes = len(gt_image_paths)
        self.gt_image_paths = gt_image_paths
        self.img_size = img_size
        
        self.normalize = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.753, 0.753, 0.754],
                         std=[0.333, 0.333, 0.332])
        ])

        self.gt_embeddings = self._get_embeddings(self.gt_image_paths)

    def _get_device(self):
        """Check for GPU availability."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _get_embeddings(self, image_paths):
        """
        Extract embeddings for each image in the directory
        """
        embeddings = []

        for img in image_paths:
            object = Image.open(img).convert('RGB')
            
            # Re-scale & pad img to size expected by encoder
            object = self.normalize(object)
            # Add batch dimension
            padded_detection = self._square_pad(object).unsqueeze(0)
            padded_detection = padded_detection.to(self.device)
            
            with torch.no_grad():
                embedding = self.encoder(padded_detection)
            
            # Normalize embedding to compute normalized dot product 
            embedding_norm = embedding / embedding.norm(p=2)
            embeddings.append(embedding_norm.squeeze(0))
        
        return torch.stack(embeddings)

    def _square_pad(self, image):
        """
        Scale and pad detection images to self.img_size x self.image_size pixels 
        for consistent sizing for YOLO inference
        """
        _, h, w = image.shape
        # Scale image while preserving aspect ratio if larger than img_size
        if h > self.img_size or w > self.img_size:
            scale_factor = self.img_size / max(h, w)
            h, w = int(h * scale_factor), int(w * scale_factor)
            image = v2.functional.resize(image, (h, w))
        
        hp = (self.img_size - w) // 2
        vp = (self.img_size - h) // 2

        top, bottom = vp, self.img_size - h - vp
        left, right = hp, self.img_size - w - hp

        # White padding 
        return v2.functional.pad(image, (left, right, top, bottom), fill=1)

    
    def classify(self, detection_path):
        """
        Classify detected objects by taking index where cosine similarity is highest between
        detection and gt embedding
        """
        # Get embeddings for detections
        detection_embeddings = self._get_embeddings(detection_path)
        
        # Classify by taking index with max inner product
        cosine_similarity_matrix = torch.matmul(detection_embeddings, self.gt_embeddings.T)
        max_similarity, best_match_indices = torch.max(cosine_similarity_matrix, dim=1)
        
        return max_similarity, best_match_indices
    


class ObjectLookupTableSAM():

    def __init__(self, labels, gt_image_paths):
        """
            Parameters:
                labels (json): JSON mapping class names & ids
                gt_image_paths (glob): Paths to Ground Truth (gt) images
        """
        self.device = self._get_device()
        self.model_name = 'facebook/sam-vit-huge'
        self.preprocessor = SamImageProcessor.from_pretrained(self.model_name)
        self.encoder = SamModel.from_pretrained(self.model_name).vision_encoder.to(self.device)
        self.encoder.eval()
        
        self.class_labels = labels
        self.n_classes = len(gt_image_paths)
        self.gt_image_paths = gt_image_paths
        self.gt_embeddings = self._get_embeddings(self.gt_image_paths)


    def _get_device(self):
        """Check for GPU availability."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    def _get_embeddings(self, image_paths):
        """
        Extract embeddings for each image in the directory
        """
        embeddings = []
        
        for img in image_paths:
            img = Image.open(img).convert('RGB')
            preprocessed_img = self.preprocessor(img, return_tensors="pt").to(self.device)
                       
            with torch.no_grad():
                output = self.encoder(pixel_values=preprocessed_img["pixel_values"])
                feature = output.last_hidden_state.squeeze(0) # [256, H/16, W/16]
                feature_pooled = feature.flatten(start_dim=1).mean(dim=1) # Mean pool to 256
                feature_normed = F.normalize(feature_pooled, dim=0)

            embeddings.append(feature_normed)
        
        return torch.stack(embeddings)

    def classify(self, detection_paths):
        """
        Classify detected objects by taking index where cosine similarity is highest between
        detection and gt embedding
        """
        # Get embeddings for detections
        detection_embeddings = self._get_embeddings(detection_paths)
        
        # Classify by taking index with max inner product
        cosine_similarity_matrix = torch.matmul(detection_embeddings, self.gt_embeddings.T)
        max_similarity, best_match_indices = torch.max(cosine_similarity_matrix, dim=1)
        
        return max_similarity, best_match_indices