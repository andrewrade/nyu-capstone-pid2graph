import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

class PIDPreprocessor():
    """
    Preprocessor for synthetic P&ID (Piping and Instrumentation Diagrams) data.
    Handles cropping of images, conversion of bounding boxes to YOLO format, and organizing
    data for training and validation.
    """
    def __init__(self, root, width, height, num_imgs, val_start, general_classes=False, crop_coords=None):
        self.root = root # Root directory where imgs are located 
        self.width = width 
        self.height = height 
        self.num_imgs = num_imgs 
        self.val_start = val_start 
        self.x1, self.y1, self.x2, self.y2 = crop_coords if crop_coords is not None else (0, 0, width, height)
        
        self.cropped = False if crop_coords is None else True 
        self.general_classes = general_classes

        if self.general_classes:
            with open(self.root / 'nyu-capstone-2024-PIDGraph'/ 'Object Detection' / 'classes_general.json', 'r') as file:
                self.class_mapping = json.load(file)

        if self.cropped:
            self.width = self.x2 - self.x1 
            self.height = self.y2 - self.y1
        

    def _get_dataset(self, img_id):
        """
        Get dataset based on val index start
        """
        return "train" if img_id < self.val_start else "val"

    def _load_img(self, img_id, folder):
        return cv2.imread(self.root / 'Data' / folder / 'images' / self._get_dataset(img_id) / f'{img_id}.jpg')

    def _save_img(self, img_id, data, folder, file_name):
        path = self.root / 'Data' / folder / 'images' / self._get_dataset(img_id) / file_name
        cv2.imwrite(path, data, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _save_annotations(self, img_id, data, folder, file_name):
        path = self.root / 'Data' / folder / 'labels' / self._get_dataset(img_id) / file_name
        data.to_csv(path, sep=' ', header=False, index=False)

    def _calculate_bbox_area(self, df):
        return (df['x2'] - df['x1']) * (df['y2'] - df['y1'])

    def _adjust_basis_and_clip(self, df, col_idx, row_idx):       
        df_adjust = df.copy()
        
        # Capture actual tile W and H in case tile is cropped
        df_adjust['tile_size_x'] = self.tile_size if (col_idx + self.tile_size) <= self.width else (self.width - col_idx)
        df_adjust['tile_size_y'] = self.tile_size if (row_idx + self.tile_size) <= self.height else (self.height - row_idx)

        # Calculate Original Bbox H & W
        df_adjust['bbx_area'] = self._calculate_bbox_area(df_adjust)

        # Clip box coordinates outside tile FOV
        df_adjust['x1'] = np.clip(df['x1'] - col_idx, 0, df_adjust['tile_size_x'])
        df_adjust['x2'] = np.clip(df['x2'] - col_idx, 0, df_adjust['tile_size_x'])
        df_adjust['y1'] = np.clip(df['y1'] - row_idx, 0, df_adjust['tile_size_y'])
        df_adjust['y2'] = np.clip(df['y2'] - row_idx, 0, df_adjust['tile_size_y'])
        
        df_adjust['bbx_clipped_area'] = self._calculate_bbox_area(df_adjust)

        # Remove bounding boxes where 15% or less of object is in view
        df_adjust = df_adjust[df_adjust['bbx_clipped_area'] > 0.2 * df_adjust['bbx_area']]

        return df_adjust


    def crop_pids(self, folder):
        """
        Crop out P&ID border and title block
        Input:
            folder (str): Name of folder containing dataset to crop. 
            Cropped images will be saved to a folder named `dataset_folder`_cropped
        """
        for i in range(self.num_imgs):
            img = self._load_img(i, folder)
            img_crop = img[self.y1:self.y2, self.x1:self.x2, ]
            self._save_img(i, img_crop, f'{folder}_cropped', f'{i}_cropped.jpg')


    def convert_segment_annotations(self, img_id, folder, row_indices, column_indices):
         """
        Adjust bounding box labels to account for image segmentation. Saves updated annotations

        Input:
            img_id (int): id of the image to segment
            folder (str): Folder for the datset to be processed
            row_indices (list): Row index offsets used during segmentation
            column_indices (list): Column index offsets used during segmentation
        """
         annotation_path = self.root / 'Data' / folder / 'labels' / self._get_dataset(img_id) / f'{img_id}.txt'
         
         df_raw = pd.read_csv(str(annotation_path), sep=" ", header=None)
         df_xyxy = df_raw.apply(lambda row: self.convert_yolo_to_torch(row), axis=1)
         
         for i, (row_idx, col_idx) in enumerate([(row, col) for row in row_indices for col in column_indices]):       
            
            df = df_xyxy.copy()   

            # Filter bounding boxes not within current segment FOV
            df_filtered = df[
                ((df['x1'] <= col_idx + self.tile_size) & (df['x2'] >= col_idx)) &
                ((df['y1'] <= row_idx + self.tile_size) & (df['y2'] >= row_idx))
            ]

            if not df_filtered.empty:
                df_adjusted = self._adjust_basis_and_clip(df_filtered, col_idx, row_idx)
                df_yolo = df_adjusted.apply(lambda row: self.convert_torch_to_yolo(row, segments=True), axis=1)
                
                if not df_yolo.empty:
                    self._save_annotations(img_id, df_yolo, f'{folder}_segmented', file_name=f'{img_id}_{self.tile_size}px_segment_{i}.txt')


    def segment_training_imgs(self, folder, tile_size=640, overlap_ratio=0.2):
        """
        Segment training images into 640 x 640 

        Input:
            folder (str): Name of folder containing dataset to segment.
            tile_size (int): Size in number of pixels to segment squares into.
            overlap (int): Overlap in pixels between image segments.
        """
        self.tile_size = tile_size
        self.step = int(self.tile_size - (self.tile_size * overlap_ratio))
        
        for i in range(self.num_imgs):
            
            img = self._load_img(i, folder)
            # Position of indices w/ overlap
            column_indices = range(0, self.width - self.step, self.step) 
            row_indices = range(0, self.height - self.step, self.step)

            for j, (row, col) in enumerate([(row, col) for row in row_indices for col in column_indices]):
                segment = img[row:row + self.tile_size, col:col + self.tile_size]
                self._save_img(i, segment, f'{folder}_segmented', f'{i}_{self.tile_size}px_segment_{j}.jpg')
            
            self.convert_segment_annotations(i, folder, row_indices, column_indices)

    
    def convert_torch_to_yolo(self, data, segments=False):
        """
        Input:
            data (Series): Bounding box coordinates in Torch format, (label, x1, y1, x2, y2) 
            where the first two entries are top left coordinates and second two are 
            bottom-right corner coordinates.
        Returns:
            yolo_labels (Series): Converts bounding boxes to YOLO format (class_id   center_x  center_y  width  height) 
        """
                       
        if self.general_classes:    
            data['label'] = data['label'].map(lambda label: self.class_mapping[label]['general_class_value'])
        else:
            # Yolo expects 0 indexed labels but our annotations are 1 indexed
            data['label'] = data['label'] - (1 if not segments else 0)
        
        if self.cropped:
            #  Cropping shifts origin, update basis
            data['x1'] = (data['x1'] - self.x1).clip(lower=0)
            data['x2'] = (data['x2'] - self.x1).clip(lower=0)
            data['y1'] = (data['y1'] - self.y1).clip(lower=0)
            data['y2'] = (data['y2'] - self.y1).clip(lower=0)

        
        # Check tile dimensions in case tile is near border and cropped
        img_width = self.width if not segments else data['tile_size_x']
        img_height = self.height if not segments else data['tile_size_y']
        
        # Calculate width, height, center_x, and center_y for YOLO format
        data['width'] = abs(data['x2'] - data['x1']) / img_width
        data['height'] = abs(data['y2'] - data['y1']) / img_height
        data['center_x'] = 0.5 * (data['x1'] + data['x2']) / img_width
        data['center_y'] = 0.5 * (data['y1'] + data['y2']) / img_height
        
        return data[['label', 'center_x', 'center_y', 'width', 'height']]
    

    def convert_yolo_to_torch(self, data):
        """
        Converts yolo format (class_id   center_x  center_y  width  height) -->
        to torch format (x1, y1, x2, y2). Yolo format is normalized & torch format 
        isn't, so need to scale by image dimensions
        """
        label, center_x, center_y, bb_width, bb_height = data

        # Bounding Boxes annotated on uncropped image, adjust to orig dims
        center_x *= (self.width + (self.x1 + (self.width - self.x2)) if self.cropped else self.width)
        center_y *= (self.height + (self.y1 + (self.height - self.y2)) if self.cropped else self.height)
        bb_width *= (self.width + (self.x1 + (self.width - self.x2)) if self.cropped else self.width) 
        bb_height *= (self.height + (self.y1 + (self.height - self.y2)) if self.cropped else self.height)

        x1, x2 = center_x - 0.5 * bb_width, center_x + 0.5 * bb_width
        y1, y2 = center_y - 0.5 * bb_height, center_y + 0.5 * bb_height

        return pd.Series({
            'label': label,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })
        

    def yolo_pre_processing(self, folder):
        """
        Pre-Processes images and labels for YOLO training.
        Writes labels to .txt files in YOLO format.
        """   
        if self.cropped:
            self._crop_pids()

        for i in tqdm(range(self.num_imgs)):

            read_path = self.root / 'Data' / f'synthetic_1' / 'raw_labels' / str(i)  / f'{i}_symbols.npy'
            
            data_df = pd.DataFrame(np.load(read_path, allow_pickle=True), columns=['label', 'x1', 'y1', 'x2', 'y2'])
            yolo_labels = data_df.apply(lambda row: self.convert_torch_to_yolo(row), axis=1)

            self._save_annotations(
                i, 
                yolo_labels, 
                f'{folder}{"_cropped" if self.cropped else ""}{"_general_classes" if self.general_classes else ""}', 
                f'{i}.txt'
            )    

