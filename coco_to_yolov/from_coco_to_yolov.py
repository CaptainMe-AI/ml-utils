import logging
import random
import shutil
import json
import yaml
import functools
from pathlib import Path

class FromCocoToYolov:
    def __init__(self, coco_data_path, yolov_data_path, loggingLevel = logging.ERROR):
        self.coco_data_path = Path(coco_data_path)
        self.yolov_data_path = Path(yolov_data_path)
        self.logger = logging.getLogger('FromCocoToYolov')
        self.logger.setLevel(loggingLevel)
    
    def convert(self):
        self._log('Starting Conversion')
        coco_data = self._load_coco_data()
        self._yolov_data_structure()
        self._create_yolov_data(coco_data)
        self._create_yolov_config(coco_data)
    
    
    def _create_yolov_config(self, coco_data):
        categories = []
        for categoy in coco_data['categories']:
            categories.append(categoy['name'])
        
        data_yaml = {
            'train': str(self._yolov_train_file_path()),
            'val': str(self._yolov_val_file_path()),
            'test': str(self._yolov_test_file_path()),
            'nc': len(categories),
            'names': categories
        }
        
        with open(self.yolov_data_path / 'data.yaml', 'w') as outfile:
            yaml.dump(data_yaml, outfile)    

    def _create_yolov_data(self, coco_data):
    # val - 3%, test - 25%, train - 72%

        total_images = len(coco_data['images'])
        train_data_size= int(total_images * 0.72)
        test_data_size = int(total_images * 0.25)
        val_data_size = int(total_images * 0.03)
        
        annotation_dic = dict()
        for item in coco_data['annotations']:
            annotation_dic[item['image_id']] = item 
            
        # Randomly assign images to train, test, val until size is reached
        random.shuffle(coco_data['images'])
        
        for image in coco_data['images']:
            annotation = annotation_dic[image['id']]
            
            if train_data_size > 0:
                self._create_yolov_train_data('train', image, annotation)
                train_data_size -= 1
            elif test_data_size > 0:
                self._create_yolov_train_data('test', image, annotation)
                test_data_size -= 1
            elif val_data_size > 0:
                self._create_yolov_train_data('val', image, annotation)
                val_data_size -= 1
            else:
                break
    
    def _create_yolov_train_data(self, data_type, image, annotation):
        self._log(f'-- Image {image["id"]} | Data Type {data_type} --')


        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        # Copy Image
        image_source_path = self.coco_data_path / image['file_name']
        image_dest_path = self._image_dest_path(data_type, image_id)
        self._log(f'Image from: {image_source_path} to: {image_dest_path}')
        shutil.copyfile(image_source_path, image_dest_path)
        
        # Write txt file
        # Box coordinates must be in normalized xywh format (from 0 - 1). 
        #   - If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
        category_id = annotation['category_id']
        image_width = image['width']
        image_height = image['height']
        bbox = annotation['bbox']

        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        
        # Finding midpoints
        x_centre = (x + (x+w))/2
        y_centre = (y + (y+h))/2
        
        # Normalization
        x_centre = x_centre / image_width
        y_centre = y_centre / image_height
        w = w / image_width
        h = h / image_height
        
        # Limiting upto fix number of decimal places
        x_centre = format(x_centre, '.6f')
        y_centre = format(y_centre, '.6f')
        w = format(w, '.6f')
        h = format(h, '.6f')
        
        annotation_line = f'{category_id} {x_centre} {y_centre} {w} {h}'
        self._log(f'Annotation Line: {annotation_line}')
        
        if data_type == 'train' or data_type == 'val':
            label_dest_path = self._label_dest_path(data_type, image_id)
        self._log(f'Label to: {label_dest_path} ')
        
        with open(label_dest_path, 'a') as the_file:
            the_file.write(f'{annotation_line}\n')
        
        # Write image ref file 
        ref_file = self._label_path(data_type)
        self._log(f'Ref File {ref_file}')
        
        image_ref_path = image_dest_path.relative_to(self.yolov_data_path)
        self._log(f'Relative Image Path {image_ref_path}')
        with open(ref_file, 'a') as the_file:
            the_file.write(f'./{image_ref_path}\n')

    def _load_coco_data(self):
        if not self.coco_data_path.exists():
            raise FileNotFoundError(f'Folder {self.coco_data_path} not found')
        
        coco_resuts_file = self.coco_data_path / 'result.json'
        if not coco_resuts_file.exists():
            raise FileNotFoundError(f'File {coco_resuts_file} not found')
        
        coco_data = json.load(open(coco_resuts_file))
        
        self._log('Coco Data Loaded')
        
        return coco_data

    def _yolov_data_structure(self):
    
        # Root structure
        os.mkdir(self.yolov_data_path) if not self.yolov_data_path.exists() else None
        # create empty reference files
        open(self._yolov_val_file_path(), 'a').close()
        open(self._yolov_test_file_path(), 'a').close()
        open(self._yolov_train_file_path(), 'a').close()

        # Annotations structure
        os.mkdir(self._yolov_annotations_path()) if not self._yolov_annotations_path().exists() else None
        # Images structure
        os.mkdir(self._yolov_images_path()) if not self._yolov_images_path().exists() else None
        os.mkdir(self._yolov_val_images_path()) if not self._yolov_val_images_path().exists() else None
        os.mkdir(self._yolov_test_images_path()) if not self._yolov_test_images_path().exists() else None
        os.mkdir(self._yolov_train_images_path()) if not self._yolov_train_images_path().exists() else None
        # Labels structure
        os.mkdir(self._yolov_labels_path()) if not self._yolov_labels_path().exists() else None
        os.mkdir(self._yolov_val_labels_path()) if not self._yolov_val_labels_path().exists() else None
        os.mkdir(self._yolov_train_labels_path()) if not self._yolov_train_labels_path().exists() else None
        
        self._log('Yolov Data Structure Created')
    
    def _image_dest_path(self, data_type, image_id):
        if data_type == 'train':
            return self._yolov_train_images_path() / f'{image_id}.jpg'
        if data_type == 'test':
            return self._yolov_test_images_path() / f'{image_id}.jpg'
        if data_type == 'val':
            return self._yolov_val_images_path() / f'{image_id}.jpg'

    def _label_dest_path(self, data_type, image_id):
        if data_type == 'train':
            return self._yolov_train_labels_path() / f'{image_id}.txt'
        if data_type == 'val':
            return self._yolov_val_labels_path() / f'{image_id}.txt'

    def _label_path(self, data_type):
        if data_type == 'train':
            return self._yolov_train_file_path()
        if data_type == 'test':
            return self._yolov_test_file_path()
        if data_type == 'val':
            return self._yolov_val_file_path()
                    
    # {root}/annotations
    @functools.cache
    def _yolov_annotations_path(self):
        return self.yolov_data_path / 'annotations'

    # {root}/images
    @functools.cache
    def _yolov_images_path(self):
        return self.yolov_data_path / 'images'

    # {root}/images/val
    def _yolov_val_images_path(self):
        return self._yolov_images_path() / 'val'

    # {root}/images/train
    def _yolov_train_images_path(self):
        return self._yolov_images_path() / 'train'

    # {root}/images/test
    def _yolov_test_images_path(self):
        return self._yolov_images_path() / 'test'

    # {root}/labels
    def _yolov_labels_path(self):
        return self.yolov_data_path / 'labels'

    # {root}/labels/val
    def _yolov_val_labels_path(self):
        return self._yolov_labels_path() / 'val'

    # {root}/labels/train
    def _yolov_train_labels_path(self):
        return self._yolov_labels_path() / 'train'

    ### {root}/val.txt
    def _yolov_val_file_path(self):
        return self.yolov_data_path / 'val.txt'

    ### {root}/test.txt
    def _yolov_test_file_path(self):
        return self.yolov_data_path / 'test.txt'

    ### {root}/train.txt
    def _yolov_train_file_path(self):
        return self.yolov_data_path / 'train.txt'
        
    def _log(self, msg):
        self.logger.debug(msg)  
    