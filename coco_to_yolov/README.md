## Convert data set from coco to yolov format 
  - with test, train, val breakdown


### EX COCO:

  - file structure for COCO format
    ```bash
    .
    ├── images                <-- Images directory
    │   └── image1.jpg        
    ├── results.json          <-- See below       
    ```
  
  - COCO json data definition
    ```json
    {
      "images": [
        {
          "width": 1920,
          "height": 1080,
          "id": 0,
          "file_name": "images/c374a449-bench_press_10.jpg"
        },
        {
          "width": 1920,
          "height": 1080,
          "id": 1,
          "file_name": "images/b1810a5a-bench_press_20.jpg"
        }
      ],
      "categories": [
        {
          "id": 0,
          "name": "bench_press"
        },
        {
          "id": 1,
          "name": "biceps_curl"
        },
      ],
      "annotations": [
        {
          "id": 0,
          "image_id": 0,
          "category_id": 0,
          "segmentation": [],
          "bbox": [
            678.918918918919,
            250.56,
            1132.972972972973,
            829.44
          ],
          "ignore": 0,
          "iscrowd": 0,
          "area": 939733.1027027027
        },
        {
          "id": 1,
          "image_id": 1,
          "category_id": 0,
          "segmentation": [],
          "bbox": [
            670.2702702702701,
            280.8,
            1154.5945945945941,
            799.2
          ],
          "ignore": 0,
          "iscrowd": 0,
          "area": 922751.9999999997
        }
      ],
      "info": {
        "year": 2023,
        "version": "1.0",
        "description": "",
        "contributor": "Label Studio",
        "url": "",
        "date_created": "2023-04-11 19:02:12.209886"
      }
    }
    ```

### NEW Format - Yolov Training

```bash
.
├── annotations .             
│   └── instances_val2017.json
├── images                   
│   └── test            
│       └── image1.jpg
│       └── ...
│   └── train                   
│       └── image2.jpg
│       └── ...          
│   └── eval                   
│       └── image3.jpg
│       └── ...
├── labels                   
│   └── train                   
│       └── image2.txt
│       └── ...          
│   └── eval                   
│       └── image3.txt
│       └── ...
├── test.txt              <-- Reference to images path
├── train.txt            
├── val.txt        
```



## Usage
```
import FromCocoToYolov

coco_data_path = 'custom-datasets/coco-format'
yolov_data_path = 'custom-datasets/yolov-format'

converter = FromCocoToYolov(coco_data_path, yolov_data_path, logging.DEBUG)
converter.convert()
```


Examples:
 - https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset
