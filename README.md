
# Palm Tree Detection using YOLOv8

This project detects and counts **palm trees** from aerial imagery using the YOLOv8 object detection model. It includes data preprocessing, annotation conversion, model training, visualization, and inference steps — all streamlined for research or production use.

# Project Structure

```
palm-tree-detection/
├── project/
│   ├── train_labels.csv        # Annotated training data
│   ├── test_labels.csv         # Annotated test data
│   ├── train_images/           # Raw training images
│   └── test/                   # Raw test images
├── yolo_dataset/
│   ├── images/train/           # Processed YOLO training images
│   ├── images/test/            # Processed YOLO test images
│   ├── labels/train/           # YOLO label .txt files for training
│   ├── labels/test/            # YOLO label .txt files for testing
│   └── dataset.yaml            # YOLOv8 dataset config
├── runs/                       # YOLOv8 training/evaluation results
├── visualized/                 # Images with bounding boxes drawn
├── yolov8_palm_count.py        # Script to count detected palms
├── visualize_predictions.py    # Script to visualize results
├── preprocess_to_yolo.py       # CSV to YOLO format + organization
└── README.md                   # This file
```

## Features

- Converts CSV annotations to YOLO format
- Organizes training and testing sets
- Trains YOLOv8 model using Ultralytics
- Visualizes ground truth and predicted bounding boxes
- Counts number of palm trees detected in an image

## Requirements

Install required packages:

```bash
pip install ultralytics opencv-python matplotlib pandas scikit-learn PyYAML
```

## Dataset Format

Annotation CSV files contain the following columns:

- `filename`, `width`, `height`
- `xmin`, `ymin`, `xmax`, `ymax`
- `class` (`Palm` or `Tree`)

## How to Use

### 1. Preprocess and Convert Annotations

```bash
python preprocess_to_yolo.py
```

This creates a YOLO-compatible dataset structure and generates the `dataset.yaml` file.

### 2. Train the Model

```bash
yolo detect train data=yolo_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

You can replace `yolov8n.pt` with any YOLOv8 variant (e.g., `yolov8m.pt`, `yolov8x.pt`).

### 3.Evaluate the Model

After training, evaluation results are saved in `runs/detect/val/`.

You can visualize results in a Jupyter notebook:

```python
from visualize_predictions import show_image

show_image("runs/detect/val/PR_curve.png")
show_image("runs/detect/val/F1_curve.png")
show_image("runs/detect/val/confusion_matrix.png")
```

### 4.Visualize Predictions on Images

```bash
python visualize_predictions.py
```

Outputs images with bounding boxes to the `visualized/` folder.

### 5.Count Detected Palm Trees

```bash
python yolov8_palm_count.py --input path/to/image_or_folder
```

This will output the number of palm trees detected per image.

## Sample Output

![Sample Output](runs/detect/predict/ck2gi1goojxa10794p17jnhw7.jpg)

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV and Matplotlib for visualization
- Kwame Nkrumah University of Science and Technology, Department of Geomatic Engineering

## Author

Emmanuel Antwi 
GitHub: [@eantwi56](https://github.com/eantwi56)

## License

This project is licensed under the MIT License.

