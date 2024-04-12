from ultralytics import YOLO

if __name__ == "__main__":

    # Load a model
    # model = YOLO('yolov8n.yaml')  # If we want: build a new model from YAML
    # model = YOLO('yolov8n.pt')    # If we want: load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # If we want: build from YAML and transfer weights
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
   
    # Train the model
    results = model.train(
        # address of the dataset (we can use ready datasets or made own dataset using roboflow.com)
        # in roboflow.com we can define dataset is for segmentation or detection
        data='MyOwnDataset\\handwriting.v6i.yolov8\\data.yaml',
        
        # for public datasets
        # data='coco128.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        #batch=-1
        )