
# Learning from Images: Alzheimer's Classifier

## Introduction
This repository dedicated to building and evaluating an Alzheimer's disease image classifier. The repo encompasses all steps from initial data handling to advanced model evaluation, demonstrating a complete workflow in medical image analysis using deep learning.

## Detailed Command Line Operations:

1. **Cloning Repository**: 
   - Command: `!git clone https://github.com/atikul-islam-sajib/GoodPractiseDSID.git`

2. **Setting Working Directory**: 
   - Command: `%cd /content/GoodPractiseDSID`

3. **Prerequisites**

For optimal utilization of this repo, the following are required:

- **Python Version**: Python 3.9 or higher.
- **Execution Requirements**: pip install -r requirements.txt.
- **Hardware Requirement**: Access to GPU/MPS resources is recommended for efficient model training and evaluation.

4. **Training the Classifier**: 
   - Command: `!python alzheimer/classifier/classifier.py --dataset /content/dataset.zip --batch_size 64 --model --epochs 500 --lr 0.001 --device gpu`
   
5.  **Training the Classifier with augmentation samples**: 
   - Command: `!python alzheimer/classifier/classifier.py --dataset /content/dataset.zip --augmentation 1000 --batch_size 64 --model --epochs 500 --lr 0.001 --device gpu`

6. **Generating Metrics**: 
   - Command: `!python alzheimer/classifier/classifier.py --get_all_metrics --device gpu`

7. **Creating Charts**: 
   - Command: `!python alzheimer/classifier/classifier.py --get_all_charts`

8. **Displaying Results**: 
   - Commands:
     - `Image('/content/GoodPractiseDSID/alzheimer/figures/image_prediction.png',)`
     - `Image('/content/GoodPractiseDSID/alzheimer/figures/training_history.png')`
     - `Image('/content/GoodPractiseDSID/alzheimer/figures/confusion_metrics.png')`

### Detailed Implementation Steps: Importing modules

1. **Importing Modules for Alzheimer's Analysis**:
   - Code:
     ```python
     from alzheimer.data.data_loader import Dataloader
     from alzheimer.features.build_features import FeatureBuilder
     from alzheimer.models.train_model import Trainer
     from alzheimer.models.model import Classifier
     from alzheimer.visualization.visualize import ChartManager
     ```

2. **Unzipping the Dataset**:
   - Code:
     ```python
     loader = Dataloader(zip_file='/content/dataset.zip')
     loader.unzip_dataset()
     ```
3. **Feature Creation**:
   - Code:
     ```python
     build_features = FeatureBuilder()
     build_features.build_feature()
     loader.extract_feature()
     ```
4. **Model Initialization**:
   - Code with GPU - CUDA:
     ```python
     import torch
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     clf = Classifier()
     model_trainer = Trainer(classifier=clf, device=device, lr=0.001)
     ```

   - Code with MAC MPS:
     ```python
     import torch
     device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
     clf = Classifier()
     model_trainer = Trainer(classifier=clf, device=device, lr=0.001)
5. **Training the Model**:
   - Code:
     ```python
     model_trainer.train(epochs=100)
     ```
6. **Model Performance Evaluation**:
   - Code:
     ```python
     model_evaluation, model_clf_report = model_trainer.model_performance()
     print(model_evaluation)
     print(model_clf_report)
     ```
7. **Visualization with ChartManager**:
   - Code:
     ```python
     charts = ChartManager()
     charts.plot_image_predictions()
     charts.plot_training_history()
     charts.plot_confusion_metrics()
     ```
