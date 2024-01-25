1. **Training the Classifier**: 
   - Command: `!python alzheimer/classifier/classifier.py --dataset /content/dataset.zip --batch_size 64 --model --epochs 500 --lr 0.001 --device gpu`
   
2.  **Training the Classifier with augmentation samples**: 
   - Command: `!python alzheimer/classifier/classifier.py --dataset /content/dataset.zip --augmentation 1000 --batch_size 64 --model --epochs 500 --lr 0.001 --device gpu`

3. **Generating Metrics**: 
   - Command: `!python alzheimer/classifier/classifier.py --get_all_metrics --device gpu`

4. **Creating Charts**: 
   - Command: `!python alzheimer/classifier/classifier.py --get_all_charts`

5. **Displaying Results**: 
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
   - Code with GPU : CUDA:
     ```python
     import torch
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     clf = Classifier()
     model_trainer = Trainer(classifier=clf, device=device, lr=0.001)
     ```

   - Code with MAC : MPS:
     ```python
     import torch
     device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
     clf = Classifier()
     model_trainer = Trainer(classifier=clf, device=device, lr=0.001)
     ```
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
