# TextSummarizer Using Huggingface

### Workflows


1. Config.yaml: Cada input de los modulos del proyecto se guardara en este archivo. Para cada modulo definimos los parámetros (Artifacts, Data Ingestion, Data Transformation, Model Trainer, etc).
    - Data Ingestion: source_URL_train, source_URL_test, source_URL_val, root_dir
2. Params.yaml: This file will contain all the hyperparameters related to the project. It will be used to store all the hyperparameters related to the project. It will be used to store all the hyperparameters related to the model training.
3. Config entity: This component will define the structure of the configuration files and provide a way to access the configuration parameters in a structured manner.
4. Configuration Manager: This component will manage the loading and updating of configuration files, ensuring that the correct configuration is used throughout the project.
5. Update components - Data Ingestion, Data Transformation, Model Trainer
6. Create pipeline -- Training Pipeline, Prediction Pipeline
7. Front end -- Api's, Training Api's, Batch Prediction Api's

