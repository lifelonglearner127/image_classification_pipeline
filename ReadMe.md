### Project Structure

- images: This folder contains demo dataset
  - training: This images are used to train the model
  - test: You can test model with these images
- config.py: You can configure settings on this file
- errors.py: Define project custom errors
- main.py: Main script
- utils.py: Define other utility functions & classes

### How to run the project

```
git clone https://github.com/lifelonglearner127/image_classification_pipeline.git
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Known Issues when installing & using virtualenv

- If you install Python package management tools(like virtualenv) on mac or ubuntu, those are installed in your home directory instead of system directory.
- Normally User level directories are not included in System PATH.
- So Please pay attention that you add your directory(contains virtualenv) to system path.

### Why HDF5?

As you know, while training we select a batch of images and use them to update the weights.
But images are encoded in image format (jpeg, png) so those should be decoded before feeding them to network.
Imagine the batch size is 8 and steps per epcho is 10 and the numer of epchos are 70.
Then you need to 5600 decode operation. That might be harmful to training time when you are trying to train large scale dataset.
So we store raw pixels data in HDF5 format and use them in traiinig. That will improve the training time. This is from my practical experiences.

### How to configure

> Please take a look at the `config.py` file

- BASE_NETWORKS_MAPPING:

  As you can imagine, this variable list all available ResNet architectures.

- NETWORK:

  The name of ResNet network that you can use as a base model. If you want to use `ResNet101` as a base model, you can set this variable to `ResNet101`.

- TRAINING_IMAGES_PATH:

  Specifiy the path of dataset. As you can see from the code, this dataset is splited into 3 subset, training, testing, validation. I skip what those dataset are because I think you have some knowledge in machine learning.

- TEST_IMAGES_PATH:

  Specifiy the path of test images. You could confirm trained model with your own eyes.

- TRAIN_HDF5:

  Specify the traing dataset HDF5 location.

- VAL_HDF5 :

  Specify the validation dataset HDF5 location.

- TEST_HDF5 :

  Specify the test dataset HDF5 location.

- MODEL_PATH :

  The path that trained model is saved to.

- OUTPUT_PATH :

  Specifiy the output path. You can see the checkpoint graph(traing graph) in this folder.

- EPOCHS :

  The number of epochs.

- BATCH_SIZE :

  The number of examples that are used in one SGD. You can set this value depending on your RAM or Video Card.

- CLASS_NUMS :

  The number of classies. I set this to 2 because I did dog & cat classification.
  If you want to do digit classification, you can set this value to 10.
