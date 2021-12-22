Use the main.py file to train and test the images.

The training, and test data load is done on the CIFAR 10 dataset mentioned in data_dir variable

For training the dataset:
Run the model.train method in the main.py comment the other methods which train, evaluate or predict.
model.train(x_train, y_train, number_of_epochs_to_run_for, resume_epoch_from)
x_train - Dataset of shape [N, 3072] - Contains images used for training
y_train - Dataset of shape [N,] - Contains labes for the images used for training
number_of_epochs_to_run_for - Runs the training for the given epochs.
resume_epoch_from - 0 if training starts from beginning, k if training starts from kth epoch - kth checkpoint (model-k.ckpt)
		in current directory is used to resume training.
Saves model checkpoints for testing accuracy or resume training from kth epoch. 

For testing the accuracy of dataset:
Run the model.evaluate method in and comment the other methods which train, evaluate or predict.
model.evaluate(x_test, y_test, list_of_ckpts_to_test_against)
x_train - Dataset of shape [M, 3072] - Contains images used for testing
y_train - Dataset of shape [M,] - Contains labes for the images used for testing
list_of_ckpts_to_test_against - List of checkpoint nums that will be used for testing at the kth checkpoint (model-k.ckpt)
			in current directory is used for testing.


