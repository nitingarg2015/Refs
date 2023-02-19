## Repository contains models and functions that can be used across datasets

### Models  
Folder contains the following models: ResNet18, ResNet34  
ResNet18 and ResNet34 expect input tensor of shape (B, C, H, W)  
Custom ResNet model with modular architecture as per requirements of Assignment 8
Other models will be added in future  

### Utils  
Utils folder contains CIFARdata, gradcam, and utils files.

**gradcam** - Contains function **plot_grad_images** for plotting gradCAM images

**utils** - Contains following functions:  
*get_mis_classified_byloader(model, device, data_loader)* - accepts input as data loader
*get_miss_classified_byimages(model, device, images, labels)* - accepts images as a tensor of shape (B, C, H, W) 
*getFractionsMissed* - to retrieve fractions missed by each class in the dataset
*plot_misclassified(image_data, targeted_labels, predicted_labels, classes, no_images)* - to plot mis classified images
*plot_LossAndAcc* - plot train/test loss and accuracies
*get_mean_and_std* - returns mean and std of the dataset
*denormalize(image)* - to denormalize an image
*imshow(img)* - to plot an image  
*find_lr(model, optimizer, criterion, data_loader, device, end_lr, num_iter) - to find learning rate using one cycle policy  

### CIFARdata
functions for loading CIFAR data has been defined. This uses Albumentations package if transformations are defined using the same  
*load_CIFAR_dataloader(train, transform, batch_size=128)* function takes the following arguments:
 train - True if for training data, False for test data
 transform - True - if transformations other than normalization to be applied, False - otherwise
 batch_size - for batch size of data loaders
 
 Returns data loader as per requirements

## main  
main_ass7 - contains functions to train and test the model for assignment 7
main_ass8 - contains functions to train and test the model for assignment 8
