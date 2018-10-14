
# Install latest anaconda version from here: https://www.anaconda.com/download/#windows, I installed
# Anaconda3-5.3.0-Windows-x86_64
# I want to use the GPU version of tensorflow for speed because I have a compatible gpu (GeForce GTX 745)
# This card is one of the cuda supported gpu's listed here: https://developer.nvidia.com/cuda-gpus
# Install Cuda: https://developer.nvidia.com/cuda-toolkit I installed cuda_9.0.176_win10
# Then go to https://developer.nvidia.com/cudnn. You'll have to sign up for an NVidia developer 
# account first, but it's pretty quick and costs nothing. Once you've signed in you'll see a 
# variety of CUDNN downloads. Here's where you'll have to match to the version of CUDA that you 
# downloaded previously. So, for example, I used CUDA 9.0, so made sure I used a CuDNN that matches 
# both this and the required version you saw in the last step (in my case version 7), so I chose 
# cuDNN v7.3.1 for CUDA 9.0, specific for windows10, choosing cuDNN v7.3.1 Library for Windows 10
# specifically.
# Then since i wanted to do a conda install I used the following where conda is default
# In RStudio then install.library("keras")
# keras::install_keras(tensorflow = "gpu"), this installs both tensorflow and keras

# Next i ran the example from here https://keras.rstudio.com/ below
library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

summary(model)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
plot(history)

model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)

# Now I am ready to do deep learning in rstudio 