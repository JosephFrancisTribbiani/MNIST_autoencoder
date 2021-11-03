# MNIST_autoencoder
Архитектура автоэнкодера схематично ихображена на рисунке ниже. 
<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/architecture.png'></img>

На вход подается изображение из датасета MNIST размера (1, 28, 28). Более подробно архитектуру можно представить с помощью torchsummary:

|Layer (type)|Output Shape|Param|
|---|---|---|
|Conv2d-1|[-1, 16, 28, 28]|160|
|BatchNorm2d-2|[-1, 16, 28, 28]|32|
|MaxPool2d-3|[-1, 16, 14, 14]|0|
|ELU-4|[-1, 16, 14, 14]|0|
|Conv2d-5|[-1, 32, 12, 12]|4,640|
|BatchNorm2d-6|[-1, 32, 12, 12]|64|
|MaxPool2d-7|[-1, 32, 6, 6]|0|
|ELU-8|[-1, 32, 6, 6]|0|
|Conv2d-9|[-1, 64, 6, 6]|18,496|
|BatchNorm2d-10|[-1, 64, 6, 6]|128|
|MaxPool2d-11|[-1, 64, 3, 3]|0|
|ELU-12|[-1, 64, 3, 3]|0|
|Conv2d-13|[-1, 128, 1, 1]|73,856|
|Flatten-14|[-1, 128]|0|
|Unflatten-15|[-1, 128, 1, 1]|0|
|ConvTranspose2d-16|[-1, 64, 3, 3]|73,792|
|Upsample-17|[-1, 64, 6, 6]|0|
|ConvTranspose2d-18|[-1, 32, 6, 6]|18,464|
|BatchNorm2d-19|[-1, 32, 6, 6]|64|
|ELU-20|[-1, 32, 6, 6]|0|
|Upsample-21|[-1, 32, 12, 12]|0|
|ConvTranspose2d-22|[-1, 16, 14, 14]|4,624|
|BatchNorm2d-23|[-1, 16, 14, 14]|32|
|ELU-24|[-1, 16, 14, 14]|0|
|Upsample-25|[-1, 16, 28, 28]|0|
|ConvTranspose2d-26|[-1, 1, 28, 28]|145|
|BatchNorm2d-27|[-1, 1, 28, 28]|2|
|ELU-28|[-1, 1, 28, 28]|0|

Total params: 194,499  
Trainable params: 194,499  
Non-trainable params: 0  

