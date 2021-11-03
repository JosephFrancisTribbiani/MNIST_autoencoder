# MNIST_autoencoder
Архитектура автоэнкодера схематично ихображена на рисунке ниже. 
<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/architecture.png'></img>

На вход подается изображение из датасета MNIST размера (1, 28, 28). Более подробно архитектуру можно представить с помощью `torchsummary`:

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

На первой итерации модель обучалась на 80-ти эпохах с изменяемым `Learning Rate`.  
Графики изменения `Learning Rate` и `Loss` на тренировочной и тестовой выборках представлены ниже.

<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/param_1.png'></img>

Наименьшее значения Loss на тестовой выборке составило **0.0025443262532174873**

Далее модель была дообучена также на 80-ти эпохах с изменяемым `Learning Rate`. При этом, к тренировочной выборке была применена аугментация. А именно `RandomPerspective` с параметрами `distortion_scale=0.4` и `p=0.5`, а также `RandomRotation` с параметром `degrees=(-10, 10)`.  
Графики изменения `Learning Rate` и `Loss` на тренировочной и тестовой выборках представлены ниже.

<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/param_2.png'></img>

Наименьшее значения Loss на тестовой выборке составило **0.0023033211376672735**

### Результаты

На вход уже обученной сети были переданы 32 изображения из тестового датасета MNIST. На картинке ниже представлен выход сети. Первая строка - исходные изображения, вторая - реконструированные.

<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/reconstructed.png'></img>

Смещая латентное представление одной цифры к латентному представлению другой, можно получить промежуточные варианты. На изображениях ниже представлено несколько вариантов.
Первая строка - реконструированные изображения, вторая - исходные изображения.

<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/homotopy_1.png'></img>  
<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/homotopy_2.png'></img>  
<img src='https://github.com/JosephFrancisTribbiani/MNIST_autoencoder/blob/main/images/homotopy_3.png'></img>
