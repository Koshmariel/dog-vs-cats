Эта найросеть классифицирует фотографии собак и кошек (или любых других классов) из набора данных the Dogs vs. Cats  https://www.kaggle.com/c/dogs-vs-cats. Её точность около 96%. После обучения она может построить тепловую карту учатсков изображения, которые она считает важными при класификации.

Изображение из набора данных должны быть помещены в следующую структуру папок:

/dataset/train_set/class1

/dataset/train_set/class2

/dataset/test_set/class1

/dataset/test_set/class2

где “class1” и “class2” категории изображений.

Файлы:

dog-vs-cats.py – сама модель

heatmap.jpg – пример тепловой карты



This ANN classifies images of dogs and cats (or any other classes) from the Dogs vs. Cats dataset https://www.kaggle.com/c/dogs-vs-cats. It's accuracy is about 96%. After learning it can draw a heatmap to highlight the features which it considers important in the classification.

Dataset images should be placed into the following  folder structure:

/dataset/train_set/class1

/dataset/train_set/class2

/dataset/test_set/class1

/dataset/test_set/class2

where “class1” and “class2” are image classes.

Files:

dog-vs-cats.py – the model itself

heatmap.jpg – heatmap sample





