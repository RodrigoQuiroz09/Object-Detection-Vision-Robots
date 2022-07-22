# Object-Detection-Vision-Robots

**Descripción del preprocesamiento**

Cada uno de los modelos utilizados requerían un primer paso de preprocesamiento, el cual consistía en modificar las dimensiones del frame/imagen a ser analizado. A continuación, se exponen los tamaños utilizados por modelo:

- Caffe: 300 x 300 píxeles
  - _blob = cv2.dnn.blobFromImage(image=image, scalefactor=1, size=(300, 300), mean=(106, 117,123))_
- Tensor Flow: 300 x 300 píxeles
  - _blob = cv2.dnn.blobFromImage(image=res, size=(300, 300), mean=(106, 115, 124))_
- YOLO: 416 x 416 píxeles
  - _blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)_

Adicionalmente, se contemplaron dos otros pasos de preprocesamiento para las imágenes de los vídeos: la aplicación de un blur gaussiano y la ecualización de la imagen previo al intento de detección dentro de la misma.

Por un lado, se buscaba aplicar un blur gaussiano a las imágenes con el objetivo de eliminar ruido dentro de las imágenes. Lo anterior se planteaba debido a qué la calidad de las imágenes no era lo suficientemente buena o destacable. Se aplicó la siguiente línea de código, sin tener resultados qué beneficiarán el análisis y la detección de objetos.

_image = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)_

Por otro lado, también se consideró aplicar una ecualización a cada uno de los frames a analizar. Se buscaba transformar la imagen al sistema de colores YUB, para luego aplicar la ecualización de color sobre el canal de “Y”. Finalmente, se regresaría el frame bajo análisis a su sistema de colores originales, para luego ser usado para la detección de objetos. Similar al blur previamente descrito, tampoco se decidió agregar este paso del preprocesamiento al no encontrar ventajas aparentes en la precisión de los programas.

_img_y = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
img_y [:,:,0] = cv2.equalizeHist(img_y [:,:,0])
img_RGB = cv2.cvtColor(img_y , cv2.COLOR_YUV2Rgb)_

Se puede deducir que, debido a la necesidad de los modelos de recibir imágenes a color, la etapa de preprocesamiento se vería fuertemente limitada. Ésto como resultado qué diferentes tipos de filtros, operaciones morfológicas y otras posibilidades están orientadas hacia imágenes en escala de grises.

**Modelos utilizados**

**_Coffe + Coco SSD300_**: Se utilizó un modelo de “Single Shot MultiBox Detector”, con el cual se busca lograr la detección de objetos usando una sola red. Existen diversos modelos qué implementan esta metodología. Sin embargo, se decidió utilizar un modelo entrenado con COCO, capaz de detectar 90 tipos diferentes de objetos, en imágenes de 300 x 300.

- <https://github.com/weiliu89/caffe/tree/ssd>
- <https://drive.google.com/file/d/0BzKzrI_SkD1_NDlVeFJDc2tIU1k/view?resourcekey=0-VIwceFdQvGpMl31jHv5RpA>

**_YOLO versión 3_**: Utiliza una red neuronal convolucional para la detección de objetos en imágenes. De acuerdo a documentación del modelo, este modelo tiene niveles de precisión más elevados para la detección de objetos pequeños. Es capaz de detectar 90 tipos diferentes de objetos, en imágenes de 416 x 416.

- <https://github.com/AlexeyAB/darknet>
- <https://viso.ai/deep-learning/yolov3-overview/>

**_MobileNet-SSD and MobileNetV2 (TensorFlow)_**: Utiliza un conjunto de entrenamiento de COCO, similar a lo presentado en el modelo “Coffe + Coco SSD300”. Asimismo, también utiliza imágenes de 300 x 300 y es capaz de detectar 90 distintos tipos de objetos y elementos en un video.

- <https://github.com/Qengineering/MobileNet_SSD_OpenCV_TensorFlow>

**Uso del programa**

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.001.png)

_python detect_vid_tensorflow.py -i .\input\video_1.mp4 -o tensor_video1_35.mp4 -c .35_ \*

**-i simboliza en video a ser analizado, -o es el nombre/ruta del video resultado (default es output.mp4) y -c es el nivel de confianza para la detección (default es de 0.35 y el valor debe ser entre 0 y 1)**

## **Documentación de parámetros, pruebas y resultados**

**_Coffe + Coco SSD300_**

Se utilizaron 3 diferentes valores de confianza con este modelo, los cuales fueron 0.25, 0.35 y 0.65. Con estos mínimos de confianza para la detección de objetos, se obtuvieron los siguientes resultados:

|     COFFE VIDEO 1      |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|          Cake          | 154.49                         | 100                            | 78.96                          |
|         Carrot         | 8.42                           | 0.23                           | -                              |
|         Couch          | 14.39                          | -                              | -                              |
|         Person         | 100.38                         | 100                            | 76.52                          |
|       Surfboard        | 100.05                         | 96.05                          | 32.05                          |
|       Wine Glass       | 0.07                           | 0.01                           | -                              |

Se aprecia que para un 25% de confianza, se tienen elementos como “pastel” y “surfboard” que aparecen en más de una ocasión por frame y que no tienen razón de existir en la detección de este video. Sucede lo mismo con un 35% de confianza. Referente al “sillón” y a la “surfboard”, creemos qué fueron detectadas por tratarse de superficies “planas”, similares a la mesa del video. En torno al 65% de confianza, se reduce drásticamente la aparición de objetos no presentes en la escena; en los tres casos, la “persona” en el vídeo es ampliamente reconocida.

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.002.png)

_Figura 1. Fotografía de vídeo 1 con modelo de Coffe al 65%_

|     COFFE VIDEO 2      |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|     Baseball glove     | 67.31                          | 2.43                           | -                              |
|          Cake          | 100.36                         | 86.6                           | 4.57                           |
|         Couch          | 36.33                          | 3.84                           | -                              |
|          Desk          | 0.16                           | -                              | -                              |
|      Eye glasses       | 0.59                           | 0.3                            | -                              |
|        Hot dog         | 22.04                          | 9.79                           | -                              |
|         Laptop         | 39.03                          | 3.19                           | -                              |
|         Mirror         | 0.07                           | -                              | -                              |
|         Person         | 117.64                         | 117.05                         | 112.68                         |
|      Sports ball       | 0.13                           | -                              | -                              |
|       Wine glass       | 100                            | 100                            | 5.81                           |

Se aprecia que para un 25% de confianza, se tienen elementos como “pastel”, “sillón” y “baseball glove” con presencia de identificación pero que son considerados como “falsos negativos”; la detección de “desk” es asumible y se puede confundir con la mesa del video. Sucede lo mismo con un 35% de confianza, pero en menor medida, debido a qué un mayor nivel de confianza reduce las detecciones incorrectas. Referente al “sillón” y a la “surfboard”, creemos qué fueron detectadas por tratarse de superficies “planas”, similares a la mesa del video. En torno al 65% de confianza, se reduce drásticamente la aparición de objetos no presentes en la escena. Debido a qué en el video aparecen dos personas, esto genera un % de aparición de personas superior al 100%

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.003.png)

_Figura 2. Fotografía de vídeo 2 con modelo de Coffe al 35%_

|     COFFE VIDEO 3      |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|          Bed           | 1.08                           | -                              | -                              |
|          Cake          | 209.6                          | 102.4                          | 15.2                           |
|         Carrot         | 0.8                            | -                              | -                              |
|         Couch          | 0.4                            | -                              | -                              |
|      Dining table      | 4                              | 0.4                            | -                              |
|         Mouse          | 7.87                           | 1.87                           | -                              |
|         Person         | 100                            | 100                            | 100                            |
|      Potted plant      | 98.4                           | 91.73                          | 71.33                          |
|      Sports Ball       | 2.8                            | 2                              | 0.8                            |

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.004.png)

_Figura 3. Fotografía de vídeo 3 con modelo de Coffe al 65%_

Para los tres niveles de confianza, se aprecia que “potted plant” tiene altos niveles de identificación. No obstante, no se encuentra este tipo de objeto en la escena; en cambio, el modelo confunde una laptop con una planta. Sigue habiendo confusión con las superficies planas (pastel VS cámara VS “mesa de comida”). Al 65% de confianza, se reducen drásticamente los falsos positivos pero siguen existiendo.

**_YOLO versión 3_**

Se utilizaron 3 diferentes valores de confianza con este modelo, los cuales fueron 0.5, 0.75 y 0.90. Con estos mínimos de confianza para la detección de objetos, se obtuvieron los siguientes resultados:

|      YOLO VIDEO 1      |                               |                                |                               |
| :--------------------: | ----------------------------- | ------------------------------ | ----------------------------- |
| Categoría identificada | Confianza en .5 (% aparición) | Confianza en .75 (% aparición) | Confianza en .9 (% aparición) |
|         Person         | 100                           | 100                            | 92.48                         |
|     Baseball glove     | 1.03                          | 0.05                           | -                             |
|       Surfboard        | 167.77                        | 150.05                         | 102.26                        |
|         Plate          | 0.02                          | -                              | -                             |
|       Wine glass       | 89.48                         | 42.87                          | 12.58                         |
|         Carrot         | 0.06                          | -                              | -                             |
|          Cake          | 94.48                         | 36.52                          | 10.05                         |
|         Laptop         | 104.7                         | 97.93                          | 84.53                         |

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.005.png)

_Figura 4. Fotografía de vídeo 1 con modelo de YOLO al 75%_

Si bien es cierto qué los niveles de confianza del modelo de YOLO son más elevados que los otros dos, presenta comportamientos muy similares respecto a niveles de confianza variados. Con el más bajo (50%), se tiene un 100% de detección perfecta de la persona en el vídeo; hay diferentes instancias de falsos positivos, como lo son la zanahoria, el “surfboard” y la computadora. El comportamiento con el 75% de confianza mejora, ya qué se reducen las apariciones de falsas detecciones. No obstante, el 90% sin duda muestra el mejor balance entre la detección del señor y la reducción de falsos positivos.

|      YOLO VIDEO 2      |                               |                                |                               |
| :--------------------: | ----------------------------- | ------------------------------ | ----------------------------- |
| Categoría identificada | Confianza en .5 (% aparición) | Confianza en .75 (% aparición) | Confianza en .9 (% aparición) |
|         Person         | 117.58                        | 117.02                         | 115.14                        |
|     Baseball glove     | 64.95                         | 57.85                          | 42.35                         |
|       Surfboard        | 86.86                         | 22.04                          | 6.08                          |
|       Wine glass       | 51.81                         | 2.17                           | 0.16                          |
|          Fork          | 192.21                        | 96.19                          | 13.21                         |
|          Cake          | 50.53                         | 10.87                          | 01.02                         |
|         Mirror         | 0.2                           | 0.13                           | -                             |
|         Window         | 4.43                          | 0.07                           | -                             |
|         Laptop         | 89.78                         | 27.37                          | 2.63                          |

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.006.png)

_Figura 5. Fotografía de vídeo 2 con modelo de YOLO al 50%. Presenta falsos positivos_

Para el segundo video con este modelo, se presenta un comportamiento muy similar al primero. La detección de personas es correcta con los tres niveles de confianza. No obstante, tanto para el 50% como para el 75% se obtienen falsos positivos como el “surfboard”, la ventana, el espejo y los pasteles; se vuelve a tener el problema con objetos de clases con “superficies planas”.

|      YOLO VIDEO 3      |                               |                                |                               |
| :--------------------: | ----------------------------- | ------------------------------ | ----------------------------- |
| Categoría identificada | Confianza en .5 (% aparición) | Confianza en .75 (% aparición) | Confianza en .9 (% aparición) |
|         Person         | 100                           | 100                            | 99.73                         |
|     Baseball glove     | 63.07                         | -                              | -                             |
|       Surfboard        | 89.73                         | 31.87                          | 0.13                          |
|       Wine glass       | 29.07                         | -                              | -                             |
|      Potted plant      | 96.4                          | 90.4                           | 82.27                         |
|          Bed           | 29.73                         | 0.93                           | -                             |
|      Dining table      | 0.4                           | 0.13                           | -                             |
|         Window         | 5.47                          | -                              | -                             |
|         Laptop         | 140.8                         | 58.8                           | 9.07                          |

Para el tercer video, la detección de la computadora/laptop en el video es ideal en el análisis con un 75% de precisión; logra un balance entre reducción de falsos positivos y la presencia de la laptop. La señora del video es detectada, en los tres casos, en casi el 100% de las instancias. El 50% permite un alto nivel de falsos positivos; tanto el 75% como el 90% realizan una disminución considerable de los mismos.

![](/images/Aspose.Words.43b40ca0-bf9d-4c9e-add5-1d0f0b40a54f.007.png)

_Figura 6. Fotografía de vídeo 3 con modelo de YOLO al 75%. Presenta falsos positivos (planta en vez de laptop)_

**_MobileNet-SSD and MobileNetV2 (TensorFlow)_**

Se utilizaron 3 diferentes valores de confianza con este modelo, los cuales fueron 0.25, 0.35 y 0.65. Con estos mínimos de confianza para la detección de objetos, se obtuvieron los siguientes resultados:

|     TENSOR VIDEO 1     |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|        Backpack        | 0.07                           | -                              | -                              |
|          Book          | 0.06                           | -                              | -                              |
|         Bottle         | 2.83                           | 0.03                           | -                              |
|         Chair          | 0.08                           | -                              | -                              |
|         Clock          | 33.05                          | 3.62                           | -                              |
|         Couch          | 3.72                           | 0.45                           | 0.02                           |
|          Cup           | 0.53                           | -                              | -                              |
|      Dining table      | 129.7                          | 88.51                          | 43.34                          |
|         Mouse          | 0.16                           | -                              | -                              |
|         Person         | 87.71                          | 46.55                          | 17.52                          |
|      Refrigerator      | 0.1                            | -                              | -                              |
|       Teddy Bear       | 0.08                           | 0.02                           | -                              |

El modelo de Tensorflow es aquel qué registra el mayor número de clases con precisiones de confianza entre 25% y 35%; hay diferentes instancias de falsos positivos para ambos casos (backpack, book, cup, clock y teddy bear, por mencionar algunas). No obstante, en la mayoría de los casos, sus porcentajes de aparición son bajos. Adicionalmente, la detección de la persona y de la mesa son altas y se deben resaltar. El 65% de confianza muestra una presencia cercana a nula de falsos positivos, pero sacrifica la precisión con objetos que si se encuentran en la escena.

![](/images/Aspose.Words.08ebcc1c-707a-489a-8fe2-019f3043032f.001.png)

_Figura 7. Fotografía de vídeo 1 con modelo de Tensor al 65%._

|     TENSOR VIDEO 2     |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|        Backpack        | 0.49                           | 0.1                            | -                              |
|          Bed           | 0.2                            | -                              | -                              |
|        Bicycle         | 2.73                           | 0.95                           | -                              |
|         Bottle         | 0.46                           | -                              | -                              |
|          Bowl          | 0.03                           | -                              | -                              |
|          Cat           | 0.03                           | -                              | -                              |
|         Chair          | 02.04                          | 0.13                           | -                              |
|         Couch          | 77.53                          | 10.09                          | 0.07                           |
|          Dog           | 0.03                           | 0.03                           | -                              |
|         Laptop         | 8.54                           | 0.39                           | -                              |
|         Person         | 114.78                         | 90.87                          | 39.26                          |
|      Refrigerator      | 8.57                           | -                              | -                              |
|         Remote         | 0.79                           | -                              | -                              |
|           Tv           | 0.1                            | -                              | -                              |

![](/images/Aspose.Words.506c9ee8-ea85-43cd-b6c2-a70c1d28d055.001.png)

_Figura 8. Fotografía de vídeo 2 con modelo de Tensor al 35%. Presenta un Falso Positivo (Bicicleta)_

La detección de un sillón (falso positivo) es notable para los niveles de confianza de 25% y 35%. Lo anterior se puede deber a la tonalidad tan obscura de la mesa, que no permite al modelo identificar la mesa de manera correcta. Adicionalmente, en el 25% hay varias clases que se detectan en un puñado de frames solamente (tv, control remoto, perro). Lo anterior se da particularmente cuando la segunda chica entra en cámara y parece haber un movimiento de sombras en el frame. El 65% de confianza evidencia su capacidad de filtro de falsos positivos pero limita la precisión con la detección de personas.

|     TENSOR VIDEO 3     |                                |                                |                                |
| :--------------------: | ------------------------------ | ------------------------------ | ------------------------------ |
| Categoría identificada | Confianza en .25 (% aparición) | Confianza en .35 (% aparición) | Confianza en .65 (% aparición) |
|        Backpack        | 1.2                            | -                              | -                              |
|          Bed           | 17.47                          | 9.33                           | 0.53                           |
|         Chair          | 0.13                           | -                              | -                              |
|         Clock          | 91.87                          | 45.33                          | 13.07                          |
|         Couch          | 32.53                          | 14.53                          | 0.27                           |
|      Dining table      | 39.2                           | 15.2                           | -                              |
|        Keyboard        | 17.47                          | -                              | -                              |
|         Laptop         | 52.13                          | 9.2                            | -                              |
|         Person         | 105.07                         | 97.73                          | 76.4                           |
|      Refrigerator      | 11.6                           | 0.13                           | -                              |
|        Scissors        | 45.73                          | -                              | -                              |
|       Teddy bear       | 0.13                           | 0.13                           | -                              |
|           Tv           | 0.13                           | -                              | -                              |

![](/images/Aspose.Words.506c9ee8-ea85-43cd-b6c2-a70c1d28d055.002.png)

_Figura 9. Fotografía de vídeo 3 con modelo de Tensor al 35%. Presenta un Falso Positivo (Reloj)_

Finalmente, se aprecia qué el 25% continúa con la tendencia de clases con muy bajos porcentajes de falsos positivos. Adicionalmente, se registra un refrigerador (se confunde con la puerta del archivero por el patrón de los cajones y la similitud con este electrodoméstico). La detección de tijeras en el 25% se deriva de lo que parece un “tripie” blanco en la parte trasera izquierda de la escena. Similar al refrigerador, el modelo distingue una forma muy similar y por eso se atreve a categorizar el objeto. En cambio, el 35% y el 65% no logran identificar objetos clave en la escena, como es el caso de la laptop, el teclado de la misma y la mesa (en menor medida).

**Mejores modelos por video**

_Video 1_: YOLO con confianza de 90% O Tensorflow con confianza de 35%

_Video 2_: Tensorflow con confianza de 35%

_Video 3_: Coffe con confianza de 65% O Tensorflow con confianza de 35%

**Conclusiones**

- La calidad de los vídeos es deficiente, lo que genera que los modelos de detección de objetos tengan menos precisión y eficacia.
- La limitante de sólo poder trabajar con imágenes a color limita las opciones de preprocesamiento y mejora de los frames a analizar. Lo anterior podría mejorar si fuera posible manipular y alimentar a las arquitecturas de imágenes en escala de grises.
- La posición de la cámara, al igual qué de los objetos, dificultan la correcta detección de objetos.
  - La mesa de los videos es visible, pero solamente la parte plana de la misma. Se esperaría que se vieran las patas de la misma para que los modelos tengan mejor precisión con este tipo de objetos.
  - Sucede algo similar con la silla en la qué están sentadas las personas en los videos. Son visibles en un porcentaje cercano a cero en esta triada de videos.
- La detección de falsos positivos ocurren por diferentes causas:
  - Hay objetos, como lo son camas, mesas, sillones, tablas de surf, entre otras, qué presentan segmentos planos muy similares, por lo que los modelos tienen dificultades para reconocer objetos entre estos tipos de clases.
  - Hay otros elementos, como las tijeras, bicicletas y relojes, que aparecen por la similitud de una especie de trípode en la escena. La forma de este objeto es similar a las clases previamente mencionadas, por lo que se entiende la confusión de los modelos.
  - Objetos como puertas, ventanas, refrigeradores y espejos también pueden causar problemáticas, debido a que comparten una característica: tener una forma rectangular en la mayoría de los casos.
  - La poca calidad de las imágenes acentúan los problemas planteados en esta serie de bullets en particular.
- Coffe muestra el desempeño con menos eficacia y capacidad de una correcta detección de objetos en los vídeos. Se concluye que esto se debe a que Coffe fue planteado como un modelo orientado, originalmente, hacia la clasificación de imágenes y no para la detección de objetos.
- Tensorflow y YOLO presentan comportamientos similares, en gran medida, por las clases con las qué fueron entrenados. Asimismo, son modelos qué llevan años innovando y mejorando en la detección de decenas de clases de objetos.
- La mejora en la iluminación de la escena del vídeo, al igual qué el incremento de calidad de la cámara o instrumento utilizado para la grabación del mismo, aseguraría una mejor detección de objetos.
- Probar con diferentes modelos de Tensorflow, YOLO y Caffe, que tengan variedad en sus arquitecturas, tipos de clases usadas para el entrenamiento, y tamaño de las imágenes, sería determinante para encontrar el modelo ideal para la detección de objetos en éste y otros conjuntos de videos.
- Otra posibilidad sería el poder entrenar los modelos con frames de videos similares a los presentados en este proyecto. Se tendría mayor fiabilidad para detectar objetos bajo esta lista de condiciones (calidad de la imagen, luminosidad, etc.).
