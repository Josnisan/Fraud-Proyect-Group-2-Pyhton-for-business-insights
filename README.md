# Detección de fraudes por medio de machine learning

En este proyecto abordamos un problema real y costoso: el fraude en reclamos de seguros de autos. Nuestro objetivo es desarrollar un modelo de Machine Learning capaz de identificar automáticamente reclamos potencialmente fraudulentos, ayudando así a las aseguradoras a reducir pérdidas, optimizar procesos y proteger a los clientes legítimos.
Utilizamos técnicas de clasificación supervisada y procesamiento de datos para construir una solución eficiente, escalable y alineada con casos de uso reales en la industria aseguradora.

![image](https://github.com/user-attachments/assets/d6a09e13-142f-4679-9fb0-f922eb04ab71)

# El problema

El fraude en los seguros de autos representa un problema crítico para la industria. Solo en EE. UU., se estima que las aseguradoras pierden al menos $29 mil millones al año, según un estudio de Verisk (2017). Este tipo de fraude puede presentarse de distintas formas, como reclamos inflados, choques simulados o documentación falsa.

Estas prácticas no solo afectan las finanzas de las aseguradoras, sino también tienen un impacto directo en los usuarios legítimos: las primas aumentan y la rentabilidad general del sistema se reduce.

El mayor desafío es que la detección tradicional, manual, resulta costosa y lenta, especialmente frente a grandes volúmenes de datos. Por eso, es clave incorporar soluciones automáticas y escalables como el Machine Learning para enfrentar este problema.

![image](https://github.com/user-attachments/assets/2fb51e42-6b53-4525-ae3f-c030c2b95254)

# Nuestra solución

Para enfrentar el reto del fraude en reclamos de seguros, proponemos una solución basada en aprendizaje automático supervisado. Específicamente, implementamos un modelo de clasificación binaria que predice si un reclamo es fraudulento o no.

Este modelo aprende a partir de datos históricos previamente etiquetados (variable objetivo: FraudFound_P), lo que le permite identificar patrones comunes en los fraudes y aplicarlos a nuevos casos de forma automática.

Nuestro enfoque busca minimizar los falsos negativos, es decir, los fraudes que no son detectados, pero sin castigar a los usuarios legítimos. Así, logramos mantener un equilibrio entre precisión, eficiencia operativa y experiencia del cliente.

![image](https://github.com/user-attachments/assets/f31bcbed-abed-4c6f-927c-0cafba228d57)

# Metodología

# 1. Análisis exploratorio 
Comenzamos nuestro proyecto realizando un análisis exploratorio del conjunto de datos, el cual contiene 15,420 registros y 33 columna que describen distintos aspectos de cada reclamo.
Confirmamos que no existen valores nulos, lo cual facilita el procesamiento inicial. Las variables fueron categorizadas en distintos grupos para entender mejor su estructura y posibles relaciones con el fraude:

1 variable target (FraudFound_P)
9 temporales
4 demográficas 
2 geográficas 
5 vehiculares 
6 relacionadas a seguros/servicios
6 de circunstancias del accidente e historial

![image](https://github.com/user-attachments/assets/37d4929f-3acd-4040-ac3b-b48bd24e09f3)

Al analizar la distribución de la variable target, observamos un fuerte desbalance de clases:
94.01% de los reclamos están etiquetados como "No fraude
Solo 5.98% corresponden a casos "fraudulentos"
Esto significa que, aunque el dataset sea amplio, los ejemplos positivos (fraudes reales) son muy pocos. Este desbalance representa un reto importante en el modelado, ya que los algoritmos tienden a favorecer la clase mayoritaria.
Por esta razón, será necesario aplicar técnicas de balanceo de clases, para asegurar que el modelo aprenda a detectar correctamente los fraudes sin quedar sesgado.

![image](https://github.com/user-attachments/assets/bdb6f727-99a1-4179-89ce-df85eb82a8a0)

Al comparar variables del dataset con FraudFound_P, identificamos patrones interesantes:
Mes del reclamo (Month) vs Fraude:
 Enero destaca como el mes con mayor cantidad de reclamos fraudulentos, lo que sugiere una posible estacionalidad en el comportamiento del fraude.
Marca del vehículo (Make) vs Fraude:
 La marca Pontiac aparece con mayor frecuencia entre los casos de fraude. Esto podría indicar una correlación fuerte que afecte la predicción, por lo que se debe analizar si este peso es real o producto de un sesgo en los datos.

![image](https://github.com/user-attachments/assets/8012fcf2-ca4b-4401-ad06-e74cfa8636c2)


Continuamos analizando variables que pueden influir en la ocurrencia de fraudes:
Área del accidente (AccidentArea):
 La mayoría de los accidentes registrados en el dataset ocurrieron en zonas urbanas, lo que sugiere un sesgo hacia regiones más pobladas. Esto puede reflejar tanto la densidad vehicular como la frecuencia de reclamos.
Sexo del conductor (Sex):
 Observamos que la mayoría de los accidentes están relacionados con conductores hombres. Este tipo de patrón puede ser relevante al evaluar el riesgo, pero también debe manejarse con cuidado para evitar conclusiones sesgadas o discriminatorias en el modelo.

![image](https://github.com/user-attachments/assets/adde51b7-ec75-443d-8784-65012cc3a0e7)

En general, no se detecta multicolinealidad severa en el conjunto de datos: la mayoría de las correlaciones son bajas o cercanas a cero.
Correlaciones destacadas:
Existe una fuerte correlación positiva (0.94) entre PolicyNumber y Year, lo cual sugiere que podrían estar representando información redundante o relacionada temporalmente.

Una moderada correlación (0.28) entre WeekOfMonth y WeekOfMonthClaimed, lógica por la cercanía temporal entre la fecha de ocurrencia y la fecha de reclamo.
 Conclusión:
Variables como PolicyNumber y Year requieren un análisis más profundo para decidir si deben ser transformadas, combinadas o eliminadas, dependiendo de su valor predictivo y el riesgo de sobreajuste.

El resto de las variables numéricas no presenta problemas serios de correlación que puedan afectar negativamente el modelado.

![image](https://github.com/user-attachments/assets/76f9db69-3e29-48ef-b922-205f0963ffb5)

# 2. Limpieza de datos
Durante el proceso de limpieza de datos, se abordaron varios problemas para mejorar el rendimiento del modelo:
Se eliminaron los registros con valores faltantes en "Age" y se redujo la multicolinealidad para evitar su impacto negativo.
Las variables "PolicyNumber" y "RepNumber", al ser identificadoras, no aportan a la predicción, por lo que se eliminaron.
También se descartaron "Month" y "Year", ya que no son relevantes para la detección de fraude.

![image](https://github.com/user-attachments/assets/ea7ed4fb-caf5-48fc-86cd-664773e111be)


# 3. Separación del data set 

Para poder empezar a tomar desiciones con respecto a la modelación de los datos primero se debe partir el data set, esto con la finalidad de tomar 3 conjuntos de datos: el data set de entrenamiento, con el cual se harán las pruebas iniciales, el data set de validación, que permitirá evaluar las métricas del modelo, y finalmente el data set de test, con el cual se realizará la prueba final para identificar si el modelo es efectivo para la detección de fraudes. 

![image](https://github.com/user-attachments/assets/8950c5f9-243a-469c-a25e-72263f0a3141)

# 4. Creación de nuevas columnas
**notas feature engineering feature 1
![image](https://github.com/user-attachments/assets/2b569830-309a-4284-a9d9-e9c698a40c7d)

**notas feature engineering feature 2
![image](https://github.com/user-attachments/assets/95ea31ed-5511-4577-a20a-92aad272a3a1)

![image](https://github.com/user-attachments/assets/e7aca9f1-eee4-4ee3-95e3-6b4585b0b3d3)


# 5. Balanceo de la target
Para mejorar la detección de fraudes, se aplicó SMOTE al conjunto de entrenamiento, elevando los casos de fraude a aproximadamente el 30%. Esto ayudó a que el modelo reconociera mejor la clase minoritaria sin alterar demasiado la distribución original.
![image](https://github.com/user-attachments/assets/993fe080-48be-4021-869c-23fab33c8eb5)


# 6. Entrenamiento del modelo

En esta fase, se entrenó un modelo de ensamblado mediante votación que integra múltiples algoritmos de clasificación. Este enfoque permite al modelo aprender a partir del conjunto de entrenamiento, utilizando las variables seleccionadas como predictores y los registros de fraude como variable objetivo. La combinación de diferentes clasificadores mejora el desempeño general, ya que aprovecha las fortalezas individuales de cada modelo. Este esquema de ensamblado se utilizó con el propósito de identificar el modelo con el mejor rendimiento.

![image](https://github.com/user-attachments/assets/6bf19557-5c09-4a08-bd74-a707e6411ab1)

El modelo XGBClassifier mostró el mejor rendimiento, especialmente al identificar casos de fraude, con una alta precisión del 99%. Aunque aún no detecta todos los fraudes (recall menor al 50%), logró una mejora notable frente a los otros modelos. En general, es un modelo equilibrado y efectivo, por lo que se considera una buena opción para detectar fraudes.

![image](https://github.com/user-attachments/assets/6d4b8daa-f429-49c3-a095-3cb73527be74)


# 7.Tuneo de hiperparámetros







