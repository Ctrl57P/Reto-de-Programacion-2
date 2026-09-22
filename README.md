# mintic-base

--------------------------------------------------------------------------------------
DATOS:

    CRISTIAN LOPEZ GARCIA 
        42300623-4
        TECNOLOGIA PARA LA INFORMACION EN CIENCIAS
        2027-1
    MINERIA DE DATOS

TAREA 2 - RETO DE PROGRAMACION - Random Forest (Bosques aleatorios)


--------------------------------------------------------------------------------------

Plantilla base del curso de Minería de Datos

**ENES Morelia, UNAM**

--------------------------------------------------------------------------------------

## Sobre el proyecto

`mintic-base` es el repositorio sobre el que se construye una librería propia de Minería de Datos. El objetivo es comprender el funcionamiento interno de los algoritmos mediante su programación desde cero.

Para este **Reto 2**, se extendió la librería creando el submódulo `mintic/ensemble`, el cual implementa un algoritmo de Random Forest basado en árboles de decisión ID3 para conjuntos de datos 100% categóricos.

## Reglas del desarrollo

- Toda la lógica del ensamble (Bootstrap, ID3, Random Forest y Votación) se implementó manualmente dentro del subpaquete `mintic/ensemble/`.
- La única librería numérica permitida para los cálculos matemáticos del algoritmo es **NumPy**. Está estrictamente prohibido usar `scikit-learn` o funciones avanzadas de Python.
- `pandas` se utiliza únicamente para la lectura y manipulación de DataFrames (carga de los datasets categóricos).
- `matplotlib` y `seaborn` están permitidos exclusivamente para la generación de gráficos.

## Estructura del repositorio

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
└── mintic/
    ├── __init__.py
    ├── eda/
    │   └── __init__.py
    ├── ensemble/
    │   ├── __init__.py
    │   └── random_forest.py
    ├── kmeans/
    │   └── __init__.py
    ├── dbscan/
    │   └── __init__.py
    ├── apriori/
    │   └── __init__.py
    └── pca/
        └── __init__.py
