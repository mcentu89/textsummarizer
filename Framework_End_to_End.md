---


## Búsqueda de Hiperparámetros con GridSearchCV en tu pipeline

En este framework, la función `train_and_select_model` (o `initiate_model_trainer` en el código) realiza automáticamente el ajuste de hiperparámetros para cada modelo utilizando `GridSearchCV` de scikit-learn. Los grids de hiperparámetros se definen en el propio código o pueden cargarse desde un archivo YAML para mayor flexibilidad.

**¿Cómo funciona?**

- Por cada modelo, se define un diccionario de hiperparámetros (por ejemplo, para Random Forest, CatBoost, XGBoost, etc.).
- La función recorre cada modelo y ejecuta una búsqueda de hiperparámetros con validación cruzada (`GridSearchCV`).
- Se selecciona el mejor conjunto de hiperparámetros y se entrena el modelo final con ellos.
- El mejor modelo se guarda automáticamente para su uso posterior.

**Fragmento de código relevante:**

```python
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    for i in range(len(models)):
        model = list(models.values())[i]
        param = params[list(models.keys())[i]]
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        # ...
```

Esta función es llamada desde `initiate_model_trainer` para automatizar el proceso de selección y ajuste de modelos.

> **Importante:** Así garantizas que tu pipeline realiza ajuste de hiperparámetros de forma reproducible y profesional, sin pasos manuales adicionales.

### Paso 1: Definir los grids de búsqueda en params.yaml

Ejemplo:

```yaml
grid_search:
    random_forest:
        n_estimators: [100, 200, 300]
        max_depth: [3, 5, 7]
        min_samples_split: [2, 5]
    catboost:
        iterations: [100, 300, 500]
        learning_rate: [0.01, 0.03, 0.1]
        depth: [4, 6, 8]
    xgboost:
        n_estimators: [100, 200]
        learning_rate: [0.01, 0.05]
        max_depth: [3, 4, 5]
```

### Paso 2: Leer los grids desde Python

```python
import yaml

with open('params.yaml', 'r') as file:
        config = yaml.safe_load(file)

rf_grid = config['grid_search']['random_forest']
cat_grid = config['grid_search']['catboost']
xgb_grid = config['grid_search']['xgboost']
```

### Paso 3: Aplicar GridSearchCV en el entrenamiento

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Para CatBoost y XGBoost importa sus clases correspondientes

# Random Forest
rf = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf, rf_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
print('Mejores hiperparámetros RF:', rf_grid_search.best_params_)

# CatBoost (ejemplo)
from catboost import CatBoostClassifier
cat = CatBoostClassifier(verbose=0)
cat_grid_search = GridSearchCV(cat, cat_grid, cv=5, scoring='accuracy', n_jobs=-1)
cat_grid_search.fit(X_train, y_train)
best_cat = cat_grid_search.best_estimator_
print('Mejores hiperparámetros CatBoost:', cat_grid_search.best_params_)

# XGBoost (ejemplo)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb_grid_search = GridSearchCV(xgb, xgb_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
print('Mejores hiperparámetros XGBoost:', xgb_grid_search.best_params_)
```

### Paso 4: Guardar el mejor modelo

Puedes guardar el mejor modelo de cada algoritmo usando pickle o joblib:

```python
import joblib
joblib.dump(best_rf, 'artifacts/models/best_rf.pkl')
joblib.dump(best_cat, 'artifacts/models/best_cat.pkl')
joblib.dump(best_xgb, 'artifacts/models/best_xgb.pkl')
```

---

**Con esto, tu pipeline realizará búsqueda de hiperparámetros de forma reproducible y profesional para cada modelo.**
# Framework End-to-End: Proyecto de Machine Learning

## Introducción

Esta guía proporciona un framework genérico para desarrollar proyectos de Machine Learning de principio a fin. La estructura y metodologías aquí descritas son aplicables a cualquier proyecto de ML/DL y reflejan las mejores prácticas utilizadas en la industria.

---

## Video 1: Configuración Inicial del Proyecto

### Objetivos del Video
- Configurar repositorio GitHub
- Crear un nuevo entorno de desarrollo
- Establecer la estructura básica del proyecto
- Configurar `setup.py` y `requirements.txt`
- Construir el proyecto como un paquete instalable

---

## Paso 1: Crear la Carpeta del Proyecto

1. Crear una carpeta en tu sistema donde se desarrollará el proyecto
2. Abrir Anaconda Prompt o terminal
3. Navegar a la carpeta del proyecto:
   ```bash
   cd E:\MLprojects
   ```

---

## Paso 2: Crear el Entorno Conda

Crear un entorno Conda dedicado para el proyecto (recomendado: con nombre).

```bash
conda create -n mlproject python=3.8 -y
```

**Explicación de los parámetros:**
- `-n mlproject`: Crea un entorno con nombre `mlproject` (más sencillo de activar y gestionar)
- `python=3.8`: Especifica la versión de Python
- `-y`: Acepta automáticamente la instalación sin solicitar confirmación

**Activar el entorno:**
```bash
conda activate mlproject
```

> **💡 Buena práctica:** Mantener un entorno con nombre facilita activar/desactivar el entorno y compartir instrucciones de setup.

---

## Paso 3: Crear el Repositorio en GitHub

1. Ir a [GitHub](https://github.com) y acceder a tu cuenta
2. Hacer clic en **"New repository"**
3. Asignar un nombre al repositorio (ej: `MLprojects`)
4. Seleccionar **Public** o **Private** según necesidad
5. **NO** seleccionar ninguna opción adicional (README, .gitignore, license)
6. Hacer clic en **"Create repository"**

---

## Paso 4: Inicializar Git y Conectar con GitHub

### 4.1 Abrir VS Code desde la carpeta del proyecto
```bash
code .
```

### 4.2 Inicializar el repositorio Git local
```bash
git init
```
> Esto crea una carpeta oculta `.git` que contiene toda la información del repositorio.

### 4.3 Crear el archivo README.md
Crear un archivo `README.md` en la raíz del proyecto:

```markdown
# End to End Machine Learning Project
```

### 4.4 Agregar el archivo al staging area
```bash
git add README.md
```

### 4.5 Realizar el primer commit
```bash
git commit -m "first commit"
```

### 4.6 Cambiar la rama a main
```bash
git branch -M main
```

### 4.7 Conectar con el repositorio remoto
```bash
git remote add origin https://github.com/TU_USUARIO/MLprojects.git
```

### 4.8 Configurar credenciales globales (si es primera vez)
```bash
git config --global user.email "tu_email@ejemplo.com"
git config --global user.name "tu_nombre_usuario"
```

### 4.9 Subir el código a GitHub
```bash
git push -u origin main
```

---

## Paso 5: Crear el archivo .gitignore

El archivo `.gitignore` especifica qué archivos y carpetas NO deben ser subidos al repositorio.

1. En VS Code, crear un nuevo archivo llamado `.gitignore`
2. Seleccionar **Python** como plantilla cuando VS Code lo sugiera
3. Agregar la carpeta del entorno virtual:

```gitignore
# Virtual environment
venv
```

4. Hacer commit y push:
```bash
git add .gitignore
git commit -m "Create .gitignore"
git push -u origin main
```

5. Sincronizar cambios locales:
```bash
git pull
```

---

## Paso 6: Crear setup.py

### ¿Qué es setup.py y por qué es necesario?

`setup.py` es el script central para construir, distribuir e instalar módulos en Python. Es responsable de:

- **Crear tu aplicación de Machine Learning como un paquete instalable**
- Permitir que otros instalen tu proyecto con `pip install`
- Definir metadatos del proyecto (nombre, versión, autor, etc.)
- Especificar dependencias requeridas
- Facilitar el despliegue en PyPI (Python Package Index)

> **💡 Concepto clave:** Con `setup.py` puedes construir tu aplicación ML como un paquete que puede ser instalado y utilizado como cualquier biblioteca de Python (pandas, numpy, seaborn, etc.)

### Crear el archivo setup.py

```python
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Esta función devuelve la lista de requisitos desde requirements.txt
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Tu_Nombre',
    author_email='tu_email@ejemplo.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
```

### Explicación detallada del código:

#### Imports
```python
from setuptools import find_packages, setup
from typing import List
```
- `find_packages`: Encuentra automáticamente todos los paquetes (carpetas con `__init__.py`)
- `setup`: Función principal para configurar el paquete
- `List`: Tipo de dato para anotaciones de tipo

#### Función get_requirements()
```python
def get_requirements(file_path: str) -> List[str]:
```

**¿Cómo funciona `typing List[str]`?**
- `file_path: str` indica que el parámetro de entrada debe ser un string
- `-> List[str]` indica que la función retorna una lista de strings
- Esto ayuda con la documentación del código y detección de errores

**¿Por qué quitamos `-e .`?**
- `-e .` es una instrucción para pip que activa `setup.py`
- NO debe incluirse en `install_requires` porque causaría un error
- Se usa solo en `requirements.txt` para conectar con `setup.py`

#### Parámetros de setup()
| Parámetro | Descripción |
|-----------|-------------|
| `name` | Nombre del paquete |
| `version` | Versión actual (formato semántico: MAJOR.MINOR.PATCH) |
| `author` | Nombre del autor |
| `author_email` | Email de contacto |
| `packages` | Lista de paquetes a incluir (find_packages los detecta automáticamente) |
| `install_requires` | Lista de dependencias requeridas |

---

## Paso 7: Crear requirements.txt

El archivo `requirements.txt` contiene todas las dependencias del proyecto.

```txt
pandas
numpy
seaborn
-e .
```

### ¿Qué significa `-e .`?

- `-e` significa "editable mode" (modo editable)
- `.` indica el directorio actual
- **Cuando pip encuentra `-e .`, busca y ejecuta `setup.py`**
- Esto permite que los cambios en el código fuente se reflejen inmediatamente sin reinstalar

> **💡 Conexión importante:** `-e .` en `requirements.txt` activa automáticamente `setup.py`, construyendo tu proyecto como paquete.

---

## Paso 8: Crear la Carpeta src (Source Code)

### 8.1 Crear la estructura de carpetas

```
proyecto/
├── src/
│   └── __init__.py
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 8.2 Crear la carpeta src y el archivo __init__.py

1. Crear carpeta `src` en la raíz del proyecto
2. Dentro de `src`, crear un archivo vacío llamado `__init__.py`

### ¿Por qué es necesario `__init__.py`?

- **`find_packages()` busca todas las carpetas que contengan `__init__.py`**
- Este archivo indica a Python que la carpeta debe tratarse como un paquete
- Puede estar vacío o contener código de inicialización
- Sin este archivo, la carpeta no será reconocida como paquete

> **💡 Regla importante:** Cada carpeta que quieras que sea un paquete debe contener un archivo `__init__.py`. Esto aplica también a subcarpetas.

---

## Paso 9: Instalar el Proyecto

Ejecutar el siguiente comando para instalar todas las dependencias y construir el paquete:

```bash
pip install -r requirements.txt
```

### ¿Qué sucede al ejecutar este comando?

1. Pip lee `requirements.txt`
2. Instala cada dependencia listada (pandas, numpy, seaborn)
3. Encuentra `-e .` y ejecuta `setup.py`
4. `setup.py` usando `find_packages()` encuentra la carpeta `src` (porque tiene `__init__.py`)
5. Construye el paquete con la información de metadatos
6. Crea una carpeta `mlproject.egg-info` con información del paquete

### Verificar la instalación

Después de la instalación, deberías ver:
- Carpeta `mlproject.egg-info` en la raíz del proyecto
- Dentro de `src`: archivos de configuración del paquete

---

## Paso 10: Commit Final

```bash
git add .
git status
git commit -m "setup.py and requirements.txt configured"
git push -u origin main
```

---

## Resumen de Archivos Creados

| Archivo/Carpeta | Propósito |
|-----------------|-----------|
| `README.md` | Documentación del proyecto |
| `.gitignore` | Archivos a ignorar en Git |
| `setup.py` | Configuración del paquete Python |
| `requirements.txt` | Lista de dependencias |
| `src/` | Carpeta del código fuente |
| `src/__init__.py` | Marca src como paquete Python |
| `venv/` | Entorno virtual (no se sube a Git) |

---

## Estructura Final del Proyecto

```
MLprojects/
├── .git/                    # Carpeta de Git (oculta)
├── venv/                    # Entorno virtual (ignorado por Git)
├── src/
│   └── __init__.py
├── mlproject.egg-info/      # Info del paquete (generado automáticamente)
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## Comandos Git más Utilizados

| Comando | Descripción |
|---------|-------------|
| `git init` | Inicializar repositorio |
| `git add .` | Agregar todos los archivos al staging |
| `git commit -m "mensaje"` | Confirmar cambios |
| `git push -u origin main` | Subir cambios a GitHub |
| `git pull` | Descargar cambios de GitHub |
| `git status` | Ver estado de archivos |
| `git remote -v` | Ver repositorios remotos configurados |

---

## Próximos Pasos (Video 2)

## Video 2: Estructura del Proyecto, Logging y Manejo de Excepciones

### Objetivos del Video
- Definir la estructura formal del proyecto
- Implementar logging (registro de eventos)
- Gestionar excepciones
- Aplicar mejores prácticas de organización de código

---

## Paso 11: Definir la Estructura del Proyecto

La estructura recomendada para proyectos de ML End-to-End es la siguiente:

```
proyecto/
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   └── ... (otros módulos)
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Explicación:**
- `logger.py`: módulo para configurar el sistema de logging
- `exception.py`: módulo para definir clases de manejo de excepciones personalizadas
- Otros módulos: aquí irán los scripts principales del proyecto (data, model, etc.)

---

## Paso 12: Implementar Logging

Crear el archivo `logger.py` en la carpeta `src`:

```python
import logging
import os

LOG_DIR = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(lineno)d: %(message)s]"
LOG_FILE = "project.log"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, LOG_FILE),
    level=logging.INFO,
    format=logging_str
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger instance.
    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
```

**¿Por qué es importante el logging?**
- Permite registrar eventos, errores y advertencias durante la ejecución
- Facilita la depuración y el monitoreo del proyecto
- Los logs se guardan en la carpeta `logs` para fácil acceso

---

## Paso 13: Manejo de Excepciones

Crear el archivo `exception.py` en la carpeta `src`:

```python
import sys

def error_message_details(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} with error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message"
```

**¿Por qué crear excepciones personalizadas?**
- Permite identificar y manejar errores específicos del proyecto
- Mejora la claridad y el control sobre el flujo de errores

---

## Paso 14: Buenas Prácticas de Organización

- Cada funcionalidad debe estar en un módulo separado
- Usar logging en cada módulo para registrar eventos importantes
- Manejar excepciones usando la clase personalizada
- Documentar cada función y clase

---

## Ejemplo de Uso de Logging y Excepciones

```python
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

try:
    # Código principal
    logger.info("Inicio del proceso de ML")
    # ...
except Exception as e:
    raise CustomException("Error en el proceso de ML", e)
```

---

## Resumen del Video 2

- Se definió la estructura formal del proyecto
- Se implementó un sistema de logging
- Se creó una clase de excepción personalizada
- Se mostraron buenas prácticas de organización

Continúa con la siguiente sesión para implementar módulos específicos (data, model, etc.)

---


---

> **📝 Nota:** Esta estructura es genérica y puede adaptarse a cualquier proyecto de Machine Learning o Deep Learning. Las técnicas aprendidas aquí son las mismas que se utilizan en la industria.

---

## Video 3: Guía práctica paso a paso (Ingesta, EDA y modularización)

Esta guía convierte la transcripción del Video 3 en pasos accionables que podrás seguir para transformar el notebook en código reproducible.

### 1 — Preparar entorno
- Crear y activar entorno (Conda/venv) y instalar dependencias:

```bash
conda create -n mlproject python=3.8 -y
conda activate mlproject
pip install -r requirements.txt
pip install scikit-learn catboost xgboost  # si faltan
```

### 2 — Organización de ficheros
- Notebooks: `notebook/`
- Datos crudos: `notebook/02_Datos/` (ej. `student_data.csv`)
- Artefactos: `artifacts/raw/`, `artifacts/processed/`, `artifacts/models/`
- Logs: `logs/project.log`

### 3 — Ingesta reproducible
- Leer CSV desde el notebook y guardar copia en `artifacts/raw/`:

```python
import pandas as pd
df = pd.read_csv('notebook/02_Datos/student_data.csv')
df.to_csv('artifacts/raw/student_data.csv', index=False)
```

### 4 — Comprobaciones iniciales (quality checks)
- Valores nulos: `df.isnull().sum()` → anotar y decidir imputación.
- Duplicados: `df.duplicated().sum()` → `df = df.drop_duplicates()` si procede.
- Tipos: `df.dtypes` → convertir columnas mal tipadas.
- Unicidad y conteos: `df['col'].nunique()`, `df['col'].value_counts()`.

### 5 — Ingeniería de variables (según vídeo)
- Crear `score_total` y `score_mean`:

```python
df['score_total'] = df['math score'] + df['reading score'] + df['writing score']
df['score_mean'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
```

---

## Video 7: Ajuste de Hiperparámetros y Configuración YAML

En este paso, aprenderás a gestionar los hiperparámetros de tus modelos de forma flexible y reproducible, utilizando archivos de configuración YAML.

### Paso 1: ¿Por qué usar archivos YAML para hiperparámetros?

- Permiten separar la configuración del código.
- Facilitan la experimentación y el ajuste sin modificar scripts.
- Son legibles y fáciles de versionar.

### Paso 2: Crear un archivo de configuración YAML

1. En la raíz del proyecto (o en una carpeta `config/`), crea un archivo llamado `params.yaml`.
2. Define los hiperparámetros de tus modelos y otros parámetros relevantes. Ejemplo:

```yaml
train_model:
    random_forest:
        n_estimators: 100
        max_depth: 5
        min_samples_split: 2
    catboost:
        iterations: 500
        learning_rate: 0.03
        depth: 6
    xgboost:
        n_estimators: 200
        learning_rate: 0.05
        max_depth: 4
```

Puedes añadir secciones para preprocesamiento, validación, etc.

### Paso 3: Leer el archivo YAML desde Python

Instala la librería `pyyaml` si no la tienes:

```bash
pip install pyyaml
```

Ejemplo de lectura en tu código:

```python
import yaml

with open('params.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Acceso a hiperparámetros
rf_params = config['train_model']['random_forest']
catboost_params = config['train_model']['catboost']
xgb_params = config['train_model']['xgboost']
```

### Paso 4: Usar los hiperparámetros en tu pipeline

Al entrenar tus modelos, pasa los parámetros leídos del YAML:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(**rf_params)
rf.fit(X_train, y_train)
```

Haz lo mismo para CatBoost, XGBoost, etc.

---

### Buenas prácticas
- Versiona siempre tu archivo `params.yaml`.
- Documenta el propósito de cada parámetro.
- Si tienes muchos experimentos, puedes crear varios archivos YAML y alternar entre ellos.

---

**¡Con esto tu proyecto será mucho más flexible y profesional!**
df['score_mean'] = df['score_total'] / 3
```
- Elegir la variable objetivo (`y`): `math score` o `score_total/score_mean`.

### 6 — Separar features numéricas y categóricas
- Numéricas: `numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()`
- Categóricas: `categorical_features = X.select_dtypes(include=['object']).columns.tolist()`

### 7 — Transformación reproducible (ColumnTransformer)
- Pipeline numérico: imputación (median) + StandardScaler.
- Pipeline categórico: imputación (most_frequent) + OneHotEncoder(handle_unknown='ignore').

Ejemplo:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', num_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)])
```

### 8 — División train/test y evaluación
- `from sklearn.model_selection import train_test_split`
- `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
- Métricas: MAE, RMSE, R2. Implementa una función `evaluate_model()` que devuelva estas métricas.

### 9 — Entrenamiento rápido y selección de modelos
- Prueba modelos base: `LinearRegression`, `Ridge`, `RandomForestRegressor`, `CatBoost`, `XGBoost`.
- Ordena por la métrica objetivo (p. ej. RMSE) y elige el candidato a mejorar.

### 10 — Logs, excepciones y trazabilidad
- Importante: si no aparecen entradas de excepción en el log, importa explícitamente `from src.logger import get_logger` en cada script/notebook y usa `logger.exception()` o `logger.info()`.
- Captura fallos importantes con `CustomException` para incluir trazas y línea de error.

### 11 — Persistencia de artefactos
- Guardar `preprocessor` y `best_model` con `joblib.dump()` en `artifacts/models/`.
- Guardar métricas en `artifacts/metrics/metrics.json`.

### 12 — Notas prácticas y problemas comunes (del vídeo)
- Al instalar librerías, comentar temporalmente `-e .` en `requirements.txt` si produce rebuilds continuos.
- Si falta un paquete (sklearn, catboost...), instalarlo y volver a ejecutar.
- Mantén el notebook para exploración; extrae bloques reproducibles a funciones en `src/`.

### 13 — Commit final de la sesión EDA

```bash
git add .
git commit -m "EDA, ingesta y transformaciones iniciales"
git push
```

---

Si deseas, puedo ahora crear un archivo separado `Video_03_Guida.md` con este contenido o dejarlo incrustado aquí; dime qué prefieres.

---

## Video 4: De Notebooks a Pipelines y Componente de Ingesta de Datos

En este video se inicia la transición del trabajo exploratorio en notebooks hacia la construcción de pipelines reproducibles y componentes modulares, comenzando por la **ingesta de datos**.

### 1 — ¿Por qué migrar de notebooks a pipelines?
- Los notebooks son ideales para exploración y prototipado, pero no para producción.
- Para un flujo CI/CD y despliegue continuo, el código debe estar modularizado y automatizado.
- El objetivo es que el pipeline pueda ejecutarse automáticamente cada vez que se requiera, sin intervención manual.

### 2 — Estructura recomendada para componentes
Organiza los componentes principales del pipeline en una carpeta, por ejemplo `src/components/`:

```
src/
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── ...
```

Cada archivo representa un paso del pipeline y debe ser una clase o función autocontenida.

### 3 — Creación del componente de ingesta de datos

**Paso a paso:**
1. Crea el archivo `data_ingestion.py` dentro de `src/components/`.
2. Define una clase `DataIngestion` que:
   - Lea los datos crudos desde la fuente (ej: CSV en `notebook/02_Datos/`)
   - Guarde una copia en `artifacts/raw/` para trazabilidad
   - Separe los datos en conjuntos de entrenamiento y prueba (`train.csv`, `test.csv` en `artifacts/`)
   - Registre los pasos en el log y maneje excepciones con `CustomException`

**Ejemplo de estructura básica:**

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException

class DataIngestion:
    def __init__(self, raw_data_path, train_path, test_path, test_size=0.2, random_state=42):
        self.raw_data_path = raw_data_path
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size
        self.random_state = random_state
        self.logger = get_logger(__name__)

    def initiate_data_ingestion(self):
        try:
            self.logger.info("Leyendo datos crudos desde %s", self.raw_data_path)
            df = pd.read_csv(self.raw_data_path)
            os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
            df.to_csv(self.train_path.replace('train.csv', 'raw.csv'), index=False)
            self.logger.info("Dividiendo datos en train/test")
            train_set, test_set = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
            train_set.to_csv(self.train_path, index=False)
            test_set.to_csv(self.test_path, index=False)
            self.logger.info("Ingesta completada: %s (train), %s (test)", self.train_path, self.test_path)
            return self.train_path, self.test_path
        except Exception as e:
            self.logger.exception("Fallo en la ingesta de datos")
            raise CustomException(e, sys)
```

**Notas:**
- Ajusta rutas y nombres según tu proyecto.
- Usa logs y excepciones para trazabilidad.
- Este patrón se repetirá para los siguientes componentes (transformación, entrenamiento, etc.).

### 4 — Próximos pasos
- Implementar el resto de componentes como módulos independientes en `src/components/`.
- Encadenar los componentes en un pipeline principal (por ejemplo, en `src/pipeline/` o un script `main.py`).
- Automatizar la ejecución con scripts o herramientas de orquestación (Makefile, scripts bash, o herramientas como Airflow).

---


**Resumen del Video 4:**
- Se inicia la migración de notebooks a código modular y pipelines.
- Se crea el primer componente: ingesta de datos.
- Se enfatiza la importancia de la trazabilidad, logs y manejo de errores.
- El objetivo es que cada paso del pipeline sea reproducible y automatizable.

---

## Video 5: Transformación de Datos

En este video se aborda el proceso de transformación de datos, un paso fundamental tras la ingesta y partición de los datos en conjuntos de entrenamiento y prueba.

### 1 — Revisión de los datos particionados
- Verifica que los archivos `train.csv` y `test.csv` se hayan generado correctamente en la carpeta `artifacts/`.
- Abre ambos archivos y revisa que contengan los datos esperados.

### 2 — Creación del componente de transformación de datos
1. Crea el archivo `data_transformation.py` dentro de `src/components/`.
2. Define una clase `DataTransformation` que:
   - Lea los archivos `train.csv` y `test.csv` generados en la etapa de ingesta.
   - Realice las transformaciones necesarias (imputación, escalado, codificación, etc.).
   - Guarde los datos transformados y el objeto de preprocesamiento en la carpeta `artifacts/`.
   - Registre los pasos en el log y maneje excepciones con `CustomException`.

**Ejemplo de estructura básica:**

```python
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from src.logger import get_logger
from src.exception import CustomException

class DataTransformation:
    def __init__(self, train_path, test_path, preprocessor_path):
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor_path = preprocessor_path
        self.logger = get_logger(__name__)

    def initiate_data_transformation(self):
        try:
            self.logger.info("Leyendo datos de entrenamiento y prueba")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)

            # Separar variables predictoras y objetivo
            X_train = train_df.drop('target', axis=1)
            y_train = train_df['target']
            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']

            # Identificar variables numéricas y categóricas
            numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

            # Pipelines de transformación
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numeric_features),
                ('cat', cat_pipeline, categorical_features)
            ])

            self.logger.info("Ajustando el preprocesador con los datos de entrenamiento")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Guardar el preprocesador para uso futuro
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.preprocessor_path)
            self.logger.info(f"Preprocesador guardado en {self.preprocessor_path}")

            # Opcional: guardar los datos transformados
            # joblib.dump((X_train_processed, y_train), 'artifacts/train_processed.pkl')
            # joblib.dump((X_test_processed, y_test), 'artifacts/test_processed.pkl')

            return X_train_processed, X_test_processed, y_train, y_test
        except Exception as e:
            self.logger.exception("Fallo en la transformación de datos")
            raise CustomException(e, sys)
```

**Notas importantes:**
- Cambia `'target'` por el nombre real de tu variable objetivo.
- Ajusta rutas y nombres según tu proyecto.
- Usa logs y excepciones para trazabilidad.

### 3 — Buenas prácticas y recomendaciones
- Documenta cada paso y justifica las transformaciones aplicadas.
- Guarda el objeto de preprocesamiento (`preprocessor.pkl`) para reutilizarlo en inferencia y producción.
- Si surgen dudas sobre la selección de variables o técnicas de transformación, consulta las notas del vídeo o experimenta en notebooks antes de fijar el pipeline.

### 4 — Commit de la etapa de transformación

```bash
git add .
git commit -m "Componente de transformación de datos"
git push
```

---


---

## Video 6: Entrenamiento de Modelos

En este video se aborda la etapa de entrenamiento de modelos, partiendo de los datos ya transformados y listos para modelar. El objetivo es construir, comparar y seleccionar el mejor modelo para el problema.

### 1 — Revisión previa
- Verifica que los archivos de datos transformados y el preprocesador estén correctamente generados en la carpeta `artifacts/`.
- Asegúrate de que los scripts de ingesta y transformación funcionen de forma reproducible.

### 2 — Creación del componente de entrenamiento de modelos
1. Crea el archivo `model_trainer.py` dentro de `src/components/`.
2. Define una clase `ModelTrainer` que:
   - Cargue los datos transformados y el preprocesador.
   - Entrene diferentes modelos (por ejemplo: `LinearRegression`, `RandomForestRegressor`, `CatBoostRegressor`, `XGBRegressor`).
   - Compare los resultados usando métricas como MAE, RMSE y R2.
   - Seleccione el mejor modelo según la métrica objetivo.
   - Guarde el modelo entrenado en la carpeta `artifacts/`.
   - Registre los pasos en el log y maneje excepciones con `CustomException`.

**Ejemplo de estructura básica:**

```python
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import get_logger
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, preprocessor_path, model_dir):
        self.preprocessor_path = preprocessor_path
        self.model_dir = model_dir
        self.logger = get_logger(__name__)

    def train_and_select_model(self, X_train, y_train, X_test, y_test):
        try:
            self.logger.info("Entrenando modelos con ajuste de hiperparámetros (GridSearchCV)")
            from sklearn.model_selection import GridSearchCV
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0),
                'XGBoost': XGBRegressor(verbosity=0)
            }
            param_grids = {
                'LinearRegression': {},
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                },
                'CatBoost': {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.03, 0.1],
                    'iterations': [100, 300, 500]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.05],
                    'max_depth': [3, 4, 5]
                }
            }
            results = {}
            for name, model in models.items():
                grid = param_grids.get(name, {})
                if grid:
                    self.logger.info(f"Buscando hiperparámetros óptimos para {name}...")
                    gs = GridSearchCV(model, grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                    self.logger.info(f"{name} mejores hiperparámetros: {gs.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'model': best_model}
                self.logger.info(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

            # Seleccionar el mejor modelo según RMSE (puedes cambiar la métrica)
            best_model_name = min(results, key=lambda n: results[n]['rmse'])
            best_model = results[best_model_name]['model']
            self.logger.info(f"Mejor modelo: {best_model_name}")

            # Guardar el modelo
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, f"{best_model_name}_model.pkl")
            joblib.dump(best_model, model_path)
            self.logger.info(f"Modelo guardado en {model_path}")

            # Guardar métricas
            metrics = {k: {m: float(v[m]) for m in ['mae', 'rmse', 'r2']} for k, v in results.items()}
            import json
            with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            return best_model_name, model_path, metrics
        except Exception as e:
            self.logger.exception("Fallo en el entrenamiento de modelos")
            raise CustomException(e, sys)
```

**Notas:**
- Ajusta los modelos y parámetros según tu problema.
- Cambia la métrica de selección si tu objetivo es otro (por ejemplo, maximizar R2).
- Usa logs y excepciones para trazabilidad.

### 3 — Commit de la etapa de entrenamiento

```bash
git add .
git commit -m "Componente de entrenamiento de modelos"
git push
```

---

Con esto, el pipeline ya cuenta con los pasos de ingesta, transformación y entrenamiento de modelos, todos modularizados y listos para ser encadenados en un flujo reproducible y automatizable.

---

## Video 8: Creación de la Tubería de Predicción y App Web

En este video se aborda cómo desplegar el modelo entrenado mediante una aplicación web sencilla, utilizando Flask, y cómo estructurar la tubería de predicción para consumir los artefactos generados (modelo y preprocesador).

### 1 — ¿Por qué una tubería de predicción?
- Permite consumir el modelo entrenado desde una interfaz web o API.
- Facilita la integración con aplicaciones reales y la validación de predicciones en producción.
- Centraliza la lógica de preprocesamiento y predicción, asegurando reproducibilidad.

### 2 — Estructura recomendada
Organiza la lógica de predicción en un módulo dedicado, por ejemplo `src/pipeline/predict_pipeline.py`, y la app web en `app.py`.

```
proyecto/
├── src/
│   └── pipeline/
│       └── predict_pipeline.py
├── app.py
├── templates/
│   ├── index.html
│   └── home.html
```

### 3 — Creación de la app Flask (`app.py`)
1. Importa las librerías necesarias:
   - `flask`, `render_template`, `request`
   - `pickle`, `numpy`, `pandas`
2. Inicializa la app Flask y define las rutas:
   - `/` para la página de inicio (`index.html`)
   - `/predictdata` para el formulario de predicción (`home.html`)
3. En la ruta `/predictdata`, gestiona tanto GET (mostrar formulario) como POST (recibir datos y predecir).

**Ejemplo básico de estructura:**

```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Aquí se capturan los datos del formulario y se llama a la tubería de predicción
        # ...
        return render_template('home.html', result=prediccion)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
```

### 4 — Plantillas HTML
- `index.html`: página de bienvenida simple.
- `home.html`: formulario con los campos requeridos para la predicción (género, raza, nivel parental, almuerzo, preparación para el examen, puntuaciones, etc.).
  - El formulario debe hacer POST a `/predictdata`.

### 5 — Tubería de predicción modular (`predict_pipeline.py`)
1. Crea una clase `PredictPipeline` que:
   - Cargue el modelo y preprocesador desde los archivos pickle (`artifacts/model.pkl`, `artifacts/preprocessor.pkl`).
   - Reciba los datos de entrada, aplique el preprocesador y realice la predicción.
2. Crea una clase `CustomData` para mapear los datos del formulario a un DataFrame listo para el modelo.

**Ejemplo simplificado:**

```python
import pandas as pd
import pickle

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        return pd.DataFrame(data)

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'

    def predict(self, features):
        with open(self.preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        data_scaled = preprocessor.transform(features)
        preds = model.predict(data_scaled)
        return preds[0]
```

### 6 — Conexión entre app y pipeline
En el método POST de `/predictdata`:
1. Recoge los datos del formulario con `request.form.get('campo')`.
2. Crea un objeto `CustomData` y genera el DataFrame.
3. Llama a `PredictPipeline().predict(df)` y muestra el resultado en la plantilla.

**Ejemplo:**

```python
if request.method == 'POST':
    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=int(request.form.get('reading_score')),
        writing_score=int(request.form.get('writing_score')),
    )
    df = data.get_data_as_dataframe()
    result = PredictPipeline().predict(df)
    return render_template('home.html', result=result)
```

### 7 — Buenas prácticas y recomendaciones
- Usa logs y manejo de excepciones en la tubería de predicción.
- Mantén los archivos pickle versionados y protegidos.
- Valida los datos de entrada antes de predecir.
- El frontend puede ser simple; lo importante es la lógica reproducible y modular.

---

Con esto, tu framework ya permite desplegar un modelo entrenado y consumirlo desde una app web, cerrando el ciclo end-to-end de un proyecto de Machine Learning.
---

## Video 9: Despliegue en la nube con AWS Elastic Beanstalk

En este apartado aprenderás a desplegar tu aplicación de Machine Learning en la nube usando AWS Elastic Beanstalk y CodePipeline, logrando un flujo de entrega continua (CD) profesional.

### 1. Preparativos previos

- Asegúrate de tener una cuenta de AWS activa (requiere tarjeta de crédito).
- Tu código debe estar en un repositorio de GitHub y probado localmente.
- El punto de entrada de tu app debe ser un archivo `app.py` (o el nombre que uses en Flask, FastAPI, etc.).

### 2. Configuración para Elastic Beanstalk

1. **Crea la carpeta `.ebextensions/` en la raíz del proyecto.**
2. Dentro de `.ebextensions/`, crea un archivo de configuración (por ejemplo, `python.config`) con el siguiente contenido:

     ```yaml
     options_settings:
        "aws:elasticbeanstalk:container:python":
            WSGIPath: app:app
     ```
     - `app:app` indica que el punto de entrada es el archivo `app.py` y el objeto Flask se llama `app`.
     - Si tu archivo principal se llama diferente, ajusta el nombre.

3. **Verifica que tu archivo `app.py` no tenga `debug=True`** en el bloque principal. Elimina o comenta esta opción antes de desplegar.

### 3. Confirmar y subir los cambios a GitHub

```bash
git add .
git commit -m "Configuración de despliegue para AWS Elastic Beanstalk"
git push -u origin main
```

### 4. Crear la aplicación en Elastic Beanstalk

1. Ingresa a la consola de AWS y busca "Elastic Beanstalk".
2. Haz clic en **Create Application**.
3. Asigna un nombre (ejemplo: `mlproject-app`).
4. Selecciona la plataforma **Python** (elige la versión compatible, por ejemplo, Python 3.8).
5. Elige la opción de ejemplo para la primera creación (luego la conectaremos a GitHub).
6. Crea la aplicación y espera a que el entorno esté listo (puede tardar unos minutos).

### 5. Configurar AWS CodePipeline para entrega continua

1. En la consola de AWS, busca **CodePipeline** y crea una nueva pipeline.
2. Asigna un nombre (ejemplo: `mlproject-pipeline`).
3. En **Source provider**, selecciona **GitHub** y autoriza el acceso a tu cuenta/repositorio.
4. Elige el repositorio y rama donde está tu código.
5. Omite la etapa de build (puedes agregarla si necesitas pasos de construcción personalizados).
6. En **Deploy provider**, selecciona **Elastic Beanstalk**.
7. Elige la aplicación y el entorno creados previamente.
8. Finaliza la creación de la pipeline.

### 6. Despliegue automático y pruebas

- Cada vez que hagas un push a la rama configurada en GitHub, CodePipeline detectará el cambio y desplegará automáticamente la nueva versión en Elastic Beanstalk.
- Puedes monitorear el estado del despliegue desde la consola de AWS.
- Una vez desplegado, accede a la URL pública que te da Elastic Beanstalk para probar tu app web.

### 7. Resolución de problemas comunes

- Si el despliegue falla, revisa los logs en Elastic Beanstalk y asegúrate de que:
    - El archivo de entrada y el objeto Flask están correctamente configurados en `.ebextensions/python.config`.
    - No hay errores de dependencias en `requirements.txt`.
    - No hay archivos innecesarios como `application.py` si tu entrada es `app.py`.
- Si cambias el nombre del archivo principal, actualiza la configuración WSGIPath.

### 8. Buenas prácticas

- Mantén tu archivo de configuración y dependencias versionados.
- No subas archivos sensibles ni credenciales al repositorio.
- Usa variables de entorno para parámetros secretos.
- Documenta el proceso de despliegue en tu README.

---

Con estos pasos, tu proyecto de Machine Learning estará desplegado en la nube, con entrega continua y acceso web público, listo para producción o pruebas reales.

---

## Video 10: Despliegue en AWS con Docker, ECR y EC2 (CI/CD con GitHub Actions)

En este apartado aprenderás a desplegar tu aplicación usando contenedores Docker, almacenando las imágenes de forma privada en **AWS ECR** (Elastic Container Registry) y ejecutándolas en una instancia **AWS EC2**, todo orquestado por un pipeline CI/CD con **GitHub Actions**.

### Arquitectura del despliegue

```
GitHub (push) → GitHub Actions (CI/CD) → AWS ECR (imagen Docker privada) → AWS EC2 (servidor Ubuntu)
```

El flujo tiene tres grandes etapas:
1. **Integración Continua (CI):** Valida el código (linting, tests unitarios).
2. **Entrega Continua — Build & Push:** Construye la imagen Docker y la sube al repositorio ECR privado.
3. **Despliegue Continuo (CD):** La instancia EC2 tira la imagen desde ECR y la ejecuta.

---

### Paso 1: Crear el Dockerfile

En la raíz del proyecto crea un archivo llamado `Dockerfile` (sin extensión):

```dockerfile
FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y awscli

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
```

**Explicación de cada instrucción:**

| Instrucción | Descripción |
|---|---|
| `FROM python:3.8-slim-buster` | Imagen base Linux (Debian Buster) con Python 3.8, ligera |
| `WORKDIR /app` | Directorio de trabajo dentro del contenedor |
| `COPY . /app` | Copia todo el proyecto al contenedor |
| `RUN apt-get update` | Actualiza los índices de paquetes del sistema |
| `RUN pip install -r requirements.txt` | Instala las dependencias de Python |
| `CMD ["python3", "app.py"]` | Comando que ejecuta la aplicación al iniciar el contenedor |

> **Importante:** El archivo de entrada debe llamarse `app.py` (o el nombre que uses). Si usas `application.py`, ajusta el `CMD` en consecuencia.

---

### Paso 2: Crear el flujo de trabajo de GitHub Actions

Crea la carpeta `.github/workflows/` en la raíz del proyecto y dentro el archivo `main.yaml`:

```yaml
name: workflow

on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'

permissions:
  id-token: write
  contents: read

jobs:

  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Lint code
        run: echo "Linting repository"

      - name: Run unit tests
        run: echo "Running unit tests"

  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Install Utilities
        run: |
          sudo apt-get update
          sudo apt-get install -y jq unzip

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"

  continuous-deploy:
    name: Continuous Deployment
    needs: build-and-push-ecr-image
    runs-on: self-hosted
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Pull latest images
        run: |
          docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

      - name: Stop and remove container if running
        run: |
          docker ps -q --filter "name=mltest" | grep -q . && docker stop mltest && docker rm -fv mltest || true

      - name: Run Docker Image to serve users
        run: |
          docker run -d -p 8080:8080 --name=mltest \
            -e 'AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}' \
            -e 'AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}' \
            -e 'AWS_REGION=${{ secrets.AWS_REGION }}' \
            ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest

      - name: Clean previous images and containers
        run: docker system prune -f
```

**Descripción de los tres jobs:**

- **`integration`**: Se ejecuta en `ubuntu-latest`. Valida el código. Aquí puedes añadir comandos reales de pytest.
- **`build-and-push-ecr-image`**: Depende de que `integration` haya tenido éxito. Construye la imagen Docker y la sube al repositorio ECR privado.
- **`continuous-deploy`**: Se ejecuta en el **runner autoalojado** (`self-hosted`) instalado en la instancia EC2. Descarga la imagen ECR y la pone en marcha.

> **Nota:** Este workflow solo se activa en push a la rama `main`. Los cambios en `README.md` se ignoran para evitar despliegues innecesarios.

---

### Paso 3: Crear usuario IAM en AWS

El usuario IAM (Identity and Access Management) permite que GitHub Actions se autentique en AWS con permisos limitados (sin exponer credenciales de administrador).

1. Accede a la consola de AWS → busca **IAM** → **Users** → **Add users**.
2. Introduce un nombre de usuario (ej: `mlproject-deploy`).
3. En **Permissions**, selecciona **Attach policies directly** y añade:
   - `AmazonEC2ContainerRegistryFullAccess` — para subir/bajar imágenes Docker en ECR.
   - `AmazonEC2FullAccess` — para gestionar la instancia EC2.
4. Revisa el resumen y haz clic en **Create user**.

#### 3.1 Generar credenciales de acceso (Access Keys)

1. Entra en el usuario recién creado → pestaña **Security credentials**.
2. Haz clic en **Create access key**.
3. Selecciona el caso de uso **Command Line Interface (CLI)**.
4. Guarda el archivo CSV descargado. Contiene:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

> **Seguridad:** Nunca compartas ni subas estas claves a un repositorio. Son credenciales privadas de tu cuenta de AWS.

---

### Paso 4: Crear el repositorio ECR (Elastic Container Registry)

ECR es el registro privado de imágenes Docker de AWS. A diferencia de Docker Hub (público), ECR mantiene tus imágenes privadas y dentro del ecosistema AWS.

1. En la consola de AWS → busca **ECR** (Elastic Container Registry).
2. Haz clic en **Create repository**.
3. Configura:
   - **Visibility settings:** `Private`
   - **Repository name:** `student-performance` (o el nombre de tu proyecto)
4. Haz clic en **Create repository**.
5. **Copia y guarda la URI del repositorio** (formato: `123456789.dkr.ecr.us-east-1.amazonaws.com/student-performance`). La necesitarás como secreto de GitHub.

---

### Paso 5: Crear la instancia EC2

EC2 es el servidor virtual en la nube donde se ejecutará tu contenedor Docker.

1. En la consola de AWS → busca **EC2** → **Launch Instance**.
2. Configura la instancia:
   - **Name:** `student-performance` (o el nombre de tu proyecto)
   - **AMI:** `Ubuntu Server 22.04 LTS` (64-bit x86)
   - **Instance type:** `t2.medium` (o mayor según los recursos que necesite tu app)
   - **Key pair:** Selecciona o crea un par de claves (guarda el archivo `.pem`)
   - **Security group:**
     - Habilita tráfico HTTP (puerto 80)
     - Habilita tráfico HTTPS (puerto 443)
     - Añade regla **Custom TCP** puerto **8080** (origen: `0.0.0.0/0`)
3. Haz clic en **Launch Instance** y espera a que pase a estado `Running`.

> **Aviso de costes:** Recuerda detener o terminar la instancia cuando no la uses para evitar cargos inesperados.

---

### Paso 6: Instalar Docker en la instancia EC2

Conéctate a la instancia EC2 desde la consola de AWS (pestaña **Connect** → **EC2 Instance Connect**) y ejecuta los siguientes comandos uno a uno:

```bash
# 1. Actualizar índices de paquetes
sudo apt-get update -y

# 2. Actualizar paquetes instalados
sudo apt-get upgrade -y

# 3. Instalar Docker
sudo apt-get install docker.io -y

# 4. Añadir tu usuario al grupo docker (evita tener que usar sudo en cada comando)
sudo usermod -aG docker ubuntu

# 5. (Opcional) Reiniciar el servicio Docker
sudo systemctl restart docker

# 6. Verificar que Docker funciona
docker --version

# 7. Cambiar al nuevo grupo para usar Docker sin sudo
newgrp docker
```

> **¿Por qué estos comandos?** Cada instancia EC2 nueva arranca sin Docker instalado. Estos pasos configuran el entorno para que el runner de GitHub Actions pueda ejecutar comandos `docker` sin privilegios de superusuario.

---

### Paso 7: Configurar EC2 como runner autoalojado en GitHub

El **runner autoalojado** (self-hosted runner) permite que GitHub Actions ejecute el job de despliegue directamente dentro de la instancia EC2.

1. En tu repositorio de GitHub → **Settings** → **Actions** → **Runners** → **New self-hosted runner**.
2. Selecciona **Linux** y **x64**.
3. Copia y ejecuta en la terminal de EC2 los comandos que GitHub genera (son tres bloques):

```bash
# Bloque 1: Descargar el runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-X.X.X.tar.gz -L https://github.com/actions/runner/releases/download/vX.X.X/actions-runner-linux-x64-X.X.X.tar.gz
echo "HASH  actions-runner-linux-x64-X.X.X.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-X.X.X.tar.gz

# Bloque 2: Configurar el runner
./config.sh --url https://github.com/TU_USUARIO/TU_REPO --token TU_TOKEN
# Cuando pregunte:
#   - Runner group: pulsa Enter (default)
#   - Name of runner: escribe "self-hosted"
#   - Additional labels: pulsa Enter
#   - Work folder: pulsa Enter

# Bloque 3: Iniciar el runner
./run.sh
```

4. Verifica en GitHub (**Settings → Actions → Runners**) que el runner aparece en estado **Idle** (círculo verde). Esto indica que está escuchando y listo para recibir jobs.

> **Importante:** El runner debe estar en ejecución antes de que se dispare el workflow. Para producción real, considera instalar el runner como servicio con `sudo ./svc.sh install && sudo ./svc.sh start`.

---

### Paso 8: Añadir los Secrets de GitHub

Los secretos permiten que el workflow acceda a credenciales de AWS sin exponerlas en el código.

Ve a tu repositorio GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

Añade los siguientes cinco secretos:

| Nombre del secret | Valor |
|---|---|
| `AWS_ACCESS_KEY_ID` | ID de clave de acceso del usuario IAM (del CSV descargado) |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta del usuario IAM (del CSV descargado) |
| `AWS_REGION` | Región de tu instancia EC2 (ej: `us-east-1`) |
| `AWS_ECR_LOGIN_URI` | URI base del registro ECR (ej: `123456789.dkr.ecr.us-east-1.amazonaws.com`) |
| `ECR_REPOSITORY_NAME` | Solo el nombre del repositorio ECR (ej: `student-performance`) |

> **Distinción importante entre los dos últimos:**
> - `AWS_ECR_LOGIN_URI`: solo la parte del host, **sin** el nombre del repositorio. Ejemplo: `123456789.dkr.ecr.us-east-1.amazonaws.com`
> - `ECR_REPOSITORY_NAME`: solo el nombre corto. Ejemplo: `student-performance`
>
> Un error frecuente es poner la URI completa en `ECR_REPOSITORY_NAME`. Esto causará un fallo en el build y push de la imagen.

---

### Paso 9: Ejecutar el pipeline y verificar el despliegue

1. Realiza cualquier cambio en tu código (por ejemplo, edita `app.py`) y haz commit y push a la rama `main`:

```bash
git add .
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

2. Ve a la pestaña **Actions** de tu repositorio en GitHub y observa el flujo:
   - **Continuous Integration** → pasa rápido.
   - **Continuous Delivery** → construye y sube la imagen a ECR (más lento).
   - **Continuous Deployment** → el runner en EC2 descarga y ejecuta la imagen.

3. Una vez completado, abre la URL pública de tu instancia EC2 con el puerto 8080:

```
http://<EC2_PUBLIC_IP>:8080
```

Deberías ver tu aplicación web funcionando.

---

### Paso 10: Configurar el grupo de seguridad (si la app no carga)

Si al acceder a la URL obtienes un error de conexión, verifica las reglas de entrada del grupo de seguridad EC2:

1. En la consola EC2 → selecciona tu instancia → pestaña **Security** → haz clic en el **Security group**.
2. **Edit inbound rules** → **Add rule**:
   - **Type:** Custom TCP
   - **Port range:** 8080
   - **Source:** `0.0.0.0/0` (cualquier IP)
3. Guarda las reglas.

---

### Resumen del flujo completo

```
┌─────────────────────────────────────────────────────────────┐
│  git push → rama main                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Job 1: CI (ubuntu-latest)                                  │
│  - Checkout código                                          │
│  - Lint / Tests unitarios                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ (solo si CI pasa)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Job 2: Build & Push (ubuntu-latest)                        │
│  - Configurar credenciales AWS                              │
│  - Login en ECR                                             │
│  - docker build + docker push → ECR privado                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ (solo si push a ECR tiene éxito)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Job 3: CD (self-hosted runner en EC2)                      │
│  - Login en ECR desde EC2                                   │
│  - docker pull (imagen más reciente)                        │
│  - docker stop + docker run                                 │
│  - docker system prune (limpieza)                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  App accesible en
         http://<EC2_IP>:8080
```

---

### Buenas prácticas y recomendaciones

- **Limpieza de recursos:** Cuando termines de probar, ejecuta estas acciones para evitar costes:
  - EC2: Estado de instancia → **Terminate instance**.
  - ECR: Selecciona el repositorio → **Delete**.
  - IAM: Elimina el usuario de despliegue o desactiva sus claves de acceso.
  - GitHub runner: Settings → Runners → **Force remove**.

- **Escalabilidad:** Si necesitas alta disponibilidad, puedes añadir un **Load Balancer** de AWS delante de la instancia EC2 y escalar horizontalmente con múltiples instancias.

- **Seguridad:**
  - Nunca expongas `AWS_ACCESS_KEY_ID` ni `AWS_SECRET_ACCESS_KEY` en el código fuente.
  - Limita las políticas IAM al mínimo necesario (principio de menor privilegio).
  - Usa HTTPS en producción.

- **Alternativa pública:** Para proyectos personales donde la imagen no necesita ser privada, puedes usar **Docker Hub** en lugar de ECR. Para entornos empresariales, ECR es la opción estándar.

---

Con estos pasos, tu proyecto de Machine Learning tendrá un pipeline CI/CD completo que despliega automáticamente en AWS cada vez que hagas un push a la rama principal, usando contenedores Docker privados en ECR y una instancia EC2 como servidor de producción.

---

## Video 11: Despliegue en Azure Services con Container Registry y Web App

En este apartado aprenderás a desplegar tu aplicación de Machine Learning en **Microsoft Azure** utilizando **Azure Container Registry** (ACR) para almacenar la imagen Docker privada y **Azure Web App for Containers** como servidor, logrando un despliegue seguro y profesional en la nube.

### Arquitectura del despliegue en Azure

```
Máquina local (Docker build) → Azure Container Registry (imagen privada) → Azure Web App for Containers (servidor)
```

El flujo tiene tres grandes etapas:
1. **Preparación local:** Construir la imagen Docker localmente.
2. **Registro privado:** Subir la imagen al Azure Container Registry (ACR).
3. **Despliegue en web:** Crear y configurar Azure Web App para tirar la imagen desde ACR y ejecutarla.

---

### Paso 1: Crear el Azure Container Registry

El Azure Container Registry es el equivalente de AWS ECR. Aquí almacenarás de forma privada tu imagen Docker.

#### 1.1 Acceder a Azure Portal

1. Abre https://portal.azure.com en tu navegador.
2. Inicia sesión con tu cuenta de Microsoft Azure.

#### 1.2 Crear un nuevo Azure Container Registry

1. En la página principal de Azure, busca **Container Registry** en la barra de búsqueda.
2. Haz clic en **Container registry** (el primer resultado).
3. Haz clic en el botón **+ Create** o **Create container registry**.
4. Se abrirá un formulario de creación con varias pestañas.

#### 1.3 Configurar el Azure Container Registry

En la pestaña **Basics**, configura los siguientes parámetros:

| Campo | Valor |
|-------|-------|
| **Subscription** | Selecciona tu suscripción activa (por defecto) |
| **Resource group** | Crea uno nuevo (ej: `rg-ml-deployment`) o selecciona uno existente |
| **Registry name** | Nombre único para tu registro (ej: `mlprojectregistry`) — **debe ser único globalmente en Azure** |
| **Region** | Selecciona la región más cercana a ti (ej: `West US 2`, `North Europe`, etc.) |
| **SKU** | Selecciona `Basic` (suficiente para desarrollo y pequeña escala) |

> **Nota sobre el nombre del registro:** Azure requiere que sea único a nivel mundial y solo puede contener caracteres alfanuméricos en minúsculas. No se permiten guiones ni caracteres especiales.

#### 1.4 Revisar y crear

1. En la pestaña **Networking** y **Encryption**, puedes dejar los valores por defecto.
2. Haz clic en **Review + create**.
3. Revisa el resumen y haz clic en **Create**.
4. Espera a que se complete la creación del recurso (puede tardar 1-2 minutos).

---

### Paso 2: Obtener las credenciales de acceso del Azure Container Registry

Para poder hacer login y push de imágenes al registro, necesitas obtener las credenciales de administrador.

#### 2.1 Acceder a las configuraciones del registro

1. Una vez creado el ACR, ve a **Home** → **All resources** y selecciona tu registro.
2. En el menú izquierdo, busca la sección **Settings** → **Access keys**.

#### 2.2 Habilitar el usuario administrador

1. En la página **Access keys**, verifica que la opción **Admin user** está habilitada (**Enable** si está deshabilitada).
2. Una vez habilitado, se mostrarán tres campos:
   - **Login server:** URL de tu servidor de login (ej: `mlprojectregistry.azurecr.io`)
   - **Username:** Nombre de usuario (usualmente coincide con el nombre del registro)
   - **password:** Contraseña de administrador (es la primera que se genera)
   - **password2:** Segunda contraseña (por seguridad, alternativa)

#### 2.3 Copiar y guardar las credenciales

1. Copia el **Login server** (URL del registro). Lo necesitarás para construir y etiquetar tu imagen Docker.
2. Copia el **Username** y la **password**. Los necesitarás para hacer login desde Docker.

> **Seguridad importante:** Guarda estas credenciales en un lugar seguro. Son sensibles como una contraseña. No las subas a repositorios públicos.

---

### Paso 3: Preparar la imagen Docker en local

Asegúrate de que tienes Docker Desktop instalado y ejecutándose en tu máquina local.

#### 3.1 Verificar que Docker está funcionando

Abre una terminal (PowerShell, CMD o bash) y ejecuta:

```bash
docker ps
```

Deberías ver una lista (vacía o con contenedores) sin errores. Si ves un error, asegúrate de que Docker Desktop está abierto y ejecutándose.

#### 3.2 Construir la imagen Docker localmente

En la raíz de tu proyecto (donde está el `Dockerfile`), ejecuta el siguiente comando:

```bash
docker build -t <login_server>/<nombre_imagen>:latest .
```

**Explicación:**

- `-t`: Asigna un nombre (tag) a la imagen.
- `<login_server>`: Reemplaza con tu URL de ACR (ej: `mlprojectregistry.azurecr.io`)
- `<nombre_imagen>`: Nombre de tu aplicación (ej: `student-performance`)
- `:latest`: Tag de versión (por defecto, latest = última versión)
- `.`: Construir usando el Dockerfile en el directorio actual.

**Ejemplo concreto:**

```bash
docker build -t mlprojectregistry.azurecr.io/student-performance:latest .
```

> **⏱️ Tiempo de construcción:** La primera construcción puede tardar 5-15 minutos, dependiendo del tamaño de tus dependencias. Las construcciones posteriores serán más rápidas gracias al cache de Docker.

#### 3.3 Verificar que la imagen se construyó correctamente

Después de que termine la construcción, ejecuta:

```bash
docker images
```

Deberías ver tu imagen en la lista con el tag que asignaste.

---

### Paso 4: Hacer login en el Azure Container Registry desde Docker

Antes de hacer push de tu imagen, debes autenticarte contra ACR.

#### 4.1 Comando de login

Ejecuta el siguiente comando:

```bash
docker login <login_server>
```

**Ejemplo:**

```bash
docker login mlprojectregistry.azurecr.io
```

#### 4.2 Proporcionar credenciales

El comando te pedirá:

1. **Username:** Introduce el nombre de usuario de ACR (copiado en el Paso 2).
2. **Password:** Introduce la contraseña (copiada en el Paso 2).

Si es correcto, verás un mensaje:

```
Login Succeeded
```

> **Nota:** En Windows, Docker puede pedir permisos elevados. Asegúrate de ejecutar la terminal como administrador si es necesario.

---

### Paso 5: Hacer push de la imagen al Azure Container Registry

Una vez autenticado, puedes subir tu imagen al registro privado de Azure.

#### 5.1 Ejecutar el comando push

```bash
docker push <login_server>/<nombre_imagen>:latest
```

**Ejemplo:**

```bash
docker push mlprojectregistry.azurecr.io/student-performance:latest
```

#### 5.2 Esperar a que se complete

El push puede tardar 2-10 minutos, dependiendo del tamaño de la imagen y la velocidad de tu conexión. Verás un progreso de capas siendo subidas.

Cuando termine, deberías ver:

```
latest: digest: sha256:abcdef1234567890... size: XXXXX
```

#### 5.3 Verificar en Azure Portal

1. Ve a **Home** → **All resources** → Tu **Container Registry**.
2. En el menú izquierdo, selecciona **Services** → **Repositories** (o **Registries** dependiendo de la versión).
3. Deberías ver tu repositorio listado (ej: `student-performance`) con el tag `latest`.

---

### Paso 6: Crear Azure Web App for Containers

Azure Web App for Containers es el servicio que alojará y ejecutará tu contenedor Docker.

#### 6.1 Crear un nuevo recurso

1. En Azure Portal → **Home** → **Create a resource**.
2. Busca **Web App for Containers** en la barra de búsqueda.
3. Haz clic en el resultado y luego en **Create**.

#### 6.2 Configurar los parámetros básicos

En la pestaña **Basics**, configura:

| Campo | Valor |
|-------|-------|
| **Subscription** | Tu suscripción activa |
| **Resource group** | Selecciona el mismo que usaste para ACR (ej: `rg-ml-deployment`) |
| **Name** | Nombre único de la app web (ej: `student-performance-app`) |
| **Publish** | Selecciona **Docker Container** |
| **Operating System** | Selecciona **Linux** |
| **Region** | Selecciona la misma región que tu ACR |
| **App Service Plan** | Selecciona un plan existente o crea uno nuevo. Para desarrollo, **Basic** es suficiente. |

#### 6.3 Configurar Docker

En la pestaña **Docker** (aparece tras seleccionar **Docker Container** en Publish):

| Campo | Valor |
|-------|-------|
| **Image Source** | Selecciona **Azure Container Registry** |
| **Registry** | Selecciona tu ACR (ej: `mlprojectregistry`) |
| **Image** | Selecciona tu imagen (ej: `student-performance`) |
| **Tag** | Selecciona `latest` |
| **Startup Command** | Déjalo en blanco (Docker usará el CMD del Dockerfile) |

#### 6.4 Revisar y crear

1. Haz clic en **Review + create**.
2. Revisa la configuración.
3. Haz clic en **Create**.
4. Espera a que se complete la creación (puede tardar 1-2 minutos).

---

### Paso 7: Verificar que la Web App está funcionando

Una vez creada la Web App, se te redirigirá a una página de resumen o podrás acceder a ella desde **Home** → **All resources** → Tu Web App.

#### 7.1 Obtener la URL pública

En la página de resumen de la Web App, busca el campo **URL** (usualmente en la parte superior derecha). Tendrá un formato como:

```
https://student-performance-app.azurewebsites.net
```

#### 7.2 Acceder a la aplicación

Abre un navegador y ve a esa URL. Normalmente:

1. La primera carga puede tardar 30-60 segundos mientras Azure descarga y inicia el contenedor.
2. Si ves tu aplicación web, ¡felicidades! El despliegue fue exitoso.
3. Si ves un error, consulta los logs (ver **Paso 8**).

---

### Paso 8: Monitorización y resolución de problemas

#### 8.1 Ver los logs de la aplicación

Si la app no funciona correctamente, revisa los logs:

1. En tu Web App → **Monitoring** → **Log stream**.
2. Aquí verás los logs en tiempo real del contenedor en ejecución.
3. Busca mensajes de error que indiquen qué salió mal.

#### 8.2 Problemas comunes

| Problema | Solución |
|----------|----------|
| **Error 502 Bad Gateway** | La app está iniciando. Espera 1-2 minutos. Si persiste, revisa los logs. |
| **Contenedor no inicia** | Verifica que el `CMD` en tu Dockerfile es correcto y que las dependencias en `requirements.txt` están completas. |
| **Error de autenticación** | Asegúrate de que habilitaste el usuario administrador en ACR y que las credenciales son correctas. |
| **Puerto incorrecto** | Por defecto, Azure espera que la app escuche en el puerto 8080. Si tu app usa otro puerto, configúralo en **Settings** → **Configuration** → agrega una variable de entorno. |

#### 8.3 Escalar la aplicación

Si necesitas más recursos (CPU, memoria), puedes cambiar el plan de App Service:

1. En tu Web App → **Scale up (App Service plan)**.
2. Selecciona un plan de mayor capacidad.
3. Haz clic en **Select**.

---

### Paso 9: Despliegues posteriores (actualizar la imagen)

Cada vez que hagas cambios en tu código y quieras actualizar la aplicación:

#### 9.1 Reconstruir la imagen Docker

```bash
docker build -t <login_server>/<nombre_imagen>:latest .
```

#### 9.2 Hacer login y push

```bash
docker login <login_server>
docker push <login_server>/<nombre_imagen>:latest
```

#### 9.3 Reiniciar la Web App (para que tire la imagen más reciente)

En Azure Portal:

1. Ve a tu Web App.
2. Haz clic en **Restart** (botón en la barra superior).
3. Espera a que se reinicie (30-60 segundos).
4. Accede a la URL nuevamente para ver los cambios.

> **Alternativa automática:** Si quieres automatizar esto con GitHub Actions, puedes adaptar el workflow del Video 10 para hacer push a ACR en lugar de ECR. Cambiarías el paso `Build, tag, and push image to Amazon ECR` por uno que use las credenciales de Azure (Access Key y password).

---

### Paso 10: Limpiar recursos (evitar costes)

Cuando termines de probar tu aplicación en Azure, elimina los recursos para evitar costes innecesarios:

#### 10.1 Eliminar la Web App

1. Ve a **Home** → **All resources** → Tu Web App.
2. Haz clic en **Delete**.
3. Confirma la eliminación.

#### 10.2 Eliminar el Azure Container Registry

1. Ve a **Home** → **All resources** → Tu **Container Registry**.
2. Haz clic en **Delete**.
3. Confirma la eliminación.

#### 10.3 Eliminar el App Service Plan (si lo creaste solo para esta app)

1. Ve a **Home** → **All resources** → Tu **App Service Plan**.
2. Haz clic en **Delete**.
3. Confirma la eliminación.

#### 10.4 Eliminar el Resource Group (más sencillo)

Alternativamente, puedes eliminar todo el Resource Group de una vez:

1. Ve a **Home** → **Resource groups** → Tu grupo (ej: `rg-ml-deployment`).
2. Haz clic en **Delete resource group**.
3. Confirma escribiendo el nombre del grupo y haz clic en **Delete**.

> **Aviso importante:** Esto eliminará **todos los recursos** dentro del grupo. Asegúrate de hacer backups si necesitas conservar algo.

---

### Comparativa: AWS vs Azure vs Google Cloud

| Aspecto | AWS | Azure | Google Cloud |
|--------|-----|-------|--------------|
| **Registro de imágenes** | ECR | Azure Container Registry (ACR) | Google Container Registry (GCR) |
| **Servidor HTTP** | EC2 | Web App for Containers | Compute Engine o Cloud Run |
| **Curva de aprendizaje** | Media-Alta | Media | Media |
| **Costes** | Variable según config | Competitive | Competitive |
| **Ecosistema** | Muy amplío | Muy amplio | Amplio |
| **Mejor para** | Empresas grandes, escalabilidad extrema | Empresas con Microsoft Stack | Startups, APIs serverless |

---

### Buenas prácticas para Azure

- **Seguridad:** Nunca compartas las credenciales de ACR en repositorios públicos. Usa Azure Key Vault para almacenarlas de forma segura.
- **Versionado de imágenes:** En lugar de siempre usar `:latest`, usa etiquetas con versión (ej: `:v1.0`, `:v1.1`). Esto facilita rollbacks.
- **Monitorización:** Habilita **Application Insights** en tu Web App para obtener métricas de rendimiento y trazas detalladas.
- **CI/CD automatizado:** Integra GitHub Actions con Azure para automatizar el build, push y despliegue cada vez que hagas commit.
- **Escalado automático:** Configura **autoscale** en tu App Service Plan para escalar automáticamente según la carga.

---

### Resumen del despliegue en Azure

```
┌────────────────────────────────────┐
│  Máquina local                     │
│  docker build → Docker image       │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│  Azure Container Registry (ACR)    │
│  docker login + docker push        │
│  Imagen privada almacenada        │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│  Azure Web App for Containers      │
│  - Descarga imagen de ACR          │
│  - Inicia contenedor               │
│  - Escucha en puerto 8080          │
│  - URL pública: *.azurewebsites.net│
└────────────────────────────────────┘
                   │
                   ▼
        Accesible desde internet
        https://app-name.azurewebsites.net
```

---

Con estos pasos, tu proyecto de Machine Learning estará desplegado en **Microsoft Azure**, listo para producción con una imagen Docker privada, escalabilidad automática y acceso público desde cualquier parte del mundo.

