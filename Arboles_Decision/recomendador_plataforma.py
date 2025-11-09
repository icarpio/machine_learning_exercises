"""
Recomendador de plataforma
Una empresa de desarrollo de software desea construir un sistema inteligente que recomiende la plataforma más adecuada (web, mobile o desktop) para nuevos proyectos, en base a las características del proyecto y la experiencia previa con proyectos similares.



✅ Objetivo

Implementa un sistema en Python utilizando clases, estructuras de datos y machine learning (árbol de decisión) que sea capaz de predecir la mejor plataforma para un nuevo proyecto.

🧱 Parte 1: Modelar los proyectos

Crea una clase llamada Project con los siguientes atributos:

name: nombre del proyecto (str)
team_size: tamaño del equipo de desarrollo (int)
budget: presupuesto en miles de dólares (float)
duration_months: duración estimada en meses (int)
realtime_required: si requiere tiempo real (bool)
needs_offline: si necesita funcionar sin conexión (bool)
target_users: tipo de usuarios objetivo (str): puede ser "local", "empresa" o "global"
recommended_platform: plataforma recomendada (str): puede ser "web", "mobile" o "desktop"

⚠️ Este último atributo (recommended_platform) solo debe tener valor en proyectos históricos. 
En los nuevos proyectos (aquellos para los que se desea predecir la plataforma), este atributo debe estar en None.

🔧 Método to_features()

Implementa un método llamado to_features(self, label_encoder=None) que convierta un proyecto a una lista de características numéricas, útil para entrenar o hacer predicciones.

✔️ Este método debe convertir realtime_required y needs_offline en 0 o 1, y codificar el atributo target_users con un LabelEncoder.



📊 Parte 2: Construir el dataset

Crea una clase llamada ProjectDataset que reciba una lista de proyectos (históricos y nuevos) y tenga:

Métodos:

_fit_encoders(): crea y entrena internamente dos LabelEncoder: uno para target_users y otro para recommended_platform.
get_X_y(): devuelve dos listas:
X: la matriz de características de los proyectos históricos
y: las plataformas codificadas de los proyectos históricos
decode_platform(label): transforma una predicción numérica a su plataforma original (str)

✔️ Usa LabelEncoder para convertir texto a números y viceversa.
🧠 Parte 3: Entrenar el modelo

Crea una clase llamada PlatformRecommender que:

Use un modelo de DecisionTreeClassifier de scikit-learn.

Tenga dos métodos:

train(dataset: ProjectDataset): entrena el modelo con los datos históricos.
predict(project: Project): predice la plataforma más adecuada para un nuevo proyecto, devolviendo "web", "mobile" o "desktop".

⚠️ Importante: Antes de usar predict(), debes haber llamado a train().



🧪 Parte 4: Prueba tu sistema

Usa los siguientes datos de proyectos históricos:

projects = [
    Project("AppGlobal", 5, 25.0, 6, True, False, "global", "web"),
    Project("IntranetCorp", 10, 40.0, 12, False, True, "empresa", "desktop"),
    Project("LocalDelivery", 3, 20.0, 4, True, True, "local", "mobile"),
    Project("CloudDashboard", 6, 50.0, 8, True, False, "empresa", "web"),
    Project("OfflineTool", 4, 15.0, 6, False, True, "local", "desktop"),
    Project("SocialBuzz", 2, 10.0, 3, True, False, "global", "mobile"),
]
Y predice la plataforma recomendada para este nuevo proyecto:

new_project = Project("AIChatApp", 4, 30.0, 5, True, False, "global")


🌳 Parte 5: Visualiza el árbol de decisión

Crea una clase llamada PlatformRecommenderVisualizer que tenga un método plot_tree() que reciba un objeto PlatformRecommender y genere un gráfico del árbol de decisión.

📌 Puedes usar el siguiente código:

plt.figure(figsize=(12, 6))
tree.plot_tree(
    model,
    feature_names=["team_size", "budget", "duration_months", "realtime_required", "needs_offline", "target_users"],
    class_names=model_classes,
    filled=True,
    rounded=True
)
plt.show()

"""

# Importamos librerías necesarias

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ===========================
# Parte 1: Modelar los proyectos
# ===========================
class Project:
    def __init__(self, name, team_size, budget, duration_months, realtime_required, needs_offline, target_users, recommended_platform=None):
        self.name = name
        self.team_size = team_size
        self.budget = budget
        self.duration_months = duration_months
        self.realtime_required = realtime_required
        self.needs_offline = needs_offline
        self.target_users = target_users
        self.recommended_platform = recommended_platform

    def to_features(self, label_encoder=None):
        """
        Convierte un proyecto en lista de características numéricas:
        - bool -> 0 o 1
        - target_users -> codificado con LabelEncoder
        """
        realtime_val = 1 if self.realtime_required else 0
        offline_val = 1 if self.needs_offline else 0
        
        if label_encoder:
            target_users_val = label_encoder.transform([self.target_users])[0]
        else:
            target_users_val = self.target_users  # solo para entrenamiento inicial
        
        return [self.team_size, self.budget, self.duration_months, realtime_val, offline_val, target_users_val]


# ===========================
# Parte 2: Dataset
# ===========================
class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self._fit_encoders()

    def _fit_encoders(self):
        """
        Creamos LabelEncoders para:
        - target_users
        - recommended_platform
        """
        self.user_encoder = LabelEncoder()
        self.platform_encoder = LabelEncoder()
        
        self.user_encoder.fit([p.target_users for p in self.projects])
        platforms = [p.recommended_platform for p in self.projects if p.recommended_platform is not None]
        self.platform_encoder.fit(platforms)

    def get_X_y(self):
        """
        Devuelve X (features) e y (plataformas codificadas) solo para proyectos históricos
        """
        X, y = [], []
        for p in self.projects:
            if p.recommended_platform is not None:
                X.append(p.to_features(self.user_encoder))
                y.append(self.platform_encoder.transform([p.recommended_platform])[0])
        return X, y

    def decode_platform(self, label):
        """Convierte la predicción numérica de vuelta a nombre de plataforma"""
        return self.platform_encoder.inverse_transform([label])[0]

# ===========================
# Parte 3: Recomendador de plataforma
# ===========================
class PlatformRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.trained = False

    def train(self, dataset: ProjectDataset):
        """
        Entrena el árbol de decisión con proyectos históricos
        """
        X, y = dataset.get_X_y()
        self.model.fit(X, y)
        self.dataset = dataset
        self.trained = True

    def predict(self, project: Project):
        """
        Predice la plataforma más adecuada para un nuevo proyecto
        """
        if not self.trained:
            raise Exception("¡Debes entrenar el modelo primero usando train()!")
        
        X_new = [project.to_features(self.dataset.user_encoder)]
        pred_label = self.model.predict(X_new)[0]
        return self.dataset.decode_platform(pred_label)


# ===========================
# Parte 5: Visualización
# ===========================
class PlatformRecommenderVisualizer:
    @staticmethod
    def plot_tree(recommender: PlatformRecommender):
        """
        Genera gráfico del árbol de decisión
        """
        plt.figure(figsize=(12, 6))
        plot_tree(
            recommender.model,
            feature_names=["team_size", "budget", "duration_months", "realtime_required", "needs_offline", "target_users"],
            class_names=recommender.dataset.platform_encoder.classes_,
            filled=True,
            rounded=True
        )
        plt.show()


# ===========================
# Bloque de prueba (solo se ejecuta si corremos este archivo directamente)
# ===========================
if __name__ == "__main__":
    # Proyectos históricos
    projects = [
        Project("AppGlobal", 5, 25.0, 6, True, False, "global", "web"),
        Project("IntranetCorp", 10, 40.0, 12, False, True, "empresa", "desktop"),
        Project("LocalDelivery", 3, 20.0, 4, True, True, "local", "mobile"),
        Project("CloudDashboard", 6, 50.0, 8, True, False, "empresa", "web"),
        Project("OfflineTool", 4, 15.0, 6, False, True, "local", "desktop"),
        Project("SocialBuzz", 2, 10.0, 3, True, False, "global", "mobile"),
    ]
    
    # Nuevo proyecto
    new_project = Project("AIChatApp", 4, 30.0, 5, True, False, "global")
    
    # Entrenamiento y predicción
    dataset = ProjectDataset(projects)
    recommender = PlatformRecommender()
    recommender.train(dataset)
    prediction = recommender.predict(new_project)
    
    print(f"Plataforma recomendada para '{new_project.name}': {prediction}")
    
    # Visualización del árbol
    visualizer = PlatformRecommenderVisualizer()
    visualizer.plot_tree(recommender)
    
"""


💡 Explicación paso a paso:

Clase Project:
Representa cada proyecto con todas sus características y permite convertirlas a números para el modelo.

to_features():
Convierte bool a 0/1 y texto a números usando LabelEncoder. Es lo que el modelo necesita para aprender.

Clase ProjectDataset:
Administra los proyectos y prepara las matrices X y y para entrenamiento.

X → características numéricas
y → plataformas codificadas

Clase PlatformRecommender:

Usa DecisionTreeClassifier para aprender reglas simples de decisión.
train() entrena el modelo.
predict() predice la plataforma para un nuevo proyecto.

Prueba del sistema:

Definimos proyectos históricos y un nuevo proyecto.
Entrenamos el modelo y hacemos la predicción.

Visualización del árbol:

plot_tree() genera un gráfico donde cada nodo muestra una regla de decisión.
Las hojas muestran la plataforma final.
Esto ayuda a entender por qué el modelo tomó cierta decisión.

"""