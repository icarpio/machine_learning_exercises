import unittest
from sklearn.datasets import load_iris
from kmedias import entrenar_y_evaluar_kmeans  # Cambia por el nombre del archivo donde guardes tu función

class TestKMeans(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.k = 3

    def test_resultados(self):
        resultados = entrenar_y_evaluar_kmeans(self.X, self.y, self.k)
        
        # Verifica que las claves existan
        self.assertIn("clusters", resultados)
        self.assertIn("inertia", resultados)
        self.assertIn("silhouette_score", resultados)
        self.assertIn("adjusted_rand_score", resultados)

        # Verifica tipos
        self.assertIsInstance(resultados["clusters"], (list, tuple, type(self.X[:,0])))  # numpy array o lista
        self.assertIsInstance(resultados["inertia"], float)
        self.assertIsInstance(resultados["silhouette_score"], float)
        self.assertIsInstance(resultados["adjusted_rand_score"], float)

if __name__ == '__main__':
    unittest.main()