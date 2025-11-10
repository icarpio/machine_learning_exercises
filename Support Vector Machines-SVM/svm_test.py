import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from solution import entrenar_y_evaluar_svm

class TestSVM(unittest.TestCase):
    def setUp(self):
        digits = load_digits()
        X = digits.data
        y = digits.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    """
    Comprueba que la función entrenar_y_evaluar_svm devuelve buenos resultados.
    Verifica que el modelo tenga una precisión (accuracy) mayor o igual al 90%.
    👉 Si el modelo tiene menos de 0.9 de precisión, la prueba falla automáticamente.
    """
    def test_accuracy(self):
        resultados = entrenar_y_evaluar_svm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertGreaterEqual(resultados["accuracy"], 0.9)
        
    """
    Esto asegura que el número de predicciones sea igual al número de ejemplos de prueba.
    👉 Si hay más o menos predicciones que ejemplos reales, significa que algo anda mal en el código.
    """
    def test_predictions_length(self):
        resultados = entrenar_y_evaluar_svm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertEqual(len(resultados["predicciones"]), len(self.y_test))

if __name__ == '__main__':
    unittest.main()
    
    
