

"""
Cluster 0: 💎🛒 Compradores Premium / High-spenders
- Gastan mucho cada mes
- Visitan la tienda con frecuencia
- Compran marcas caras y productos impulsivos
Ej: Luis, Hugo, Diego, Raúl

Cluster 1: 🟢🛍 Compradores Moderados / Estables
- Gasto intermedio
- Visitas regulares
- Compras equilibradas, pocas impulsivas
Ej: Ana, Marta, Patricia, Teresa, Sofía, Jorge, Valeria

Cluster 2: 🐢💰 Compradores Ahorra-dores / Low-spenders
- Gastan poco cada mes
- Van pocas veces
- Compras racionales, casi nada impulsivas
Ej: Lucía, Bruno, Elena
"""



#  CUSTOMER ANALYTICS – CLUSTERING DE CLIENTES


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

##############################################################
# 1. CLASE CUSTOMER
##############################################################
class Customer:
    def __init__(self, name, monthly_spending, visits_per_month, impulse_rate, premium_preference):
        self.name = name
        self.monthly_spending = monthly_spending         # gasto mensual
        self.visits_per_month = visits_per_month         # visitas al supermercado
        self.impulse_rate = impulse_rate                 # compras impulsivas (0–1)
        self.premium_preference = premium_preference     # gusto por productos premium (0–1)

    def to_vector(self):
        return [
            self.monthly_spending,
            self.visits_per_month,
            self.impulse_rate,
            self.premium_preference
        ]


##############################################################
# 2. CUSTOMERCLUSTERER
##############################################################
class CustomerClusterer:
    def __init__(self):
        self.model = None

    def fit(self, customers, n_clusters):
        data = [c.to_vector() for c in customers]
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(data)

    def predict(self, customer):
        return int(self.model.predict([customer.to_vector()])[0])

    def print_clusters(self, customers):
        labels = self.model.labels_
        clusters = {}

        for c, label in zip(customers, labels):
            clusters.setdefault(label, []).append(c.name)

        for cid, names in clusters.items():
            print(f"Cluster {cid}:")
            for n in names:
                print(f"  - {n}")
            print()

    def plot(self, customers):
        labels = self.model.labels_
        x = [c.monthly_spending for c in customers]
        y = [c.visits_per_month for c in customers]

        plt.scatter(x, y, c=labels, s=120, cmap="viridis")
        for i, c in enumerate(customers):
            plt.text(x[i] + 2, y[i] + 0.1, c.name)
        plt.xlabel("Gasto mensual (€)")
        plt.ylabel("Visitas por mes")
        plt.title("Segmentación de Clientes")
        plt.grid(True)
        plt.show()


##############################################################
# 3. SUPERMARKET ANALYTICS
##############################################################
class SupermarketAnalytics:
    def __init__(self):
        data = [
            ("Ana", 120, 12, 0.3, 0.2),
            ("Luis", 500, 20, 0.6, 0.9),
            ("Carla", 90, 8, 0.25, 0.1),
            ("Mario", 300, 15, 0.5, 0.7),
            ("Lucía", 70, 6, 0.1, 0.05),
            ("Hugo", 450, 18, 0.55, 0.8),
            ("Rosa", 110, 10, 0.28, 0.15),
            ("Pedro", 250, 14, 0.35, 0.4),
            ("Sofía", 160, 13, 0.32, 0.25),   # compradora media, estable
            ("Diego", 520, 22, 0.65, 0.95),   # comprador heavy + premium
            ("Elena", 85, 7, 0.2, 0.12),      # perfil ahorrador
            ("Jorge", 310, 16, 0.48, 0.6),    # comprador habitual medio
            ("Marta", 140, 11, 0.33, 0.3),    # equilibrada
            ("Raúl", 470, 19, 0.58, 0.85),    # similar a Hugo
            ("Patricia", 200, 12, 0.40, 0.35),# clase media del supermercado
            ("Bruno", 65, 5, 0.12, 0.08),     # muy ahorrador
            ("Teresa", 260, 14, 0.37, 0.45),  # consumidora estable
            ("Valeria", 330, 17, 0.52, 0.5)   # comprador fuerte balanceado
        ]

        self.customers = [Customer(*row) for row in data]
        self.clusterer = CustomerClusterer()

    def run(self):
        self.clusterer.fit(self.customers, 3)
        self.clusterer.print_clusters(self.customers)

        new_customer = Customer("Daniel", 200, 9, 0.45, 0.3)
        print("El cliente Daniel pertenece al cluster:",
              self.clusterer.predict(new_customer))

        self.clusterer.plot(self.customers)


SupermarketAnalytics().run()
