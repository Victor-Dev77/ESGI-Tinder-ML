import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Neuron:
    def __init__(self, size):
        self.weights = [0] * size #size est le nombre d'entrées

    def compute(self, inputs):
        '''
        On passe les valeurs des entrées à traiter
        '''
        output = 0
        for i in range(len(inputs)): #JAVA : for(i = 0 ; i < len(inputs) ; i++)
            output += inputs[i] * self.weights[i]

        return output

    def learn(self, inputs, error, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * error * inputs[i]

def load_data(filename):
    f = open(filename)
    # Si utilise dataset_images.csv laisser cette ligne
    # Si utilise myfilename.csv commenter cette ligne !
   # f.readline() #on lit la première ligne d'entêtes pour s'en débarasser
    data = np.loadtxt(f, delimiter=',')
    inputs = data[:, :-1] #toutes les colonnes sauf la dernière
    desired = data[:, -1] #seulement la dernière colonne (le résultat, la sortie souhaitée)
    return inputs, desired

def evaluate_accuracy(nn, inputs, outputs):
    score = 0
    for i in range(len(inputs)):
        output = nn.compute(inputs[i])
        if round(output) == outputs[i]:
            score += 1
    score /= len(inputs) #Erreur RMS
    return score

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

if __name__ == "__main__":
    # Changer nom du fichier entre dataset_images et myfilename

    #Ligne 1 / 2 / 9 a supprimer car 756 pixel au lieu de 784
    inputs, desired = load_data('myfilename.csv')
    inputs /= 255.0 #On normalise les entrées

    #On sépare ensemble d'apprentissage et ensemble de test (ensemble = data)
    TEST_PERCENTAGE = 30
    training_size = int(len(inputs) * (1.0 - TEST_PERCENTAGE / 100.0))
    train_inputs, train_outputs = inputs[:training_size], desired[:training_size] #On coupe le jeu de données en 2
    test_inputs, test_outputs = inputs[training_size:], desired[training_size:]

    # Permet d'affichier graphiquement la 1ere images en noir et blanc
    #fig, ax = plt.subplots()
    #ax.matshow(inputs[0].reshape(28, 28), cmap=plt.cm.gray)
    #plt.show()


    X_train, X_test, y_train, y_test = train_test_split(inputs, desired)

    print(f"desired: {desired}")
    print(len(desired))

    scaler = StandardScaler()
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #mlp = MLPClassifier(hidden_layer_sizes=(150,100,50),
    #                    activation = 'relu',solver='adam',random_state=1,
    #                    max_iter=300)
    #mlp.fit(train_inputs, train_outputs)


    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    mlp.fit(X_train,y_train)

    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

    #print(len(inputs[0])) #Taille du vecteur d'entrée
    #print(len(inputs))  #Nombre d'exemples
    #print(inputs)       #Tableau de données

    """
    n = Neuron(784)   
    for i in range(100): #Nombre d'itération de l'apprentissage => précision de l'apprentissage
        globalError = 0
        for j in range(len(train_inputs)): #On boucle sur l'ensemble des exemples
            output = n.compute(train_inputs[j])
            error = train_outputs[j] - output
            n.learn(train_inputs[j], error, 0.01) #On choisit un taux d'apprentissage arbitraire de 0.1
            globalError += error * error
        
        globalError = sqrt(globalError) / len(train_inputs) #Erreur RMS
        if i % 10 == 0:
            print(f"#{i} Erreur d'apprentissage :", globalError) #Poids après apprentissage
    """

    #Evaluer sur l'ensemble d'apprentissage la qualité de mon modèle
    #learning_score = evaluate_accuracy(n, train_inputs, train_outputs)
    #print(f"#Score d'apprentissage : {round(learning_score * 100)}%")

    #Evaluer sur l'ensemble de test la qualité de mon modèle
    #learning_score = evaluate_accuracy(n, test_inputs, test_outputs)
    #print(f"#Score de test : {round(learning_score * 100)}%")

    learning_score = mlp.score(test_inputs, test_outputs)
    print(f"#Score d'apprentissage test data: {round(learning_score * 100)}%")

    #Evaluer sur l'ensemble d'apprentissage la qualité de mon modèle
    learning_score = mlp.score(train_inputs, train_outputs)
    print(f"#Score d'apprentissage : {round(learning_score * 100)}%")

    #Evaluer sur l'ensemble de test la qualité de mon modèle
    learning_score = mlp.score(inputs, desired)
    print(f"#Score de test : {round(learning_score * 100)}%")


    # Prediction
    arr = [inputs[9], inputs[10]]
    res = [desired[9], desired[10]]
    prediction = mlp.predict(arr)
    print(f"Prediction 10e image: {prediction}")

    cm = confusion_matrix(prediction, res)
    print(f"Accuracy of MLPClassifier : {accuracy(cm)}")