import numpy as np
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    f = open(filename)
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
    inputs, desired = load_data('csv_deepfake.csv') #myfilename #csv_deepfake
    inputs /= 255.0 #On normalise les entrées

    testinput, testdesired = load_data('csv_images.csv') #csv_images
    testinput /= 255.0

    #On sépare ensemble d'apprentissage et ensemble de test (ensemble = data)
    TEST_PERCENTAGE = 30 #30
    training_size = int(len(inputs) * (1.0 - TEST_PERCENTAGE / 100.0))
    train_inputs, train_outputs = inputs[:training_size], desired[:training_size] #On coupe le jeu de données en 2
    test_inputs, test_outputs = inputs[training_size:], desired[training_size:]

    X_train, X_test, y_train, y_test = train_test_split(inputs, desired)

    scaler = StandardScaler()
    scaler.fit(X_train)
    # Applique transformations sur les données:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #mlp = MLPClassifier(hidden_layer_sizes=(150,100,50),
    #                    activation='relu',solver='adam',random_state=1,
    #                    max_iter=300)
    #mlp.fit(train_inputs, train_outputs)


    mlp = MLPClassifier(hidden_layer_sizes=(13, 6,),
                        solver='sgd',
                        max_iter=2000,
                        learning_rate_init=0.01,
                        momentum=0.9,
                        n_iter_no_change=10)
    #MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    #mlp.fit(X_train,y_train)
    mlp.fit(train_inputs, train_outputs)

    deepfake = False
    inputs_test = []
    outputs_test = []
    if deepfake:
        inputs_test = testinput
        outputs_test = testdesired
    else:
        inputs_test = test_inputs
        outputs_test = test_outputs


    np.random.shuffle(inputs_test)
    predictions = mlp.predict(inputs_test)
    print(f"Test: {TEST_PERCENTAGE}%")
    print("Desired: Femme = 0 | Homme = 1")

    for i in range(len(inputs_test)):
        if (outputs_test[i] != predictions[i]):
            print(f"{i}_ desired : {outputs_test[i]} | predict : {predictions[i]} -> Wrong")
        else:
            print(f"{i}_ desired : {outputs_test[i]} | predict : {predictions[i]}")

    #Evaluer sur l'ensemble d'apprentissage la qualité de mon modèle
    learning_score = mlp.score(train_inputs, train_outputs)
    print(f"#Score d'apprentissage : {round(learning_score * 100)}%")

    learning_score = mlp.score(inputs_test, outputs_test)
    print(f"#Score de test : {round(learning_score * 100)}%")

    print(confusion_matrix(outputs_test, predictions, labels=[0, 1]))
    print(classification_report(outputs_test, predictions))

    #Evaluer sur l'ensemble de test la qualité de mon modèle
    learning_score = mlp.score(inputs, desired)
    print(f"Score d'apprentissage input desired : {round(learning_score * 100)}%")

    # Prediction
    arr = [inputs[9]]
    res = [desired[9]]
    prediction = mlp.predict(arr)
    print(f"Prediction 10e image: {prediction} -> res attendu: {res}")

    cm = confusion_matrix(prediction, res)
    print(f"Accuracy of MLPClassifier : {accuracy(cm)}")