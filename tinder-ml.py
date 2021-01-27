import numpy as np
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

def load_data(filename):
    f = open(filename)
    data = np.loadtxt(f, delimiter=',')
    inputs = data[:, :-1] #toutes les colonnes sauf la dernière
    desired = data[:, -1] #seulement la dernière colonne (le résultat, la sortie souhaitée)
    return inputs, desired

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

def showLearningCurve(mlp):
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    #plt.yscale('log')
    ax.set_title("Loss During GD (Rate=0.001)")
   # fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Changer nom du fichier entre dataset_images et myfilename

    deepfake_inputs, deepfake_desired = load_data('csv_deepfake.csv') #myfilename #csv_deepfake
    deepfake_inputs /= 255.0 #On normalise les entrées

    res_inputs_test, res_output_test = load_data('csv_images.csv') #csv_images
    res_inputs_test /= 255.0

    deepfake = False
    TEST_PERCENTAGE = 100

    inputs_test = []
    outputs_test = []
    train_inputs, train_outputs = [], []
    if deepfake:
        inputs_test = res_inputs_test
        outputs_test = res_output_test
        train_inputs, train_outputs = deepfake_inputs, deepfake_desired
    else:
        #On sépare ensemble d'apprentissage et ensemble de test (ensemble = data)
        TEST_PERCENTAGE = 30 #30
        training_size = int(len(deepfake_inputs) * (1.0 - TEST_PERCENTAGE / 100.0))
        train_inputs, train_outputs = deepfake_inputs[:training_size], deepfake_desired[:training_size] #On coupe le jeu de données en 2
        test_inputs, test_outputs = deepfake_inputs[training_size:], deepfake_desired[training_size:]
        inputs_test = test_inputs
        outputs_test = test_outputs

    
    X_train, X_test, y_train, y_test = train_test_split(deepfake_inputs, deepfake_desired, test_size=0.3, random_state=1)

    """
    mlp = MLPClassifier(hidden_layer_sizes=(100,), #40
                        solver='sgd', #adam
                        activation = 'relu', #logistic
                        max_iter= 1000,
                        #tol=1e-4,
                        #random_state=1,
                        shuffle=True,
                        learning_rate='constant', #adaptive
                        learning_rate_init=0.01, #0.1
                        momentum=0.9, #0.9
                        n_iter_no_change=10)
    """

    # Load IA mlp
    #"""
    filename = "ia_72p.mlp"
    file = open(filename, 'rb')
    mlp = pickle.load(file)
    file.close()
    #"""

    print(len(train_inputs)) #84 #120
    #mlp.fit(train_inputs, train_outputs)


    #Save model
    """
    filename = "ia.mlp"
    file = open(filename, "wb")
    pickle.dump(mlp, file)
    file.close()
    """

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

    print("   F  H")
    print(confusion_matrix(outputs_test, predictions, labels=[0, 1]))
    print(classification_report(outputs_test, predictions, zero_division=0))

    #Evaluer sur l'ensemble de test la qualité de mon modèle
    learning_score = mlp.score(deepfake_inputs, deepfake_desired)
    print(f"Score d'apprentissage input desired : {round(learning_score * 100)}%")

    res = accuracy_score(outputs_test, predictions)
    print(f"accurancy --> {round(res * 100)}%")

  #  showLearningCurve(mlp)

    """fig, ax = plt.subplots(1, 1, figsize=(15,6))
    ax.imshow(np.transpose(mlp.coefs_[0]), cmap=plt.get_cmap("gray"), aspect="auto")
    plt.show()"""
    """
    hidden_2 = np.transpose(mlp.coefs_[0])[16]  # Pull weightings on inputs to the 2nd neuron in the first hidden layer
    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.imshow(np.reshape(hidden_2, (30,30)), cmap=plt.get_cmap("gray"), aspect="auto")
    plt.show()
    """