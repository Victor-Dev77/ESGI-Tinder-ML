import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle

def load_data(filename):
    f = open(filename)
    data = np.loadtxt(f, delimiter=',')
    inputs = data[:, :-1] #toutes les colonnes sauf la dernière
    desired = data[:, -1] #seulement la dernière colonne (le résultat, la sortie souhaitée)
    return inputs, desired

def showLearningCurve(mlp):
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    #plt.yscale('log')
    ax.set_title("Loss During GD (Rate=0.001)")
    #fig.tight_layout()
    plt.show()

if __name__ == "__main__":


    """
        EXTRACTION DES DONNEES
    """

    deepfake_inputs, deepfake_desired = load_data('csv_deepfake.csv') #csv_deepfake
    deepfake_inputs /= 255.0 #On normalise les entrées

    res_inputs_test, res_output_test = load_data('csv_images.csv') #csv_images
    res_inputs_test /= 255.0

    deepfake = True
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



    """
        APPRENTISSAGE
    """
    
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
    filename = "ia.mlp" #"ia_72p.mlp"
    file = open(filename, 'rb')
    mlp = pickle.load(file)
    file.close()
    #"""

    print(len(train_inputs))
    #mlp.fit(train_inputs, train_outputs)

    #Save model
    """
    filename = "ia_full_real.mlp"
    file = open(filename, "wb")
    pickle.dump(mlp, file)
    file.close()
    """




    """
        PREDICTIONS
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




    """
        RESULTATS
    """
    #Evaluer sur l'ensemble d'apprentissage la qualité de mon modèle
    learning_score = mlp.score(train_inputs, train_outputs)
    print(f"#Score d'apprentissage : {round(learning_score * 100)}%")

    learning_score = mlp.score(inputs_test, outputs_test)
    print(f"#Score de test : {round(learning_score * 100)}%")

    print("   F  H")
    print(confusion_matrix(outputs_test, predictions, labels=[0, 1]))
    print(classification_report(outputs_test, predictions, zero_division=0))

    #showLearningCurve(mlp)
