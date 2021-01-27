import pickle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def showLearningCurve(mlp):
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    #plt.yscale('log')
    ax.set_title("Loss During GD (Rate=0.001)")
   # fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load IA mlp
    filename = "ia.mlp"
    file = open(filename, 'rb')
    mlp = pickle.load(file)
    file.close()

    showLearningCurve(mlp)


