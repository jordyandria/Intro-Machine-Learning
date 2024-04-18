#################
## squelette fourni dans TP4-mes_fonctions.py :
#################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_blobs(N):
    # data set 3
    N1=N//4
    N2=N-N1
    D=2
    np.random.seed(42)

    # parameters for the 1st blob of points
    mu1=(0.3,0.3)
    sigma1=((1, 0.3))
    X1 = np.random.normal( mu1, sigma1,(N1,D))

    # parameters for the 2nd blob of points
    mu2=(-2,-2)
    sigma2=((2, 0.5))
    X2 = np.random.normal( mu2, sigma2,(N2,D))

    # the two blobs are merged, and labels  +1/-1  are assigned
    Xraw = np.concatenate( (X1, X2) )
    Y = np.concatenate( (np.ones(N1), -np.ones(N2)) ) # .reshape(N,1)
    X = Xraw.copy() # then X will be the extended vector, with the ones added

    X =  np.hstack((np.ones((N,1)), X))  # extended vector

    return X,Y


def display(X, Y, wInit, iteration):

    # def norme(w):
    #     return np.sum(w**2)**0.5
    def norme(w):
        return np.linalg.norm(w)

    # TODO : cette fonction devrait etre obtenue en faisant le TD2/3,
    # dans lequel on calcule la distance plan-droite,
    # et on apprend a maitriser la relation géométrie / vecteurs

    w=wInit.copy()
    w0= w[0] # c'est la partie qui caractérise la distance à l'origine, qui détermine l'ordonnée à l'origine (mais ce n'est pas égal à ça)
    ## on normalise les composantes du vrai vecteur w
    wprime = w[1:] # /(w[1]**2+w[2]**2)**0.5
    print(w, w0, wprime)

    u_w = (wprime/norme(wprime)) # vecteur unitaire donnant la direction
    distance_origine_droite = -w0/norme(wprime)
    projete_de_Origine_sur_droite = u_w * distance_origine_droite
    print("projete_de_Origine_sur_droite", projete_de_Origine_sur_droite)

    vecteur_Orthogonal_A_La_Droite = np.array([wprime[1],-wprime[0]])
    extremite1 = projete_de_Origine_sur_droite + vecteur_Orthogonal_A_La_Droite*10
    extremite2 = projete_de_Origine_sur_droite - vecteur_Orthogonal_A_La_Droite*10
    extremites_abscisses = np.array([extremite1[0], extremite2[0]])
    extremites_ordonnees = np.array([extremite1[1], extremite2[1]])

    plt.figure(1)
    Class1 = X[Y==-1,1:] # points of class "-1"
    Class2 = X[Y== 1,1:] # points of class "+1"
    plt.plot(Class1[:,0], Class1[:,1], 'r+') # points of class "-1"
    plt.plot(Class2[:,0], Class2[:,1], 'bx') # points of class "+1"
    cmap = cm.jet
    colorGradient=cmap(np.linspace(0.0,1.0,12))
    plt.plot(extremites_abscisses, extremites_ordonnees,  color=colorGradient[iteration%(len(colorGradient))])
    print(extremites_abscisses)




## TODO: créer une fonction qui inialise les poids
## (soit aléatoirement, soit de façon pré-fixée (c'est moins bien))
def initializeWeights(X, type):
    pass
    return wparameters

## TODO: remplir cette fonction (au moins, sans les affichages
## mais de sorte que le résultat soit correct)
def perceptronFullBatch_version_minimale(X,Y,eta, w0, maxIter=20, plot=None, verbose=None, Loss=None):
    pass
    return wparameters


## TODO-BONUS: faire aussi une version avec l'affichage
## des positions intermédiaires de la droite séparatrice,
## en couleurs, comme vu en cours.
# def perceptronFullBatch_version_decoree(X,Y,eta, w0, maxIter=20, plot=True, verbose=True, Loss="ReLU"):
#     for iteration in range(maxIter):

#         pass ## TODO ici

#         if plot :
#             display(X,Y, wparameters, iteration)

#     ## TODO ici

#     return wparameters

