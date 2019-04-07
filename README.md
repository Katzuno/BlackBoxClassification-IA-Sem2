# BlackBoxClassification-IA-Sem2
University project for blackbox classifcation in machine-learning

# DOCUMENTATION

Descrierea problemei : “A black-box learning challenge in which competitors train a
classifier on a dataset that is not human readable, without knowledge of what the data consists of.
They are scored based on classification accuracy on a given test set.”
Scor obtinut pe datele de test (20%) : 0.835
Solutia propusa:
Pentru clasificarea datelor am aplicat o retea de perceptron (MLPClassifier) astfel:
Initial am incarcat datele folosind libraria pandas. Din teste precedente, prin
studierea matricei de confuzie am constatat ca datele sunt relative inconsistente
pe clasele 5, 6, 7, dar in mod special pe 5 si 7, iar (probabil) aceasta este una
dintre cauzele pentru care aveam foarte multe clasificari gresite pe aceste date.
Pentru rezolvarea acestei probleme am triplet datele clasificate cu 5 si 7, iar
datele clasificate cu 6 au fost dublate.
Dupa aceasta operatiune am dublat dataset-ul pentru a oferi retelei ocazia sa
“invete” mai bine corelarile dintre date, intrucat am aplicat si un “noise”. De
asemenea, am amestecat datele dupa aceasta operatiune pentru a simula un
mediu mai “realist” de invatare.
Plecand de la idea scrisului de mana care difera de la persoana la persoana, am
presupus ca datele de test pot fi distorsionate. Astfel am aplicat un “gaussian
noise” intre 0 si 0.07 cu ajutorul numpy, de dimensiunea dataset-ului, pe care lam adaugat datelor mele.
Avand datele procesate, am impartit dataset-ul in date de antrenare (80%) si
testare (20%) si voi antrena reteaua de perceptroni pe datele de antrenare si voi
testa pe cele de validare.
Pentru reteaua de perceptroni am ales sa construiesc 2 straturi ascunse de 200
respectiv 150, functia de activare “Rectified Linear Unit” (Relu) si algoritmul de
compilare Adam (asemanator cu Stochastic Gradient Descent insa cu learning rate
adaptive). De asemenea am folosit parametrul “shuffle=True” pentru a nu testa
mereu pe aceleasi date, intrucat imi pot face o idee mai clara de performanta 
algoritmului, iar toleranta pentru Early Stopping am lasat-o la valoarea de
0.0001(1e-4) pentru a preveni overfitting-ul.
Am calculat matricile de confuzie atat pentru datele de antrenare cat si de
testare, pentru a imi face o idee in privinta overfitting-ului.
In final am testat prin metoda 3-Fold Cross Validation folosind functia
cross_val_score din sklearn pentru a avea o metrica mai buna in privinta preciziei
algoritmului, obtinand o acuratete medie de “0.9711” si o deviatie standard de
“0.0050”.
COD SURSA: 
