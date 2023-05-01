import numpy as np
from matplotlib import pyplot as plt


#disaster


def rosenblatt(train_X, train_y, max_iter=10000):
    # Inicializujeme váhy na nulu
    w = np.zeros(train_X.shape[1])
    # Provedeme až max_iter iterací
    for i in range(max_iter):
        # Projdeme všechny trénovací příklady
        for j in range(train_X.shape[0]):
            # Spočteme výstup klasifikační funkce
            output = np.dot(train_X[j], w)
            # Pokud byl klasifikační výstup nesprávný, aktualizujeme váhy
            if np.sign(output) != train_y[j]:
                w += train_y[j] * train_X[j]
        # Pokud jsou všechny trénovací příklady klasifikovány správně, ukončíme učení
        if np.all(np.sign(np.dot(train_X, w)) == train_y):
            break
    return w

def constant_increment(train_X, train_y, alpha=0.1, max_iter=10000):
    # Inicializujeme váhy na nulu
    w = np.zeros(train_X.shape[1])
    # Provedeme až max_iter iterací
    for i in range(max_iter):
        # Projdeme všechny trénovací příklady
        for j in range(train_X.shape[0]):
            # Spočteme výstup klasifikační funkce
            output = np.dot(train_X[j], w)
            # Aktualizujeme váhy na základě konstanty učení a nesprávné klasifikace
            if np.sign(output) != train_y[j]:
                w += alpha * train_y[j] * train_X[j]
    return w

def plot_ros_and_const_incr(train_X, train_y, w_ros, w_const_incr):
    plt.figure(figsize=(8, 8))
    # Vykreslíme trénovací data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='bwr')

    # Vykreslíme lineární diskriminační funkce
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_plot = np.linspace(x_min, x_max, 100)
    y_plot = np.linspace(y_min, y_max, 100)

    y_rosenblatt = -(w_ros[0] * x_plot) / w_ros[1]
    y_constant_increment = -(w_const_incr[0] * x_plot) / w_const_incr[1]
    plt.plot(x_plot, y_rosenblatt, label='Rosenblatt')
    plt.plot(y_plot, y_constant_increment, label='Constant increment')

    plt.show()

#Tyto implementace předpokládají, že train_X je matice trénovacích příkladů, train_y je vektor klasifikačních cílů, alpha je konstanta učení a max_iter je maximální počet iterací, které mají být provedeny. Funkce vracejí váhy lineární diskriminační funkce pro daný trénovací dataset.

#Klasifikátor s lineárními diskriminačními funkcemi slouží k rozdělení dat do dvou tříd na základě jejich lineární separability. Rosenblattův algoritmus a upravená metoda konstantních přírůstků jsou dva způsoby učení tohoto klasifikátoru.
#Rosenblattův algoritmus je iterativní algoritmus, který aktualizuje váhy lineární diskriminační funkce po každém předložení trénovacího příkladu. Algoritmus se ukončí, když jsou všechny trénovací příklady klasifikovány správně, nebo když je dosaženo maximálního počtu iterací.
#Upravená metoda konstantních přírůstků je také iterativní algoritmus, který aktualizuje váhy lineární diskriminační funkce po každém předložení trénovacího příkladu, ale používá konstantního kroku učení namísto adaptivního kroku, který se používá v Rosenblattově algoritmu.
#Potřebný počet iterací závisí na mnoha faktorech, jako jsou složitost dat, velikost trénovací sady, volba konstanty učení a počátečních vah. Obecně platí, že Rosenblattův algoritmus má tendenci se učit rychleji než upravená metoda konstantních přírůstků, ale může být náchylnější k oscilacím a nemusí být schopen konvergovat v některých případech.
#Pro zvolené konstanty učení by bylo třeba vyzkoušet oba algoritmy na trénovacích datech a porovnat počet iterací, které jsou potřebné pro dosažení určité úrovně přesnosti klasifikace. Obecně platí, že volba optimální konstanty učení závisí na konkrétních datech a musí být experimentálně stanovena.