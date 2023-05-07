import numpy as np
from matplotlib import pyplot as plt

#predelat cely

def rosenblatt(train_X, train_y, max_iter=10000):
    # Inicializujeme váhy na nulu pro každou třídu
    w = np.zeros((3, train_X.shape[1]))
    # Provedeme až max_iter iterací
    for i in range(max_iter):
        # Projdeme všechny trénovací příklady
        for j in range(train_X.shape[0]):
            # Spočteme výstupy klasifikačních funkcí pro každou třídu
            outputs = np.dot(train_X[j], w.T)
            # Pokud byl klasifikační výstup nesprávný, aktualizujeme váhy pro příslušnou třídu
            if np.argmax(outputs) != train_y[j]:
                correct_class = train_y[j]
                incorrect_class = np.argmax(outputs)
                w[correct_class] += train_X[j]
                w[incorrect_class] -= train_X[j]
        # Pokud jsou všechny trénovací příklady klasifikovány správně, ukončíme učení
        if np.all(np.argmax(np.dot(train_X, w.T), axis=1) == train_y):
            break
    return w


def constant_increment(train_X, train_y, alpha=0.1, max_iter=10000):
    # Inicializujeme váhy na nulu pro každou třídu
    w = np.zeros((3, train_X.shape[1]))
    # Provedeme až max_iter iterací
    for i in range(max_iter):
        # Projdeme všechny trénovací příklady
        for j in range(train_X.shape[0]):
            # Spočteme výstupy klasifikačních funkcí pro každou třídu
            outputs = np.dot(train_X[j], w.T)
            # Aktualizujeme váhy pro příslušnou třídu na základě konstanty učení a nesprávné klasifikace
            if np.argmax(outputs) != train_y[j]:
                correct_class = train_y[j]
                incorrect_class = np.argmax(outputs)
                w[correct_class] += alpha * train_X[j]
                w[incorrect_class] -= alpha * train_X[j]
    return w


def plot_ros_and_const_incr(train_X, train_y, w_ros, w_const_incr):
    plt.figure(figsize=(8, 8))
    # Vykreslíme trénovací data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='viridis')

    # Vykreslíme lineární diskriminační funkce
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    x_plot = np.linspace(x_min, x_max, 100)

    # První přímka
    w1_ros = w_ros[0]
    y_plot1_ros = -(w1_ros[0] * x_plot) / w1_ros[1]
    plt.plot(x_plot, y_plot1_ros, label='Rosenblatt - Třída 1')

    w1_const_incr = w_const_incr[0]
    y_plot1_const_incr = -(w1_const_incr[0] * x_plot) / w1_const_incr[1]
    plt.plot(x_plot, y_plot1_const_incr, label='Metoda konstantních přírůstků - Třída 1')

    # Druhá přímka
    w2_ros = w_ros[1]
    y_plot2_ros = -(w2_ros[0] * x_plot) / w2_ros[1]
    plt.plot(x_plot, y_plot2_ros, label='Rosenblatt - Třída 2')

    w2_const_incr = w_const_incr[1]
    y_plot2_const_incr = -(w2_const_incr[0] * x_plot) / w2_const_incr[1]
    plt.plot(x_plot, y_plot2_const_incr, label='Metoda konstantních přírůstků - Třída 2')

    # Třetí přímka
    w3_ros = w_ros[2]
    y_plot3_ros = -(w3_ros[0] * x_plot) / w3_ros[1]
    plt.plot(x_plot, y_plot3_ros, label='Rosenblatt - Třída 3')

    w3_const_incr = w_const_incr[2]
    y_plot3_const_incr = -(w3_const_incr[0] * x_plot) / w3_const_incr[1]
    plt.plot(x_plot, y_plot3_const_incr, label='Metoda konstantních přírůstků - Třída 3')

    plt.title("Klasifikátor s lineárními diskriminačními funkcemi")
    plt.legend()
    plt.show()


def compare_iterations(train_X, train_y):
    alphas = [0.01, 0.05, 0.1, 0.2]
    iterations_rosenblatt = []
    iterations_const_incr = []

    for alpha in alphas:
        w_ros = rosenblatt(train_X, train_y)
        w_const_incr = constant_increment(train_X, train_y, alpha=alpha)

        # Počet iterací Rosenblattova algoritmu
        iterations_rosenblatt.append(len(w_ros))

        # Počet iterací upravené metody konstantních přírůstků
        iterations_const_incr.append(len(w_const_incr))

    for i, alpha in enumerate(alphas):
        print(f"Alpha: {alpha}")
        print(f"Počet iterací Rosenblattova algoritmu: {iterations_rosenblatt[i]}")
        print(f"Počet iterací upravené metody konstantních přírůstků: {iterations_const_incr[i]}")
        print()
    return iterations_rosenblatt, iterations_const_incr

# Tyto implementace předpokládají, že train_X je matice trénovacích příkladů, train_y je vektor klasifikačních cílů, alpha je konstanta učení a max_iter je maximální počet iterací, které mají být provedeny. Funkce vracejí váhy lineární diskriminační funkce pro daný trénovací dataset.

# Klasifikátor s lineárními diskriminačními funkcemi slouží k rozdělení dat do dvou tříd na základě jejich lineární separability. Rosenblattův algoritmus a upravená metoda konstantních přírůstků jsou dva způsoby učení tohoto klasifikátoru.
# Rosenblattův algoritmus je iterativní algoritmus, který aktualizuje váhy lineární diskriminační funkce po každém předložení trénovacího příkladu. Algoritmus se ukončí, když jsou všechny trénovací příklady klasifikovány správně, nebo když je dosaženo maximálního počtu iterací.
# Upravená metoda konstantních přírůstků je také iterativní algoritmus, který aktualizuje váhy lineární diskriminační funkce po každém předložení trénovacího příkladu, ale používá konstantního kroku učení namísto adaptivního kroku, který se používá v Rosenblattově algoritmu.
# Potřebný počet iterací závisí na mnoha faktorech, jako jsou složitost dat, velikost trénovací sady, volba konstanty učení a počátečních vah. Obecně platí, že Rosenblattův algoritmus má tendenci se učit rychleji než upravená metoda konstantních přírůstků, ale může být náchylnější k oscilacím a nemusí být schopen konvergovat v některých případech.
# Pro zvolené konstanty učení by bylo třeba vyzkoušet oba algoritmy na trénovacích datech a porovnat počet iterací, které jsou potřebné pro dosažení určité úrovně přesnosti klasifikace. Obecně platí, že volba optimální konstanty učení závisí na konkrétních datech a musí být experimentálně stanovena.
