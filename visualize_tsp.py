import matplotlib.pyplot as plt
def plotTSP(paths, points, num_iters=1):
    """
    path: Düğümlerin ziyaret edildiği farklı sıralara sahip listelerin listesi
    points: Farklı düğümler için koordinatlar
    num_iters: Yol listesindeki yol sayısı
    """
    x = []
    y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    plt.plot(x, y, 'co')
    a_scale = float(max(x))/float(100)
    # Eski yolları çizdik
    if num_iters > 1:
        for i in range(1, num_iters):
            # Eski yolları bir listeye dönüştürdük
            xi = []
            yi = []
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])
            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                    head_width=a_scale, color='r',
                    length_includes_head=True, ls='dashed',
                    width=0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width=a_scale, color='r', length_includes_head=True,
                        ls='dashed', width=0.001/float(num_iters))
    # Tsp problemi için birincil yolu çiz
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
            color='g', length_includes_head=True)
    for i in range(0, len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width=a_scale,
                color='g', length_includes_head=True)
    plt.xlim(min(x)*1.1, max(x)*1.1)
    plt.ylim(min(y)*1.1, max(y)*1.1)
    plt.show()
