import matplotlib.pyplot as plt


def plot_preference(x, y, label):
    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, y, label='Preference Function')
    ax1.set_ylim((0, 100))
    ax1.set_title(label)
    ax1.set_xlabel(label)
    ax1.set_ylabel('Preference function outcome')
    ax1.legend()
    ax1.grid()


def plot_score(c, p, c_res, p_res):
    c_res = [c_res]
    fig = plt.figure(dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(c, p, label='Preference Function')
    for i in range(len(c_res)):
        ax1.scatter(c_res[i], float(p_res), label="Label", s=150)
    ax1.set_ylim((0, 100))
    ax1.set_title('Profit', fontsize=16)
    ax1.set_xlabel(r'Daily profit [€ per day]', fontsize=14)
    ax1.set_ylabel('Preference function outcome', fontsize=14)
    ax1.legend()
    ax1.grid()
