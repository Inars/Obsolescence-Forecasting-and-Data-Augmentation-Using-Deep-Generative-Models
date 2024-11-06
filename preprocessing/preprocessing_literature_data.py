import matplotlib.pyplot as plt
from sklearn import datasets

def main():
    n_samples = 10000  # number of samples to generate
    noise = 0.1  # noise to add to sample locations
    x, y = datasets.make_moons(n_samples=n_samples, noise=noise)

    # data contains x y pairs
    data = list(zip(x, y))

    # save data to a csv file
    with open("../data/moons.csv", "w") as f:
        f.write("x1,x2,label\n")
        for point, c in data:
            f.write(f"{point[0]},{point[1]},{c}\n")

    plt.scatter(*x.T, c=y, cmap=plt.cm.Accent)
    plt.title("moons dataset: %d samples" % n_samples)
    plt.show()

if __name__ == '__main__':
    main()