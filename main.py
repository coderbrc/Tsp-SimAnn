from anneal import SimAnneal
def read_coords(path):#Coord.txt dosyasından koordinatlar okunup coords listesi oluşturulur.
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords
if __name__ == "__main__":
    coords = read_coords("coord.txt")
    sa = SimAnneal(coords, stopping_iter=10000)
    sa.anneal()
    sa.visualize_routes()