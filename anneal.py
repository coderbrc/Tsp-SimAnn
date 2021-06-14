import math
import random
import visualize_tsp
import matplotlib.pyplot as plt
class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = 500 if T == -1 else T
        self.T_save = self.T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1 if stopping_T == -1 else stopping_T
        self.stopping_iter = 10000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.nodes = [i for i in range(self.N)]
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []
    def initial_solution(self):
        cur_node = random.choice(self.nodes)  # rastgele bir düğüm oluştur
        solution = [cur_node]
        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # en yakın komşu
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # Eğer best fitness uzaksa best fitness'ı güncelle
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit
    def dist(self, node_0, node_1):
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)
    def fitness(self, solution):
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit
    def p_accept(self, candidate_fitness):
        """
        Adayın mevcuttan daha kötü olup olmadığı olasılığı
        mevcut sıcaklığa ve aday ile mevcut arasındaki farka bağlıdır
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)
    def accept(self, candidate):
        """
        Eğer aday mevcuttan daha iyiyse 1 olasılıkla kabul et
        Eğer aday kötüyse p_accept() olasılığı ile kabul et
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate
    def anneal(self):
        """
        SA algoritmasını çalıştırıyoruz
        """
        self.cur_solution, self.cur_fitness = self.initial_solution()
        print("Annealing optimizasyonu başlatılıyor...")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)
        print("Best fitness elde edildi: ", self.best_fitness)

    def visualize_routes(self):
        """
       TSP yolunu matloplit ile görselleştirdik,
        """
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()