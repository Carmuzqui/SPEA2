# """
# SPEA2 (Strength Pareto Evolutionary Algorithm 2) - Implementação Completa
# Baseado no artigo: Zitzler, E., Laumanns, M., & Thiele, L. (2001)

# Esta implementação corrige as limitações do SPEA original:
# 1. Novo esquema de fitness com dois componentes (Raw Fitness + Density)
# 2. Truncamento por distância em vez de clustering
# 3. Melhor preservação da diversidade
# 4. Tratamento adequado de soluções não-dominadas

# Autor: Implementação baseada em Valdecy Pereira, adaptada para SPEA2
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from copy import deepcopy
# import math

# class SPEA2:
#     def __init__(self, population_size=100, archive_size=50, generations=250, 
#                  mutation_rate=0.1, crossover_rate=0.9, k_neighbors=1):
#         """
#         Inicializa o algoritmo SPEA2
        
#         Parâmetros:
#         - population_size: Tamanho da população
#         - archive_size: Tamanho fixo do arquivo externo
#         - generations: Número de gerações
#         - mutation_rate: Taxa de mutação
#         - crossover_rate: Taxa de cruzamento
#         - k_neighbors: Número de vizinhos para cálculo de densidade
#         """
#         self.population_size = population_size
#         self.archive_size = archive_size
#         self.generations = generations
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.k_neighbors = k_neighbors
        
#     def target_function(self, variables):
#         """
#         Função objetivo - Problema de Kursawe (3 objetivos)
#         Minimização de ambos os objetivos
#         """
#         f1 = sum(-10 * np.exp(-0.2 * np.sqrt(variables[i]**2 + variables[i+1]**2)) 
#                  for i in range(len(variables)-1))
#         f2 = sum(abs(variables[i])**0.8 + 5 * np.sin(variables[i]**3) 
#                  for i in range(len(variables)))
#         return [f1, f2]
    
#     def initial_population(self, n_variables, min_values, max_values):
#         """
#         Gera população inicial aleatória
#         """
#         population = []
#         for _ in range(self.population_size):
#             individual = []
#             for j in range(n_variables):
#                 individual.append(random.uniform(min_values[j], max_values[j]))
#             population.append(individual)
#         return population
    
#     def dominates(self, solution_a, solution_b):
#         """
#         Verifica se solution_a domina solution_b (para minimização)
        
#         Retorna True se solution_a domina solution_b:
#         - solution_a é melhor ou igual em todos os objetivos
#         - solution_a é estritamente melhor em pelo menos um objetivo
#         """
#         objectives_a = solution_a['objectives']
#         objectives_b = solution_b['objectives']
        
#         # Verifica se A é melhor ou igual em todos os objetivos
#         better_or_equal = all(a <= b for a, b in zip(objectives_a, objectives_b))
        
#         # Verifica se A é estritamente melhor em pelo menos um objetivo
#         strictly_better = any(a < b for a, b in zip(objectives_a, objectives_b))
        
#         return better_or_equal and strictly_better
    
#     def calculate_strength_values(self, population):
#         """
#         Calcula valores de força S(i) para cada indivíduo
#         S(i) = número de soluções que i domina
#         """
#         for individual in population:
#             strength = 0
#             for other in population:
#                 if self.dominates(individual, other):
#                     strength += 1
#             individual['strength'] = strength
    
#     def calculate_raw_fitness(self, population):
#         """
#         Calcula Raw Fitness R(i) para cada indivíduo
#         R(i) = soma dos valores S(j) de todos os j que dominam i
        
#         Esta é a principal melhoria do SPEA2:
#         - Soluções não-dominadas têm R(i) = 0
#         - Soluções dominadas têm R(i) > 0
#         """
#         for individual in population:
#             raw_fitness = 0
#             for other in population:
#                 if self.dominates(other, individual):
#                     raw_fitness += other['strength']
#             individual['raw_fitness'] = raw_fitness
    
#     def euclidean_distance(self, obj1, obj2):
#         """
#         Calcula distância euclidiana entre dois vetores de objetivos
#         """
#         return math.sqrt(sum((a - b)**2 for a, b in zip(obj1, obj2)))
    
#     def calculate_density(self, population):
#         """
#         Calcula componente de densidade D(i) para cada indivíduo
#         D(i) = 1 / (σᵢᵏ + 2)
#         onde σᵢᵏ é a distância ao k-ésimo vizinho mais próximo
        
#         Esta é a segunda melhoria principal do SPEA2
#         """
#         for i, individual in enumerate(population):
#             # Calcula distâncias para todos os outros indivíduos
#             distances = []
#             for j, other in enumerate(population):
#                 if i != j:
#                     dist = self.euclidean_distance(
#                         individual['objectives'], 
#                         other['objectives']
#                     )
#                     distances.append(dist)
            
#             # Ordena distâncias e pega a k-ésima menor
#             distances.sort()
#             k_distance = distances[min(self.k_neighbors - 1, len(distances) - 1)]
            
#             # Calcula densidade
#             density = 1.0 / (k_distance + 2.0)
#             individual['density'] = density
    
#     def calculate_fitness(self, population):
#         """
#         Calcula fitness final F(i) = R(i) + D(i)
        
#         Esta é a fórmula completa do SPEA2:
#         - Menor fitness = melhor solução
#         - Prioriza dominância (R) sobre diversidade (D)
#         """
#         # Passo 1: Calcular valores de força
#         self.calculate_strength_values(population)
        
#         # Passo 2: Calcular raw fitness
#         self.calculate_raw_fitness(population)
        
#         # Passo 3: Calcular densidade
#         self.calculate_density(population)
        
#         # Passo 4: Calcular fitness final
#         for individual in population:
#             individual['fitness'] = individual['raw_fitness'] + individual['density']
    
#     def environmental_selection(self, population, archive):
#         """
#         Seleção ambiental do SPEA2
        
#         Etapas:
#         1. Copia todas as soluções não-dominadas para o novo arquivo
#         2. Se arquivo < tamanho_desejado: preenche com melhores dominadas
#         3. Se arquivo > tamanho_desejado: trunca usando distâncias
#         """
#         # Combina população e arquivo atual
#         combined = population + archive
        
#         # Calcula fitness para todos
#         self.calculate_fitness(combined)
        
#         # Separa não-dominadas (raw_fitness = 0)
#         non_dominated = [ind for ind in combined if ind['raw_fitness'] == 0]
#         dominated = [ind for ind in combined if ind['raw_fitness'] > 0]
        
#         new_archive = []
        
#         # Caso 1: Não-dominadas cabem no arquivo
#         if len(non_dominated) <= self.archive_size:
#             new_archive.extend(non_dominated)
            
#             # Preenche com melhores dominadas se necessário
#             if len(new_archive) < self.archive_size:
#                 dominated.sort(key=lambda x: x['fitness'])
#                 remaining = self.archive_size - len(new_archive)
#                 new_archive.extend(dominated[:remaining])
        
#         # Caso 2: Muitas não-dominadas - precisa truncar
#         else:
#             new_archive = self.truncate_archive(non_dominated)
        
#         return new_archive
    
#     def truncate_archive(self, non_dominated):
#         """
#         Truncamento por distância do SPEA2
        
#         Remove iterativamente a solução com menor distância ao vizinho mais próximo
#         Preserva soluções de fronteira (extremos)
#         """
#         archive = deepcopy(non_dominated)
        
#         while len(archive) > self.archive_size:
#             # Calcula distância ao vizinho mais próximo para cada solução
#             min_distances = []
            
#             for i, individual in enumerate(archive):
#                 distances = []
#                 for j, other in enumerate(archive):
#                     if i != j:
#                         dist = self.euclidean_distance(
#                             individual['objectives'], 
#                             other['objectives']
#                         )
#                         distances.append(dist)
                
#                 min_dist = min(distances) if distances else float('inf')
#                 min_distances.append((min_dist, i))
            
#             # Remove o indivíduo com menor distância ao vizinho
#             min_distances.sort()
#             index_to_remove = min_distances[0][1]
#             archive.pop(index_to_remove)
        
#         return archive
    
#     def tournament_selection(self, population, tournament_size=2):
#         """
#         Seleção por torneio baseada no fitness
#         """
#         tournament = random.sample(population, tournament_size)
#         return min(tournament, key=lambda x: x['fitness'])
    
#     def crossover(self, parent1, parent2):
#         """
#         Cruzamento aritmético
#         """
#         if random.random() > self.crossover_rate:
#             return deepcopy(parent1), deepcopy(parent2)
        
#         alpha = random.random()
#         child1_vars = []
#         child2_vars = []
        
#         for i in range(len(parent1['variables'])):
#             var1 = alpha * parent1['variables'][i] + (1 - alpha) * parent2['variables'][i]
#             var2 = (1 - alpha) * parent1['variables'][i] + alpha * parent2['variables'][i]
#             child1_vars.append(var1)
#             child2_vars.append(var2)
        
#         child1 = {'variables': child1_vars}
#         child2 = {'variables': child2_vars}
        
#         return child1, child2
    
#     def mutation(self, individual, min_values, max_values):
#         """
#         Mutação gaussiana
#         """
#         mutated = deepcopy(individual)
        
#         for i in range(len(mutated['variables'])):
#             if random.random() < self.mutation_rate:
#                 # Mutação gaussiana com 10% do range
#                 range_val = max_values[i] - min_values[i]
#                 mutation_strength = 0.1 * range_val
                
#                 mutated['variables'][i] += random.gauss(0, mutation_strength)
                
#                 # Garante que está dentro dos limites
#                 mutated['variables'][i] = max(min_values[i], 
#                                             min(max_values[i], mutated['variables'][i]))
        
#         return mutated
    
#     def evaluate_population(self, population):
#         """
#         Avalia objetivos para toda a população
#         """
#         for individual in population:
#             individual['objectives'] = self.target_function(individual['variables'])
    
#     def optimize(self, n_variables=3, min_values=None, max_values=None, verbose=True):
#         """
#         Executa o algoritmo SPEA2 completo
#         """
#         if min_values is None:
#             min_values = [-5.0] * n_variables
#         if max_values is None:
#             max_values = [5.0] * n_variables
        
#         # Inicialização
#         population = []
#         for individual_vars in self.initial_population(n_variables, min_values, max_values):
#             individual = {'variables': individual_vars}
#             population.append(individual)
        
#         # Avalia população inicial
#         self.evaluate_population(population)
        
#         # Arquivo inicial vazio
#         archive = []
        
#         # Histórico para análise
#         history = {'generations': [], 'archive_sizes': [], 'best_fitness': []}
        
#         # Loop principal
#         for generation in range(self.generations):
#             # Seleção ambiental (atualiza arquivo)
#             archive = self.environmental_selection(population, archive)
            
#             # Gera nova população
#             new_population = []
            
#             # Pool de seleção = arquivo + população
#             selection_pool = archive + population
#             self.calculate_fitness(selection_pool)
            
#             while len(new_population) < self.population_size:
#                 # Seleção por torneio
#                 parent1 = self.tournament_selection(selection_pool)
#                 parent2 = self.tournament_selection(selection_pool)
                
#                 # Cruzamento
#                 child1, child2 = self.crossover(parent1, parent2)
                
#                 # Mutação
#                 child1 = self.mutation(child1, min_values, max_values)
#                 child2 = self.mutation(child2, min_values, max_values)
                
#                 new_population.extend([child1, child2])
            
#             # Limita ao tamanho da população
#             population = new_population[:self.population_size]
            
#             # Avalia nova população
#             self.evaluate_population(population)
            
#             # Registra estatísticas
#             if archive:
#                 best_fitness = min(ind.get('fitness', float('inf')) for ind in archive)
#                 history['best_fitness'].append(best_fitness)
#             else:
#                 history['best_fitness'].append(float('inf'))
            
#             history['generations'].append(generation)
#             history['archive_sizes'].append(len(archive))
            
#             # Log de progresso
#             if verbose and generation % 50 == 0:
#                 print(f"Geração {generation}: Arquivo com {len(archive)} soluções")
        
#         # Seleção ambiental final
#         final_archive = self.environmental_selection(population, archive)
        
#         return final_archive, history
    
#     def plot_results(self, archive, history, save_plot=True):
#         """
#         Visualiza resultados do SPEA2
#         """
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
#         # Gráfico 1: Frente de Pareto final
#         if archive:
#             obj1_vals = [ind['objectives'][0] for ind in archive]
#             obj2_vals = [ind['objectives'][1] for ind in archive]
            
#             ax1.scatter(obj1_vals, obj2_vals, c='red', alpha=0.7, s=50)
#             ax1.set_xlabel('Objetivo 1 (f1)')
#             ax1.set_ylabel('Objetivo 2 (f2)')
#             ax1.set_title(f'Frente de Pareto Final - SPEA2\n{len(archive)} soluções')
#             ax1.grid(True, alpha=0.3)
        
#         # Gráfico 2: Evolução do tamanho do arquivo
#         ax2.plot(history['generations'], history['archive_sizes'], 'b-', linewidth=2)
#         ax2.set_xlabel('Geração')
#         ax2.set_ylabel('Tamanho do Arquivo')
#         ax2.set_title('Evolução do Tamanho do Arquivo')
#         ax2.grid(True, alpha=0.3)
        
#         # Gráfico 3: Evolução do melhor fitness
#         ax3.plot(history['generations'], history['best_fitness'], 'g-', linewidth=2)
#         ax3.set_xlabel('Geração')
#         ax3.set_ylabel('Melhor Fitness')
#         ax3.set_title('Evolução do Melhor Fitness')
#         ax3.grid(True, alpha=0.3)
        
#         # Gráfico 4: Distribuição de fitness no arquivo final
#         if archive:
#             fitness_vals = [ind.get('fitness', 0) for ind in archive]
#             ax4.hist(fitness_vals, bins=20, alpha=0.7, color='purple')
#             ax4.set_xlabel('Fitness')
#             ax4.set_ylabel('Frequência')
#             ax4.set_title('Distribuição de Fitness no Arquivo Final')
#             ax4.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         if save_plot:
#             plt.savefig('spea2_results.png', dpi=300, bbox_inches='tight')
#             print("Gráficos salvos em 'spea2_results.png'")
        
#         plt.show()

# # Exemplo de uso
# if __name__ == "__main__":
#     print("=== SPEA2 - Implementação Completa ===")
#     print("Executando otimização com problema de Kursawe...")
    
#     # Configuração do algoritmo
#     spea2 = SPEA2(
#         population_size=100,
#         archive_size=50,
#         generations=250,
#         mutation_rate=0.1,
#         crossover_rate=0.9,
#         k_neighbors=1
#     )
    
#     # Executa otimização
#     final_archive, history = spea2.optimize(
#         n_variables=3,
#         min_values=[-5.0, -5.0, -5.0],
#         max_values=[5.0, 5.0, 5.0],
#         verbose=True
#     )
    
#     # Exibe resultados
#     print(f"\n=== Resultados Finais ===")
#     print(f"Número de soluções no frente de Pareto: {len(final_archive)}")
    
#     if final_archive:
#         print("\nMelhores 5 soluções:")
#         # Ordena por fitness
#         sorted_archive = sorted(final_archive, key=lambda x: x.get('fitness', float('inf')))
        
#         for i, solution in enumerate(sorted_archive[:5]):
#             obj = solution['objectives']
#             fitness = solution.get('fitness', 'N/A')
#             print(f"  {i+1}. Objetivos: [{obj[0]:.4f}, {obj[1]:.4f}], Fitness: {fitness:.4f}")
    
#     # Visualiza resultados
#     spea2.plot_results(final_archive, history)
    
#     print("\nOtimização concluída!")









############################################################################
# SPEA2 (Strength Pareto Evolutionary Algorithm 2) - Implementação tutorial
# Baseado no código original, mas com implementação real do SPEA2
# Formato idêntico para comparação direta
############################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

# Function: Initialize Variables
def initial_population(population_size=5, min_values=[-5,-5], max_values=[5,5], list_of_functions=[func_1, func_2]):
    population = np.zeros((population_size, len(min_values) + len(list_of_functions)))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population[i,-k] = list_of_functions[-k](list(population[i,0:population.shape[1]-len(list_of_functions)]))
    return population

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions=2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1[-k] <= solution_2[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Strength Values (S) - SPEA2
def strength_values_function(population, number_of_functions=2):    
    strength = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1=population[i,:], solution_2=population[j,:], number_of_functions=number_of_functions):
                    strength[i,0] = strength[i,0] + 1
    return strength

# Function: Raw Fitness (R) - SPEA2 tutorial
def raw_fitness_function(population, strength, number_of_functions=2):    
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1=population[j,:], solution_2=population[i,:], number_of_functions=number_of_functions):
                    raw_fitness[i,0] = raw_fitness[i,0] + strength[j,0]
    return raw_fitness

# Function: Distance Calculations
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Density (D) - SPEA2 tutorial
def density_calculation(population, number_of_functions=2, k=1):
    density = np.zeros((population.shape[0], 1))
    distance = np.zeros((population.shape[0], population.shape[0]))
    
    # Calcula matriz de distâncias
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                x = np.copy(population[i, population.shape[1]-number_of_functions:])
                y = np.copy(population[j, population.shape[1]-number_of_functions:])
                distance[i,j] = euclidean_distance(x=x, y=y)
    
    # Calcula densidade baseada no k-ésimo vizinho
    for i in range(0, population.shape[0]):
        distances_to_i = distance[i,:]
        distances_to_i = distances_to_i[distances_to_i > 0]  # Remove distância zero (próprio ponto)
        distances_to_i = np.sort(distances_to_i)
        
        if len(distances_to_i) >= k:
            k_distance = distances_to_i[k-1]
        else:
            k_distance = distances_to_i[-1] if len(distances_to_i) > 0 else 1.0
            
        density[i,0] = 1.0 / (k_distance + 2.0)
    
    return density

# Function: Fitness SPEA2 - F(i) = R(i) + D(i)
def fitness_calculation_spea2(population, number_of_functions=2):
    # Passo 1: Calcular valores de força
    strength = strength_values_function(population, number_of_functions)
    
    # Passo 2: Calcular raw fitness
    raw_fitness = raw_fitness_function(population, strength, number_of_functions)
    
    # Passo 3: Calcular densidade
    density = density_calculation(population, number_of_functions, k=1)
    
    # Passo 4: Fitness final
    fitness = raw_fitness + density
    
    return fitness, raw_fitness

# Function: Environmental Selection - SPEA2
def environmental_selection_spea2(population, archive_size, number_of_functions=2):
    # Calcula fitness
    fitness, raw_fitness = fitness_calculation_spea2(population, number_of_functions)
    
    # Separa não-dominadas (raw_fitness = 0) das dominadas
    non_dominated_idx = np.where(raw_fitness.flatten() == 0)[0]
    dominated_idx = np.where(raw_fitness.flatten() > 0)[0]
    
    new_archive = np.zeros((archive_size, population.shape[1]))
    
    if len(non_dominated_idx) <= archive_size:
        # Caso 1: Não-dominadas cabem no arquivo
        archive_count = 0
        
        # Adiciona todas as não-dominadas
        for idx in non_dominated_idx:
            if archive_count < archive_size:
                new_archive[archive_count,:] = population[idx,:]
                archive_count += 1
        
        # Preenche com melhores dominadas se necessário
        if archive_count < archive_size and len(dominated_idx) > 0:
            dominated_fitness = fitness[dominated_idx]
            sorted_dominated_idx = dominated_idx[np.argsort(dominated_fitness.flatten())]
            
            remaining = archive_size - archive_count
            for i in range(min(remaining, len(sorted_dominated_idx))):
                new_archive[archive_count,:] = population[sorted_dominated_idx[i],:]
                archive_count += 1
    else:
        # Caso 2: Muitas não-dominadas - truncamento por distância
        non_dominated_pop = population[non_dominated_idx,:]
        truncated = truncate_by_distance(non_dominated_pop, archive_size, number_of_functions)
        new_archive = truncated
    
    return new_archive

# Function: Truncate by Distance - SPEA2
def truncate_by_distance(population, target_size, number_of_functions=2):
    current_pop = np.copy(population)
    
    while current_pop.shape[0] > target_size:
        # Calcula distâncias mínimas
        min_distances = []
        
        for i in range(current_pop.shape[0]):
            distances = []
            for j in range(current_pop.shape[0]):
                if i != j:
                    x = current_pop[i, current_pop.shape[1]-number_of_functions:]
                    y = current_pop[j, current_pop.shape[1]-number_of_functions:]
                    dist = euclidean_distance(x, y)
                    distances.append(dist)
            
            min_dist = min(distances) if distances else float('inf')
            min_distances.append(min_dist)
        
        # Remove o indivíduo com menor distância
        idx_to_remove = np.argmin(min_distances)
        current_pop = np.delete(current_pop, idx_to_remove, axis=0)
    
    return current_pop

# Function: Sort Population by Fitness
def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness[:,-1])
    fitness_new = np.zeros((population.shape[0], 1))
    population_new = np.zeros((population.shape[0], population.shape[1]))
    for i in range(0, population.shape[0]):
        fitness_new[i,0] = fitness[idx[i],0] 
        for k in range(0, population.shape[1]):
            population_new[i,k] = population[idx[i],k]
    return population_new, fitness_new

# Function: Selection
def roulette_wheel(fitness_new): 
    fitness = np.zeros((fitness_new.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ fitness_new[i,0] + abs(fitness_new[:,0].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    ix = 0
    random_val = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random_val <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values=[-5,-5], max_values=[5,5], mu=1, list_of_functions=[func_1, func_2]):
    offspring = np.copy(population)
    b_offspring = 0
    for i in range (0, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring[i+1,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j]) 
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate=0.1, eta=1, min_values=[-5,-5], max_values=[5,5], list_of_functions=[func_1, func_2]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - len(list_of_functions)):
            probability = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])                        
        for k in range (1, len(list_of_functions) + 1):
            offspring[i,-k] = list_of_functions[-k](offspring[i,0:offspring.shape[1]-len(list_of_functions)])
    return offspring 

# SPEA-2 Function - IMPLEMENTAÇÃO tutorial
def strength_pareto_evolutionary_algorithm_2(population_size=5, archive_size=5, mutation_rate=0.1, min_values=[-5,-5], max_values=[5,5], list_of_functions=[func_1, func_2], generations=5, mu=1, eta=1):        
    count = 0   
    population = initial_population(population_size=population_size, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions) 
    archive = np.zeros((archive_size, population.shape[1]))  # Arquivo inicial vazio
    
    while (count <= generations):       
        print("Generation = ", count)
        
        # Combina população e arquivo
        if count == 0:
            combined = population
        else:
            combined = np.vstack([population, archive])
        
        # Seleção ambiental SPEA2
        archive = environmental_selection_spea2(combined, archive_size, len(list_of_functions))
        
        # Calcula fitness para seleção
        fitness, _ = fitness_calculation_spea2(archive, len(list_of_functions))
        
        # Gera nova população
        population = breeding(archive, fitness, mu=mu, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions)
        population = mutation(population, mutation_rate=mutation_rate, eta=eta, min_values=min_values, max_values=max_values, list_of_functions=list_of_functions)             
        
        count = count + 1              
    return archive

######################## Part 1 - Usage ####################################

# Schaffer Function 1
def schaffer_f1(variables_values=[0]):
    y = variables_values[0]**2
    return y

# Schaffer Function 2
def schaffer_f2(variables_values=[0]):
    y = (variables_values[0]-2)**2
    return y

# Calling SPEA-2 Function
print("=== Executando SPEA2 - Função Schaffer ===")
spea_2_schaffer = strength_pareto_evolutionary_algorithm_2(population_size=50, archive_size=50, mutation_rate=0.1, min_values=[-1000], max_values=[1000], list_of_functions=[schaffer_f1, schaffer_f2], generations=5, mu=1, eta=1)

# Shaffer Pareto Front
schaffer = np.zeros((200, 3))
x = np.arange(0.0, 2.0, 0.01)
for i in range (0, schaffer.shape[0]):
    schaffer[i,0] = x[i]
    schaffer[i,1] = schaffer_f1(variables_values=[schaffer[i,0]])
    schaffer[i,2] = schaffer_f2(variables_values=[schaffer[i,0]])

schaffer_1 = schaffer[:,1]
schaffer_2 = schaffer[:,2]

# Graph Pareto Front Solutions
func_1_values = spea_2_schaffer[:,-2]
func_2_values = spea_2_schaffer[:,-1]
ax1 = plt.figure(figsize=(15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize=12)
plt.ylabel('Function 2', fontsize=12)
ax1.scatter(func_1_values, func_2_values, c='red',   s=25, marker='o', label='SPEA-2')
ax1.scatter(schaffer_1,    schaffer_2,    c='black', s=2,  marker='s', label='Pareto Front')
plt.legend(loc='upper right')
plt.title('SPEA2 - Função Schaffer')
plt.show()

######################## Part 2 - Usage ####################################

# Kursawe Function 1
def kursawe_f1(variables_values=[0, 0]):
    f1 = 0
    if (len(variables_values) == 1):
        f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[0]**2 + variables_values[0]**2))
    else:
        for i in range(0, len(variables_values)-1):
            f1 = f1 - 10 * math.exp(-0.2 * math.sqrt(variables_values[i]**2 + variables_values[i + 1]**2))
    return f1

# Kursawe Function 2
def kursawe_f2(variables_values=[0, 0]):
    f2 = 0
    for i in range(0, len(variables_values)):
        f2 = f2 + abs(variables_values[i])**0.8 + 5 * math.sin(variables_values[i]**3)
    return f2

# Calling SPEA-2 Function
print("\n=== Executando SPEA2 - Função Kursawe ===")
spea_2_kursawe = strength_pareto_evolutionary_algorithm_2(population_size=50, archive_size=50, mutation_rate=0.1, min_values=[-5,-5], max_values=[5,5], list_of_functions=[kursawe_f1, kursawe_f2], generations=5, mu=1, eta=1)

# Graph Pareto Front Solutions
func_1_values = spea_2_kursawe[:,-2]
func_2_values = spea_2_kursawe[:,-1]
ax1 = plt.figure(figsize=(15,15)).add_subplot(111)
plt.xlabel('Function 1', fontsize=12)
plt.ylabel('Function 2', fontsize=12)
ax1.scatter(func_1_values, func_2_values, c='red', s=25, marker='o', label='SPEA-2')
plt.legend(loc='upper right')
plt.title('SPEA2 - Função Kursawe')
plt.show()

print("\n=== SPEA2 CONCLUÍDO ===")