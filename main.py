# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:20:44 2025

@author: 3mrullah
"""

import numpy as np
from os import listdir
from os.path import isfile, join
from copy import deepcopy
import uuid
import random
import os
import time


def read_instance(folder_name, file_no):
    # Get list of files in the folder
    filenames = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    file_path = f"{folder_name}/{filenames[file_no]}"
    print(f"Reading file: {file_path}")

    with open(file_path, "r") as f:
        # Skip first two lines
        f.readline()
        f.readline()

        # Read m, n, C from third line
        line1 = f.readline()
        start = line1.index('=')
        stop = line1.index(' ', start)
        m = int(line1[start + 1:stop])
        start = line1.index('=', stop)
        stop = line1.index(' ', start)
        n = int(line1[start + 1:stop])
        start = line1.index('=', stop)
        iseof = line1.find(' ', start)
        stop = len(line1) - 1 if iseof == -1 else line1.index(' ', start)
        C = int(line1[start + 1:stop])

        # Skip next two lines
        f.readline()
        f.readline()

        # Read profits (p)
        p = np.array(list(map(int, f.readline().split())))

        # Skip next two lines
        f.readline()
        f.readline()

        # Read weights (w)
        w = np.array(list(map(int, f.readline().split())))

        # Skip next two lines
        f.readline()
        f.readline()

        # Read rmatrix and build items
        rmatrix = np.zeros((m, n), dtype=bool)
        items = []
        for i in range(m):
            rm = list(map(int, f.readline().split()))
            rmatrix[i, :] = rm[:]
            items.append(np.where(rmatrix[i, :] == True)[0])

    # Compute frequencies and R values
    freqs = np.sum(rmatrix, axis=0)
    R = np.zeros(m)
    for i in range(m):
        weights = [w[j] / freqs[j] for j in range(n) if rmatrix[i][j]]
        R[i] = p[i] / np.sum(weights) if weights else 0  # Handle empty case

    # Sort indices by R in descending order
    H = np.argsort(R)[::-1][:m]

    return m, n, C, p, w, rmatrix, items, H


def repair_solution(solution, m, n, C, w, items, H):
    temp_sol = np.zeros(m, dtype=bool)
    temp = np.zeros(n, dtype=bool)
    trial = np.array(temp, copy=True)

    for i in range(m):
        a = H[i]
        if solution[a]:
            b = items[a]
            trial[b] = True
            cap_val = np.sum(w, where=trial)
            if cap_val <= C:
                temp[b] = True
                temp_sol[a] = True
            else:
                trial = temp.copy()

    return temp_sol, temp


def optimizing_stage(solution, temp, m, n, C, w, items, H):
    trial = temp.copy()
    for i in range(m):
        a = H[i]
        if solution[a] == 0:
            b = items[a]
            trial[b] = 1
            cap_val = np.sum(w, where=trial)
            if cap_val <= C:
                temp[b] = True
                solution[a] = True
            else:
                trial = temp.copy()

    return solution


def objective_function(solution, m, n, C, p, w, items, H):
    temp = np.zeros(n, dtype=bool)
    for i in range(m):
        if solution[i]:
            temp[items[i]] = True
    cap_val = np.sum(w, where=temp)

    if cap_val > C:
        solution, temp = repair_solution(solution, m, n, C, w, items, H)
        solution = optimizing_stage(solution, temp, m, n, C, w, items, H)
    else:
        solution = optimizing_stage(solution, temp, m, n, C, w, items, H)

    # sum_val = np.sum(p[i] for i, val in enumerate(solution) if val)
    # return solution, sum_val

    sum_val = np.sum([p[i] for i, val in enumerate(solution) if val])
    return solution, sum_val


def generate_neighbor(solution, m):
    neighbor = solution.copy()
    idx = np.random.randint(0, m)
    neighbor[idx] = not neighbor[idx]
    return neighbor


# def hill_climbing(m, n, C, p, w, items, H, max_iterations=100):
#     # Initialize random solution
#     current_solution = np.random.choice([True, False], size=m)
#     current_solution, current_value = objective_function(current_solution, m, n, C, p, w, items, H)
#
#     for _ in range(max_iterations):
#         # Generate neighbor
#         neighbor = generate_neighbor(current_solution, m)
#         neighbor_solution, neighbor_value = objective_function(neighbor, m, n, C, p, w, items, H)
#
#         # If neighbor is better, update current solution
#         if neighbor_value >= current_value:
#             current_solution = neighbor_solution
#             current_value = neighbor_value
#
#     return current_solution, current_value


# def main():
#     # Example usage: adjust folder_name and file_no as needed
#     folder_name = "sukp_instances"  # Update with actual folder path
#     file_no = 0  # Index of the file to read
#
#     # Read problem instance
#     m, n, C, p, w, rmatrix, items, H = read_instance(folder_name, file_no)
#
#     # Run hill climbing
#     solution, value = hill_climbing(m, n, C, p, w, items, H)
#
#     print(f"Best solution value: {value}")
#     print(f"Solution: {solution}")
#
#
# if __name__ == "__main__":
#     main()
def memetic_algorithm(
    m, n, C, p, w, items, H,
    pop_size=100,
    generations=500,
    crossover_rate=0.9,
    mutation_rate=0.1,
    elitism_rate=0.1,
    local_search_iters=5
):
    def crossover(a, b):
        pt = random.randint(1, m - 1)
        return np.concatenate([a[:pt], b[pt:]])

    def mutate(ind):
        for i in range(m):
            if random.random() < mutation_rate:
                ind[i] = not ind[i]
        return ind

    # Initialize population
    population = [
        objective_function(
            np.random.choice([False, True], size=m),
            m, n, C, p, w, items, H
        )[0]
        for _ in range(pop_size)
    ]

    elite_count = max(1, int(pop_size * elitism_rate))

    for gen in range(generations):
        # Evaluate all individuals
        scored = [
            (ind, objective_function(ind, m, n, C, p, w, items, H)[1])
            for ind in population
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Elitism: carry forward top individuals unchanged
        new_pop = [ind for ind, _ in scored[:elite_count]]

        # Create the rest by crossover + mutation + local search
        while len(new_pop) < pop_size:
            # Parent selection: fitness‐proportionate (roulette‐wheel)
            weights = [score for _, score in scored]
            parents = random.choices(
                [ind for ind, _ in scored],
                weights=weights,
                k=2
            )
            # Crossover
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = parents[0].copy()
            # Mutation
            child = mutate(child)
            # Repair & Optimize
            repaired, covered = repair_solution(child, m, n, C, w, items, H)
            polished = optimizing_stage(repaired, covered, m, n, C, w, items, H)
            # Local Hill‐Climbing
            current = polished
            current_val = objective_function(current, m, n, C, p, w, items, H)[1]
            for _ in range(local_search_iters):
                neighbor = generate_neighbor(current, m)
                _, neigh_val = objective_function(neighbor, m, n, C, p, w, items, H)
                if neigh_val > current_val:
                    current, current_val = neighbor, neigh_val
            new_pop.append(current)

        population = new_pop

    # Return best found
    best = max(population, key=lambda ind: objective_function(ind, m, n, C, p, w, items, H)[1])
    return best, objective_function(best, m, n, C, p, w, items, H)[1]


# -----------------------
# Main
# -----------------------
def main():
    folder_name = "sukp_instances"
    file_no     = 1

    # Load your instance once
    m, n, C, p, w, rmatrix, items, H = read_instance(folder_name, file_no)

    runs = 5
    results = []
    times = []

    total_start = time.time()
    for run in range(1, runs + 1):
        print(f"\n--- Run {run} ---")
        start = time.time()
        mem_sol, mem_val = memetic_algorithm(m, n, C, p, w, items, H)
        elapsed = time.time() - start

        print(f"Memetic Algorithm value: {mem_val} (time: {elapsed:.2f}s)")
        results.append(mem_val)
        times.append(elapsed)

    total_elapsed = time.time() - total_start

    # Sort runs by value descending for summary
    sorted_results = sorted(enumerate(results, 1), key=lambda x: x[1], reverse=True)

    print("\n=== Summary of 5 Runs ===")
    for run_idx, val in sorted_results:
        print(f"Run {run_idx}: value={val} (time: {times[run_idx-1]:.2f}s)")

    best_idx, best_val = sorted_results[0]
    print(f"\nBest over all runs: Run {best_idx} value={best_val} (time: {times[best_idx-1]:.2f}s)")
    print(f"Total time for {runs} runs: {total_elapsed:.2f}s")
    print("Known optimum: 13283")


if __name__ == "__main__":
    main()