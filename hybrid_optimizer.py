#!/usr/bin/env python3


import os
import torch
import torch.multiprocessing as mp
import numpy as np
import math

class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, num_hypercubes_per_edge=100, samples_per_hypercube=5000, start_nth_degree_of_freedom=12, nth_degree_of_freedom_step=3, smallest_degree_of_freedom=0, total_num_tests=1, batch_size=10000):
        defaults = dict(lr=lr)
        super(HybridOptimizer, self).__init__(params, defaults)
        self.num_hypercubes_per_edge = num_hypercubes_per_edge
        self.samples_per_hypercube = samples_per_hypercube
        self.start_nth_degree_of_freedom = start_nth_degree_of_freedom
        self.nth_degree_of_freedom_step = nth_degree_of_freedom_step
        self.smallest_degree_of_freedom = smallest_degree_of_freedom
        self.total_num_tests = total_num_tests
        self.batch_size = batch_size
        mp.set_start_method('spawn', force=True)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data.cpu().numpy()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['sampled_hypercubes'] = set()

                dim = grad.size
                space_min = np.zeros(dim, dtype=np.float32)
                space_max = np.ones(dim, dtype=np.float32)
                total_batches = (self.samples_per_hypercube + self.batch_size - 1) // self.batch_size
                best_sample_gd = None

                # Gradient Descent (GD) sampling method
                for degrees_of_freedom in range(self.start_nth_degree_of_freedom, self.smallest_degree_of_freedom - 1, -self.nth_degree_of_freedom_step):
                    for batch in range(total_batches):
                        current_batch_size = min(self.batch_size, self.samples_per_hypercube - batch * self.batch_size)
                        if degrees_of_freedom == self.start_nth_degree_of_freedom or degrees_of_freedom == self.smallest_degree_of_freedom:
                            current_batch_size = int(current_batch_size * 1.3)  # Increase sample size by 30%
                        sampled_points_gd = self.parallel_sampling(dim, total_batches, current_batch_size, degrees_of_freedom)
                        sampled_points_gd = sampled_points_gd.reshape(grad.shape)
                        current_best_sample_gd = min(sampled_points_gd, key=lambda x: np.sum(grad * x))
                        if best_sample_gd is None or np.sum(grad * current_best_sample_gd) < np.sum(grad * best_sample_gd):
                            best_sample_gd = current_best_sample_gd

                    if best_sample_g
