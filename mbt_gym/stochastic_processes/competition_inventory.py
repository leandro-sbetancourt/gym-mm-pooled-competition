import abc
from typing import Optional
from math import sqrt
import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel


class CompetitionInventoryModel(StochasticProcessModel):
    """CompetitionInventoryProcess models the inventory of the competition."""

    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None):
        pass

    @abc.abstractmethod
    def get_competition_depth(self) -> np.ndarray:
        pass


class BhsbInventoryModel(CompetitionInventoryModel):
    def __init__(
        self,
        min_value_inventory: float = -100,
        max_value_inventory: float = 100,
        alpha: float = 0.001,
        beta: float = 0.001,
        step_size: float = 0.1,
        sigma: float = 1,
        num_trajectories: int = 1,
    ):
        self.min_value_inventory = min_value_inventory
        self.max_value_inventory = max_value_inventory
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        super().__init__(
            min_value=np.array([[min_value_inventory, min_value_inventory]]),
            max_value=np.array([[max_value_inventory, min_value_inventory]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[0, 0]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        ones = np.ones((self.num_trajectories, 1))
        fill_multiplier = np.append(-ones, ones, axis=1)
        self.current_state[:,0] = (self.current_state[:,0] 
                                + np.reshape(np.sum(arrivals * (1. - fills) * -fill_multiplier, axis=1), (self.num_trajectories,)) )
        self.current_state[:,1] = (self.current_state[:,1]
                                + self.sigma *sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, )))
        
    def get_competition_depth(self):
        #print(self.current_state.shape)
        comp_ask_depths = self.alpha - self.beta * self.current_state[:,0].reshape(-1,1) - self.current_state[:,1].reshape(-1,1)
        comp_bid_depths = self.alpha + self.beta * self.current_state[:,0].reshape(-1,1) + self.current_state[:,1].reshape(-1,1)
        return np.append(comp_bid_depths, comp_ask_depths, axis=1)