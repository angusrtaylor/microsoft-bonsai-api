import json
import os
import random
import time

import requests

from tqdm import tqdm

import argparse

import numpy as np
import pandas as pd

from sim import extrusion_model as em
from sim import units


# time step (seconds) between state updates
Δt = 1


class ExtruderSimulation():
    def reset(
        self,
        ω0_s: float = 1e-6,
        Δω0_s: float = 0,
        f0_c: float = 1e-6,
        Δf0_c: float = 0,
        T: float = units.celsius_to_kelvin(190),
        L0: float = 1 * 12 * units.METERS_PER_INCH,
        ε: float = 0.1 * units.METERS_PER_INCH,
    ):
        """
        Extruder model for simulation.

        Parameters
        ----------
        ω0_s : float, optional
            Initial screw angular speed (radians / second).
        Δω0_s : float, optional
            Initial change in screw angular speed (radians / second^2).
        f0_c : float, optional
            Initial cutter frequency (hertz).
        Δf0_c : float, optional
            Initial change in cutter frequency (1 / second^2).
        T : float, optional
            Initial temperature (Kelvin).
        L0 : float, optional
            Target product length (meters).
        ε : float, optional
            Product tolerance (meters).
        """

        # angular speed of the extruder screw (radians / second)
        self.ω_s = ω0_s

        # change in angular speed of the extruder screw (radians / second^2)
        self.Δω_s = Δω0_s
        self.Δω_eff = self.Δω_s

        # frequency of the cutter (hertz)
        self.f_c = f0_c

        # change in cutter frequency (1 / second^2)
        self.Δf_c = Δf0_c
        self.Δf_eff = self.Δf_c

        # temperature (Kelvin)
        self.T = T

        self.L0 = L0
        self.ε = ε

        model = em.ExtrusionModel(
            ω=self.ω_s, Δω=self.Δω_s, f_c=self.f_c, T=self.T, Δt=Δt
        )

        self.T += model.ΔT

        # material flow rate (meters^3 / second)
        self.Q = model.Q_op

        # product length (meters)
        self.L = model.L

        # manufacturing yield, defined as the number of good parts
        # per iteration (dimensionless)
        self.yield_ = model.yield_

    def episode_start(self, config) -> None:
        # config == dict
        self.reset(
            ω0_s=config["initial_screw_angular_speed"],
            Δω0_s=config["initial_screw_angular_acceleration"],
            f0_c=config["initial_cutter_frequency"],
            Δf0_c=config["initial_cutter_acceleration"],
            T=config["initial_temperature"],
            L0=config["target_length"]
        )

    def step(self):

        # add a small amount of random noise to the actions to avoid
        # the trivial solution of simply applying zero acceleration
        # on each iteration
        σ_max = 0.0001
        σ_s = random.uniform(-σ_max, σ_max)
        σ_c = random.uniform(-σ_max, σ_max)

        self.Δω_eff = self.Δω_s * (1 + σ_s)
        self.ω_s += Δt * self.Δω_eff

        self.Δf_eff = self.Δf_c * (1 + σ_c)
        self.f_c += Δt * self.Δf_eff

        model = em.ExtrusionModel(
            ω=self.ω_s, Δω=self.Δω_eff, f_c=self.f_c, T=self.T, Δt=Δt, L0=self.L0
        )

        self.T += model.ΔT

        # material flow rate (meters^3 / second)
        self.Q = model.Q_op

        # product length (meters)
        self.L = model.L

        # manufacturing yield, defined as the number of good parts
        # per iteration (dimensionless)
        self.yield_ = model.yield_

    def episode_step(self, action) -> None:
        self.Δω_s = action["screw_angular_acceleration"]
        self.Δf_c = action["cutter_acceleration"]

        self.step()

    def get_state(self):
        return {
            "screw_angular_speed": self.ω_s,
            "screw_angular_acceleration": self.Δω_eff,
            "cutter_frequency": self.f_c,
            "cutter_acceleration": self.Δf_eff,
            "temperature": self.T,
            "product_length": self.L,
            "flow_rate": self.Q,
            "yield": self.yield_,
            "target_length": self.L0
        }


class Brain():

    def __init__(self):
        self.url = "http://localhost:5000"
        self.predictionPath = "/v1/prediction"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.endpoint = self.url + self.predictionPath
    
    def get_actions(self, state):
        response = requests.post(
                self.endpoint,
                data = json.dumps(state),
                headers = self.headers
            )
        return response.json()


def output_episode(episode, episode_id, states, actions, output_file):
    states_df = pd.DataFrame.from_records(states)
    states_df.columns = ["state." + x for x in states_df.columns]
    actions_df = pd.DataFrame.from_records(actions)
    actions_df.columns = ["action." + x for x in actions_df.columns]
    episode_df = pd.concat([states_df, actions_df], axis=1)
    for k, v in episode.items():
        episode_df[k] = v
    episode_df["episode_id"] = episode_id
    episode_df["iteration_id"] = [x for x in range(episode_df.shape[0])]
    episode_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))


def remove_complex_numbers(state):
    for k, v in state.items():
        if isinstance(v, complex):
            state[k] = v.real
    return state


def run_episode(extruder_sim, brain, episode, episode_id, output_file, iterations):
    request_times = []
    states, actions = [], []
    extruder_sim.episode_start(episode)
    for i in range(iterations):
        state = extruder_sim.get_state()
        state = remove_complex_numbers(state)
        request_start = time.perf_counter()
        action = brain.get_actions(state)
        request_end = time.perf_counter()
        elapsed_time = request_end - request_start
        request_times.append(elapsed_time)
        extruder_sim.episode_step(action)
        states.append(state)
        actions.append(action)
    output_episode(episode, episode_id, states, actions, output_file)
    median_time = np.median(request_times)
    return median_time



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("episode_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("iterations", type=int)
    args = parser.parse_args()

    episode_file = args.episode_file
    output_file = args.output_file
    iterations = args.iterations

    # episode_file = "./eval/eval_continuous.json"
    # output_file = "./eval/eval_continuous_output.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    extruder_sim = ExtruderSimulation()
    brain = Brain()

    with open(episode_file, "r") as infile:
        eval_episodes = json.load(infile)["episodeConfigurations"]

    median_times = []
    for episode_id, episode in enumerate(tqdm(eval_episodes)):
        median_time = run_episode(extruder_sim, brain, episode, episode_id, output_file, iterations)
        median_times.append(median_time)
    
    mean_time = np.mean(median_times)
    iterations_per_second = 1 / mean_time
    print("Average scoring time {} or {} iterations per second".format(mean_time, iterations_per_second))

if __name__ == "__main__":
    main()
