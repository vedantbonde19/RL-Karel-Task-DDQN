from gym.envs.registration import register

register(
    id="gym_examples/GridWorldMarker-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
    kwargs={"jsondata" : "C:/Users/vedan/Downloads/RL_Project/0_task.json"},
)