from gymnasium.envs.registration import register

from dmash.envs.dmc.registration import register_dmc_envs


register(
    id="dmash/DI-sparse-v0",
    entry_point="dmash.envs:DIEnv",
    max_episode_steps=100,
    kwargs={"reward_type": "sparse"},
)

register(
    id="dmash/DI-friction-sparse-v0",
    entry_point="dmash.envs:DIEnv",
    max_episode_steps=100,
    kwargs={"friction": 1.0, "reward_type": "sparse"},
)

register(
    id='dmash/ODE-v0',
    entry_point='dmash.envs:ComplexODE',
    max_episode_steps=200,
)
for i in range(1, 21):
    kwargs = {f"parameter_{j}": 1.0 for j in range(i)}
    kwargs["only_two"] = False
    register(
        id=f'dmash/ODE-{i}-v0',
        entry_point='dmash.envs:ComplexODE',
        max_episode_steps=200,
        kwargs=kwargs,
    )

register_dmc_envs()
