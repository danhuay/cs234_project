import retro
import src.actions as actions
from src.wrapper import JoypadSpace


def main():
    env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")
    env.reset()

    # Wrap the environment to use our discrete, simple action space
    env = JoypadSpace(env, actions.SIMPLE_MOVEMENT)

    for _ in range(1000000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
