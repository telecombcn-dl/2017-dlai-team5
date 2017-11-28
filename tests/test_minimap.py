import numpy as np

from scipy.ndimage import zoom

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import colors, features

from absl import app
from PIL import Image

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

def colorize(imin):
    palette = colors.PLAYER_RELATIVE_PALETTE
    imout = np.zeros(imin.shape+(3,), dtype=np.uint8)
    for idx, color in enumerate(palette):
        imout[imin==idx] = color
    print(imin.shape)
    print(imin.dtype)
    print(imout.shape)
    print(imout.dtype)
    return imout


def main(argv):

    with sc2_env.SC2Env(
            map_name='FindAndDefeatZerglings',
            step_mul=8,
            game_steps_per_episode=8,
            screen_size_px=(84,84),
            minimap_size_px=(64,64),
            visualize=True) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        timestep = env.reset()[0]
        minimap = timestep.observation['minimap']
        screen = timestep.observation['screen']
        player_relative = zoom(minimap[_PLAYER_RELATIVE], 4, order=0)
        player_relative = colorize(player_relative)
        pic = Image.fromarray(player_relative)
        pic.show()
        input("Press any key to continue")

if __name__ == '__main__':
    app.run(main)

