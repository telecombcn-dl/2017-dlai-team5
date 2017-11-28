import numpy as np

from skimage import transform
from skimage import measure

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import colors, features

from absl import app
from PIL import Image

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_HOSTILE = 4

def colorize(imin):
    palette = colors.PLAYER_RELATIVE_PALETTE
    imout = np.zeros(imin.shape+(3,), dtype=np.uint8)
    for idx, color in enumerate(palette):
        imout[imin==idx] = color
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
        player_relative = minimap[_PLAYER_RELATIVE]
        player_relative_color = colorize(transform.resize(player_relative,
            (256, 256), order=0, preserve_range=True))
        _, number_of_enemies = measure.label(player_relative == _PLAYER_HOSTILE,
            connectivity=1, return_num=True)
        print("Number of enemies in minimap: " + str(number_of_enemies))
        pic = Image.fromarray(player_relative_color)
        pic.show()
        input("Press any key to continue")

if __name__ == '__main__':
    app.run(main)

