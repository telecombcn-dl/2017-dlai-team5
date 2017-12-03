import numpy as np

from skimage import transform
from skimage import measure

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import colors, features

from math import sqrt

from absl import app
from PIL import Image

_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_HOSTILE = 4
_VISIBLE = 2
INF = float('inf')

def colorize(imin):
    palette = colors.PLAYER_RELATIVE_PALETTE
    imout = np.zeros(imin.shape+(3,), dtype=np.uint8)
    for idx, color in enumerate(palette):
        imout[imin==idx] = color
    return imout

def count_units(obs, minimap=False):
    obs = obs.observation
    imin = obs['minimap'] if minimap else obs['screen']
    imin = imin[_PLAYER_RELATIVE]
    _, number_of_units = measure.label(imin, connectivity=1, return_num=True)
    return number_of_units

def proportion_visible_onscreen(obs):
    obs = obs.observation
    imin = obs['screen'][_VISIBILITY]
    visible_pix = np.count_nonzero(imin == _VISIBLE)
    return visible_pix / imin.size


def count_enemies(imin):
    _, number_of_enemies = measure.label(imin == _PLAYER_HOSTILE, connectivity=1,
            return_num=True)
    return number_of_enemies

def min_distance_to_enemy(imin):
    player_x, player_y = (imin == _PLAYER_FRIENDLY).nonzero()
    enemy_x, enemy_y = (imin == _PLAYER_HOSTILE).nonzero()
    min_sqdist = INF
    for x, y in zip(enemy_x, enemy_y):
        for x_, y_ in zip(player_x, player_y):
            dx = x - x_
            dy = y - y_
            sqdist = dx*dx + dy*dy
            if sqdist < min_sqdist: min_sqdist = sqdist
    return sqrt(min_sqdist)

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
        screen = timestep.observation['screen'][_PLAYER_RELATIVE]
        player_relative = minimap[_PLAYER_RELATIVE]
        player_relative_color = colorize(transform.resize(player_relative,
            (256, 256), order=0, preserve_range=True))
        number_of_enemies = count_enemies(player_relative)
        min_distance = min_distance_to_enemy(screen)
        units = count_units(timestep)
        print("Number of units in screen: " + str(units))
        print("Number of enemies in minimap: " + str(number_of_enemies))
        print("Minimum distance to enemy: " + str(min_distance))
        print("Proportion visible: {}%".format(100*proportion_visible_onscreen(timestep)))
        pic = Image.fromarray(player_relative_color)
        pic.show()
        input("Press any key to continue")

if __name__ == '__main__':
    app.run(main)

