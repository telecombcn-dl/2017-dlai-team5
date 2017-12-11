# DLAI 2017 - Project Work Group 5 :
# Playing StarCraft II with reinforcement learning agents 
This is the project repository for the group 5 at the DLAI. The Team is made up by:

| <img src="images/Luis.jpg" width="200"   alt="" /> | <img src="images/Luis.jpg" width="200"  alt="" /> | <img src="images/Luis.jpg" width="200"  alt="" /> | <img src="images/Alejandro.png" width="200"  alt="" /> |
| :---: | :---: | :---: | :---: |
| Luis Esteve Elfau | Godefroy Goffe | Carlos Roig Marí | Alejandro Suárez Hernández |

This project was developed during the [Deep Learning for Artificial Intelligence](https://telecombcn-dl.github.io/2017-dlai/) Course at UPC TelecomBCN, Autumn 2017.

<img src="images/UPC_ETSETB.jpg" width="600"  alt="" />

## StarCraft II
<p align="center">
<img src="images/img-sc2-logo--large.png"  alt="" />
</p>

<p style="text-align: justify">
As defined on the Blizzard website (the company that develops the game) 'StarCraft II: Wings of Liberty is the long-awaited sequel to the original StarCraft, Blizzard Entertainment’s critically acclaimed sci-fi real-time strategy (RTS) game. StarCraft II: Wings of Liberty is both a challenging single-player game and a fast-paced multiplayer game.
In typical real-time strategy games, players build armies and vie for control of the battlefield. The armies in play can be as small as a single squad of Marines or as large as a full-blown planetary invasion force. As commander, you observe the battlefield from a top-down perspective and issue orders to your units in real time. Strategic thinking is key to success; you need to gather information about your opponents, anticipate their moves, outflank their attacks, and formulate a winning strategy.'
</p>

<p style="text-align: justify">
It combines fast paced micro-actions with the need for high-level planning and execution. Over the previous two decades, StarCraft I and II have been pioneering and enduring e-sports, 2 with millions of casual and highly competitive professional players. Defeating top human players therefore becomes a meaningful and measurable long-term objective.
</p>

<p  style="text-align: justify">
From a reinforcement learning perspective, StarCraft II also offers an unparalleled opportunity to explore many challenging new frontiers:
</p>

<ol>
  <li>It is a multi-agent problem in which several players compete for influence and resources. It is also multi-agent at a lower-level: each player controls hundreds of units, which need to collaborate to achieve a common goal.</li>
  <li>It is an imperfect information game. The map is only partially observed via a local camera, which must be actively moved in order for the player to integrate.</li>
</ol>

### PySC2 Environment

| <img src="images/logo_sc2.png" width="200"   alt="" /> | <img src="images/logo_deepmind.png" width="200"  alt="" /> | <img src="images/logo_python.png" width="200"  alt="" /> |
| :---: | :---: | :---: |
| StarCarft II | Google DeepMind | Python |

<p style="text-align: justify">
PySC2 is DeepMind's Python component of the StarCraft II Learning Environment (<strong>SC2LE</strong>). It exposes Blizzard Entertainment's StarCraft II Machine Learning API as a Python reinforcement learning (<strong>RL</strong>) Environment. 
This is a collaboration between DeepMind and Blizzard to develop StarCraft II into a rich environment for RL research. PySC2 provides an interface for RL agents to interact with StarCraft 2, getting observations and rewards and sending actions.
</p>

<p style="text-align: justify">
Scheme 1 explains how SC2LE works combining StarCarft II API with Google DeepMind Libraries:
</p>

<p align="center">
  <img src="images/sc2le.png"  alt="" />
  <br/>
  <br/>
  Sheme 1. SC2LE
</p>

### Objectives

<p style="text-align: justify">
Playing the whole game is quite an ambitious goal that currently is whithin the reach of scripted agents. However, the StarCraft II learning environment provides several bounded challenges that are most appropriate to test the learning capabilities of an intelligent agent. It is our intention to develop an intelligent Deep RL agent that can perform successfully on several mini-games with bound objectives. Moreover, we want to experiment with the reward system to see how several changes may influence the behaviour of the agent. That's why we can define our objectives by:
</p>

<ul>
  <li>Focusing on small mini-games.</li>
  <li>Training and evaluating several RL agents.</li>
  <li>Dealing and try to improve the reward system.</li>
</ul>


## Techniques

### Learning Curves

### A3C

## Proceedings

## Results and Conclusion

## References
