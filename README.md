To show that your system works, you must provide a README file that precisely
describes the basic use of your system, including both general instructions and
a specific example that walks through an interaction with the system and the
expected output. Your implementation must also include all necessary source and
data files and a Makefile that allows me to build your project with the make
command.



# CSC 213 &ndash; Final Project

Anna Blinderman, David Kraemer, Zachary Segall

# General Instructions 

This guide assumes that the executable file `life` exists at the root of the
repository. If it does not exist, run
```
$ make
```
at the root of the repository, which will produce the executable. To run the
program without a preset board, run
```
$ ./life
```
which generates a blank SDL window. This is the Game of Life simulator,
initialized in pause mode. 

## User input

User input is read from the mouse and from the keyboard. By left-clicking a
cell, or left-clicking and dragging through multiple cells, the user toggles
the cell(s) to the "living" state. By right-clicking a cell, or right-clicking
and dragging through multiple cells, the user toggles the cell(s) to the "dead"
state.

The following keyboard commands provide additional mechanisms for user
interface:

| Command | Description |
|:--------|:------------|
| `CTRL-P` | Toggles between paused and unpaused simulation modes. |
| `CTRL-C` | Clears the game board of all living cells. |
| `CTRL-G` | Populates the region of cells where the mouse is located with a glider<sup>1</sup> cell. |
| `CTRL-Q` | Quits the simulation (and exits the SDL window). |
| `CTRL-SPACE` | Advances one step through the simulation (only valid when in pause mode). |

## Loading preset boards

An instance of the Game of Life simulator can begin with a preset board. In the
`boards/` directory, there exist game board files which specify an initial
configuration of the board. given a game board file, `sample.board`, the usage 
for the program is
```
$ ./life boards/sample.board
```

<sup>1</sup>The glider is a pattern that steadily traverses the board: 
![Glider pattern](https://upload.wikimedia.org/wikipedia/commons/f/f2/Game_of_life_animated_glider.gif)

<sup>2</sup>The glider gun is a pattern that steadily produces gliders:
![Glider gun
pattern](https://upload.wikimedia.org/wikipedia/commons/e/e5/Gospers_glider_gun.gif)

# Specific Example 

In this example we will explore the glider gun<sup>2</sup> pattern. The file
`gun.board` is provided in the `boards` directory. We load the board into the
program using
```
$ ./life boards/gun.board
```
This produces the image:

![Glider gun initialization](images/glider-gun-00.png)

Let's update the board a few times manually. If we press `CTRL-SPACE` 23 times,
we get the first glider:

![Glider gun after 23 steps](images/glider-gun-01.png)

Notice how the colors of the glider gun have changed dramatically. This is
because the colors indicate the advancing age of each living cell. Let's
advance a bit more by pressing `CTRL-SPACE` 6 times. We now have the board:

![Glider gun after 29 steps](images/glider-gun-02.png)

Let's add a bunch of gliders. Remember that you can draw a glider on the grid
by pressing `CTRL-G`:

![Glider gun plus a bunch](images/glider-gun-03.png)

These are all going to evolve the same way, so let's advance by 7 turns using
`CTRL-SPACE`. We end up with the board:

![Glider gun plus a bunch evolution](images/glider-gun-04.png)

We're about to get a new glider from the gun, but at this point, let's just see
how this runs for a while. Pressing `CTRL-P`, we can see the real-time
evolution of our board.

![Glider gun after a while](images/glider-gun-05.png)

Here we see the glider gun in action, having produced a good number of gliders.
Also, there seemed to have been some interactions (read: explosions) between
our gliders on the left. Interesting! Let's see some more.

![Glider gun after a longer while](images/glider-gun-06.png)

The conflagration to the left looks like it might interfere with the glider
gun.  But right now, everything is still intact. You can see that the gliders
are all colliding on the bottom right, which is because the boundary of the
game board is a hard wall.

![Glidermaggedon](images/glider-gun-07.png)

Oh no, our glider gun exploded! Oh well...

![Glider graveyard](images/glider-gun-08.png)

This is the steady state of our game of life. It's kind of sad, but still neat.

# Navigating the repository




