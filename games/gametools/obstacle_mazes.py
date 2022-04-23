# from 1104488c00de90587914e85866c865a1bffcfef7

from gym_minigrid.minigrid import MiniGridEnv, Goal, Wall, Grid, Lava
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from gym_minigrid.register import register

def parse_tile_array(tile_array, padding=1, ignore='.'):
    objs = {}
    h, w = len(tile_array), len(tile_array[0])
    v_pad = ['='*w for _ in range(padding)]
    h_pad = '='*padding
    for y, row in enumerate(v_pad+tile_array+v_pad):
        padded_row = h_pad+row+h_pad
        for x, c in enumerate(padded_row):
            if c in ignore:
                continue
            loc = (x, y)
            obj = objs.get(c, set([])) | {loc}
            objs[c] = obj
    return objs

class ObstacleMaze(MiniGridEnv):
    """
    Maze composed of obstacles
    """

    def __init__(
        self,
        tile_array,
        min_obstacles=0,
        max_obstacles=float('inf'),
        random_agent_start=False,
        random_goal=False,
        obstacle_type='wall',
        max_steps=None,
    ):
        self.tile_array = tile_array
        self.objs = parse_tile_array(tile_array)
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.random_agent_start = random_agent_start
        self.random_goal = random_goal
        self.obstacle_type = obstacle_type

        super().__init__(
            height=len(tile_array)+2,
            width=len(tile_array[0])+2,
            max_steps=max_steps or 2*len(tile_array)*len(tile_array[0]),
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Put obstacles
        obstacles = list(set(self.objs.keys()) & set("0123456789"))
        min_obs = self.min_obstacles
        max_obs = min(self.max_obstacles, len(obstacles))
        n_obstacles = self._rand_int(min_obs, max_obs+1)
        obstacles = self._rand_subset(obstacles, n_obstacles)
        for o in obstacles:
            locs = self.objs[o]
            for loc in locs:
                if self.obstacle_type == 'lava':
                    self.put_obj(Lava(), *loc)
                elif self.obstacle_type == 'wall':
                    self.put_obj(Wall(color='purple'), *loc)
                else:
                    raise Exception("Unknown obstacle type")

        # Put outer walls
        for loc in self.objs["="]:
            self.put_obj(Wall(color='grey'), *loc)

        # Put walls
        for loc in self.objs["#"]:
            if self.obstacle_type == 'lava':
                self.put_obj(Lava(), *loc)
            elif self.obstacle_type == 'wall':
                self.put_obj(Wall(color='grey'), *loc)
            else:
                raise Exception("Unknown obstacle type")

        # Put down agent
        if self.random_agent_start:
            self.place_agent(rand_dir=True)
        else:
            agent_locs = list(self.objs['S'])
            self.agent_pos = self._rand_elem(agent_locs)
            self.agent_dir = self._rand_int(0, 4)

        # Place goal
        if self.random_goal:
            goal_pos = self.place_obj(None)
        else:
            goal_pos = self._rand_elem(list(self.objs['G']))
        self.put_obj(Goal(), *goal_pos)

        self.mission = "get to the green goal square"

mazes = {
    "grid-0-0": [
        "...3.0....G",
        ".333.0.....",
        ".....00.444",
        "6....#..4..",
        "6....#.....",
        "6..#####...",
        "6....#.....",
        "..1..#.2...",
        "111..222...",
        "..........5",
        "S.......555"
    ],
    "grid-1-0": [
        "66..4444..G",
        "6..........",
        "6..........",
        ".....#.....",
        "..22.#.333.",
        "..2#####.3.",
        "5.2..#.....",
        "5....#.....",
        "55........1",
        "0000......1",
        "S........11"
    ],
    "grid-2-0": [
        "G.44.333...",
        "..4....3..5",
        "..4.......5",
        ".....#....5",
        "..0..#....5",
        "000#####...",
        ".111.#.....",
        "...1.#..222",
        "........2..",
        "...........",
        "S...6666..."
    ],
    "grid-3-0": [
        "..11..0...G",
        "...1..0....",
        "...1..0..4.",
        ".2...#0444.",
        ".2...#.....",
        ".22#####.33",
        "....6#....3",
        "....6#....3",
        "....66..555",
        "..........5",
        "S.........."
    ],
    "grid-4-0": [
        "G.......666",
        "...44.....6",
        "...4.....55",
        "...4.#...5.",
        "000..#...5.",
        "..0#####...",
        ".....#.2...",
        "...3.#.222.",
        ".333.......",
        "......1....",
        "S.....111.."
    ],
    "grid-5-0": [
        ".6666.0...G",
        "......0.555",
        ".....00.5..",
        ".111.#.....",
        "...1.#.....",
        "2..#####...",
        "2....#....4",
        "22...#.3444",
        ".......3...",
        "......33...",
        "S.........."
    ],
    "grid-6-0": [
        "..222G..555",
        "..2......5.",
        "....66.....",
        "....6#.....",
        "000.6#.....",
        "..0#####.44",
        ".....#....4",
        "...1.#....4",
        ".111.......",
        "..........3",
        "S.......333"
    ],
    "grid-7-0": [
        "444..G..555",
        ".4........5",
        "........111",
        "666..#....1",
        "6....#.....",
        "...#####000",
        ".....#....0",
        ".22..#.....",
        ".2........3",
        ".2........3",
        ".....S...33"
    ],
    "grid-8-0": [
        "55...444000",
        "5....4..0..",
        "5.G........",
        ".....#.....",
        "11...#..222",
        "1..#####2..",
        "1....#....3",
        "666..#....3",
        "6........33",
        "...........",
        "..........S"
    ],
    "grid-9-0": [
        "..........G",
        "..33.......",
        "...3444....",
        "22.3.#4....",
        ".2...#.....",
        ".2.#####555",
        ".....#0...5",
        "66...#0....",
        ".6....00...",
        ".6........1",
        "S.......111"
    ],
    "grid-10-0": [
        "000.......G",
        ".0........6",
        ".....555..6",
        ".....#.5.66",
        "..1..#.....",
        "..1#####222",
        ".11..#....2",
        ".....#.33..",
        "........3..",
        "....4...3..",
        "S.444......"
    ],
    "grid-11-0": [
        "..444.....G",
        "....4.....6",
        "..3.......6",
        "..333#...66",
        ".....#....2",
        ".00#####..2",
        "..0..#...22",
        "..0..#.....",
        ".....1....5",
        ".....1..555",
        "S....11...."
    ]
}

class ObsMaze_00_2Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=mazes["grid-0-0"],
            min_obstacles=2,
            max_obstacles=2,
            random_agent_start=False,
            random_goal=True,
            **kwargs
        )
class ObsMaze_00_5to7Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=mazes["grid-0-0"],
            min_obstacles=5,
            max_obstacles=7,
            random_agent_start=False,
            random_goal=True,
            **kwargs
        )

'''
register(
    id='MiniGrid-ObsMaze-00-2Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMaze_00_2Obs'
)
register(
    id='MiniGrid-ObsMaze-00-5to7Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMaze_00_5to7Obs'
)
'''

simple_mazes = {
    "simple-grid-0-0": [
        "...0..G",
        "33.0.44",
        "6..#...",
        "6.###..",
        "...#...",
        "11.22.5",
        "S.....5",
    ],

    "simple-grid-1-0": [
        "S0..G",
        ".1...",
        "...#.",
        "S..#G",
    ]
}

class ObsMazeSimple_00_2Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=simple_mazes["simple-grid-0-0"],
            min_obstacles=2,
            max_obstacles=2,
            random_agent_start=False,
            random_goal=True,
            **kwargs
        )
class ObsMazeSimple_00_5to7Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=simple_mazes["simple-grid-0-0"],
            min_obstacles=5,
            max_obstacles=7,
            random_agent_start=False,
            random_goal=True,
            **kwargs
        )

'''
register(
    id='MiniGrid-ObsMazeSimple-00-2Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMazeSimple_00_2Obs'
)
register(
    id='MiniGrid-ObsMazeSimple-00-5to7Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMazeSimple_00_5to7Obs'
)
'''

class ObsMazeSimple_01_1Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=simple_mazes["simple-grid-1-0"],
            min_obstacles=1,
            max_obstacles=1,
            random_agent_start=False,
            random_goal=False,
            **kwargs
        )

class ObsMazeSimple_01_2Obs(ObstacleMaze):
    def __init__(self, **kwargs):
        super().__init__(
            tile_array=simple_mazes["simple-grid-1-0"],
            min_obstacles=2,
            max_obstacles=2,
            random_agent_start=False,
            random_goal=False,
            **kwargs
        )
'''
register(
    id='MiniGrid-ObsMazeSimple-01-1Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMazeSimple_01_1Obs'
)
register(
    id='MiniGrid-ObsMazeSimple-01-2Obs-v0',
    entry_point='project_code.obstacle_mazes:ObsMazeSimple_01_2Obs'
)
'''
