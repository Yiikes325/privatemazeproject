import numpy as np
from PIL import Image
import random
import sys

class MazeGeneration:
    def __init__(self, width, height, scale = 3):
        self.width = width
        self.height = height
        self.scale = scale
        self.grid = np.zeros((width, height), dtype = bool)

    def cellWalls(self, x, y):
        #Creation of the cell walls, checks to see where to start based on height and width.
        walls = set()
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            if x > 1 and not self.grid[x - 2][y]:
                walls.add((x - 2, y))
            if x + 2 < self.width and not self.grid[x + 2][y]:
                walls.add((x + 2, y))
            if y > 1 and not self.grid[x][y - 2]:
                walls.add((x, y - 2))
            if y + 2 < self.height and not self.grid[x][y + 2]:
                walls.add((x, y + 2))
        return walls

    def mazeNeighbours(self, x, y):
        #Creation of neighbouring cells.
        neighbours = set()
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            if x > 1 and self.grid[x - 2][y]:
                neighbours.add((x - 2, y))
            if x + 2 < self.width and self.grid[x + 2][y]:
                neighbours.add((x + 2, y))
            if y > 1 and self.grid[x][y - 2]:
                neighbours.add((x, y - 2))
            if y + 2 < self.height and self.grid[x][y + 2]:
                neighbours.add((x, y + 2))
        return neighbours

    def wallConnection(self, x1, y1, x2, y2):
        #Connects the outer walls together.
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        self.grid[x1][y1] = True
        self.grid[x][y] = True

    def generate(self, labeller_function): #Labeller function was an attempt to connect the GCN to the generator.
        generation = set()
        x, y = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.grid[x][y] = True
        wallSet = self.cellWalls(x, y)
        for walls in wallSet:
            generation.add(walls)
        while generation:
            x, y = random.choice(tuple(generation))
            generation.remove((x, y))
            neighbourSet = self.mazeNeighbours(x, y)
            if neighbourSet:
                neighbourX, neighbourY = random.choice(tuple(neighbourSet))
                self.wallConnection(x, y, neighbourX, neighbourY)
            wallSet = self.cellWalls(x, y)
            for walls in wallSet:
                generation.add(walls)

    def getImage(self, passageColour = (255, 255, 255), wallColour = (0, 0, 0)):
        #Uses OpenCV to get the image based on how the maze was generated.
        im = Image.new('RGB', (self.width, self.height))
        pixels = im.load()
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y]:
                    pixels[x, y] = passageColour
                else:
                    pixels[x, y] = wallColour
        im.save("Maze5x5.png", "PNG")
        return im


def main():
    #Alternatives if the command hasn't been input properly when testing from this program script.
    try:
        args = sys.argv
        if len(sys.argv) == 3:
            MazeGeneration(int(args[1]), int(args[2])) #If 2 arguments have been put in, follow the instructions of 2 arguments.
        elif len(sys.argv) == 2:
            MazeGeneration(int(args[1]), int(args[1])) #If 1 argument has been put in, duplicate that 1 argument.
        elif len(sys.argv) == 1:
            MazeGeneration(int(30), int(30)) #If no argument has been put in, generate a maze of size 30x30
        else:
            raise ValueError #If the arguments are all wrong, throw a value error.

    except ValueError:
        print("If running the program directly through the maze generation script, please run the program by inputting any of the following: ")
        print(">python miniMazeTest.py width: (insert integer, e.g. 5) height: (insert integer, e.g. 5)")
        print(">python miniMazeTest.py width: (insert integer, e.g. 5)")
        print(">python miniMazeTest.py")

if __name__ == "__main__":
    main()
