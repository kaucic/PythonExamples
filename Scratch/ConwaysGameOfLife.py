from ivisual import *

    
scene.x = 0
scene.y = 0
scene.width = 1000
scene.height = 1000
scene.title = "Conway's Game of Life in VPython on a Toroidal Surface"
scene.ambient = 1
scene.background = (0.3,0.3,0.3)
invisBall = sphere(pos = (-11,-11,-11), opacity = 0)

UNIVERSE_SIZE = 20

generationCounter = 0

universe = [[0 for x in range (UNIVERSE_SIZE)] for x in range (UNIVERSE_SIZE)]
tempUniverse = [[0 for x in range (UNIVERSE_SIZE)] for x in range (UNIVERSE_SIZE)]
neighborCounts = [[0 for x in range(UNIVERSE_SIZE)] for x in range (UNIVERSE_SIZE)]
cell = box(pos = (0,0,0), length = 1, width = 1, height = 1, color = color.green, visible = False)
cells = [[cell for x in range(UNIVERSE_SIZE)] for y in range(UNIVERSE_SIZE)]

def checkNeighbors(x, y):
    neighborCount = 0
    for i in range (-1, 2):
        for j in range (-1, 2):
            if(i == 0 and j == 0):
                continue
            
            xcol = x+i
            if(xcol < 0):
                xcol += universe.__len__()
            if(xcol > universe.__len__() - 1):
                xcol -= universe.__len__()
                
            ycol = y+j
            if(ycol < 0):
                ycol += universe.__len__()
            if(ycol > universe.__len__() - 1):
                ycol -= universe.__len__()
                
            if(universe[xcol][ycol] == 1):
                neighborCount += 1
                
    return neighborCount

def updateUniverse():
    for i in range(universe.__len__()):
        for j in range(universe.__len__()):
            neighborCounts[i][j] = checkNeighbors(i,j)
            tempUniverse[i][j] = universe[i][j]
            
    for i in range(universe.__len__()):
        for j in range(universe.__len__()):
            n = neighborCounts[i][j]
            s = tempUniverse[i][j]
            if(s == 0 and n == 3):
                universe[i][j] = 1
            elif(s == 1 and n < 2):
                universe[i][j] = 0
            elif(s == 1 and n > 3):
                universe[i][j] = 0
    
def drawUniverse():
    for i in range(universe.__len__()):
        for j in range(universe.__len__()):
            if(universe[i][j] == 1):
                cells[i][j].visible = True
            else:
                cells[i][j].visible = False
                
#edit this method to change the initial state of the universe
def setInitialConditions():
    for i in range(universe.__len__()):
        for j in range(universe.__len__()):
            cells[i][j] = box(pos = ((i - universe.__len__() / 2) * 1.1, (j - universe.__len__() / 2) * 1.1, 0), length = 1, width = 1, height = 1, color = color.green)
            
    glider(5,5)
    
def acorn(x, y):
    universe[x+1][y] = 1
    universe[x+3][y+1] = 1
    universe[x][y+2] = 1
    universe[x+1][y+2] = 1
    universe[x+4][y+2] = 1
    universe[x+5][y+2] = 1
    universe[x+6][y+2] = 1

def glider(x, y):
    universe[x][y] = 1
    universe[x+1][y+1] = 1
    universe[x+2][y-1] = 1
    universe[x+2][y] = 1
    universe[x+2][y+1] = 1

def lwss(x, y):
    universe[x+1][y] = 1
    universe[x+2][y] = 1
    universe[x+3][y] = 1
    universe[x+4][y] = 1
    universe[x][y+1] = 1
    universe[x+4][y+1] = 1
    universe[x+4][y+2] = 1
    universe[x][y+3] = 1
    universe[x+3][y+3] = 1
    
def printDoubleArray(array):
    for i in range(array.__len__()):
        s = "[ "
        for j in range(array.__len__()):
            s = s + str(array[i][j]) + " "
        s = s + "]"
        print(s)
        
setInitialConditions()

while(True):
    rate(5)
    drawUniverse()
    updateUniverse()
    generationCounter += 1
    print(generationCounter)
