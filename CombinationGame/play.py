import pygame
import CombTask.CombinationGame as CG
# Define some colors
BLACK = (149, 79, 2)
WHITE = (246, 219, 198)
GREEN = (139, 172, 151)
RED = (214, 50, 46)
 
env = CG(2)
env.
# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid_dim = (7, 7)
grid = []
for row in range(grid_dim[0]):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(grid_dim[1]):
        grid[row].append(0)  # Append a cell
 
# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
grid[1][5] = 1
 
# Initialize pygame
pygame.init()
 
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [450, 450]

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = int(WINDOW_SIZE[1]/(grid_dim[1]+(grid_dim[1]+1)/4))
HEIGHT = int(WINDOW_SIZE[0]/(grid_dim[0]+(grid_dim[0]+1)/4))
# This sets the margin between each cell
MARGIN = WIDTH//4
print(WIDTH, HEIGHT, MARGIN)

screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Array Backed Grid")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            column = pos[0] // (WIDTH + MARGIN)
            row = pos[1] // (HEIGHT + MARGIN)
            # Set that location to one
            grid[row][column] = 1
            print("Click ", pos, "Grid coordinates: ", row, column)
 
    # Set the screen background
    screen.fill(BLACK)
 
    # Draw the grid
    for row in range(grid_dim[0]):
        for column in range(grid_dim[1]):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])
 
    # Limit to 60 frames per second
    clock.tick(60)
 
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
 
# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()