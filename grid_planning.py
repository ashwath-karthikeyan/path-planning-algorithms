import pygame
from math import sqrt, atan2, pi

# Constants
WIDTH, HEIGHT = 1200, 800
NODE_RADIUS = 10
NODE_DISTANCE = 40

# PyGame stuff
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
OBSTACLE_COLOR = (128, 128, 128)

# PyGame setup
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dijkstra's Pathfinding Animation with Obstacles")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)  # Create a font object

def draw_nodes_and_edges(nodes, final_path, open_set, closed_set, start, goal, iteration_count, path_length):
    win.fill(BLACK)
    for ox, oy, ow, oh in obstacles:
        pygame.draw.rect(win, OBSTACLE_COLOR, (ox, oy, ow, oh))
    for node in nodes.values():
        color = YELLOW if node.position in final_path else BLUE if node in closed_set else WHITE if node in open_set else WHITE
        if node == start:
            color = GREEN
        if node == goal:
            color = RED
        pygame.draw.circle(win, color, node.position, NODE_RADIUS)
    
    # Render iteration count and path length in the top right corner
    iterations_surf = font.render(f"Iterations: {iteration_count}", True, WHITE)
    path_length_surf = font.render(f"Path Length: {path_length}", True, WHITE)
    iterations_pos = (WIDTH - iterations_surf.get_width() - 200, 5)  # Top right, adjust margin
    path_length_pos = (WIDTH - path_length_surf.get_width() - 30, 5)  # Below the iterations count
    
    win.blit(iterations_surf, iterations_pos)
    win.blit(path_length_surf, path_length_pos)

# Planner setup
class Node:
    def __init__(self, position):
        self.position = position
        self.g_cost = float('inf')
        self.parent = None
        self.neighbors = []
        self.direction = None

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

def is_within_obstacle(x, y, obstacles):
    for ox, oy, ow, oh in obstacles:
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            return True
    return False

def create_graph(obstacles):
    nodes = {}
    for x in range(NODE_DISTANCE, WIDTH, NODE_DISTANCE):
        for y in range(NODE_DISTANCE, HEIGHT, NODE_DISTANCE):
            if not is_within_obstacle(x, y, obstacles):
                nodes[(x, y)] = Node((x, y))
    for node in nodes.values():
        x, y = node.position
        four_connected = [
            (-NODE_DISTANCE, 0), (NODE_DISTANCE, 0), (0, -NODE_DISTANCE), (0, NODE_DISTANCE)
            ]
        eight_connected = [
            (-NODE_DISTANCE, 0), (NODE_DISTANCE, 0),
            (0, -NODE_DISTANCE), (0, NODE_DISTANCE),
            (-NODE_DISTANCE, -NODE_DISTANCE), (NODE_DISTANCE, NODE_DISTANCE),
            (NODE_DISTANCE, -NODE_DISTANCE), (-NODE_DISTANCE, NODE_DISTANCE)  # Diagonal connections
        ]
        for dx, dy in eight_connected:
            neighbor_position = (x + dx, y + dy)
            if neighbor_position in nodes:
                node.add_neighbor(nodes[neighbor_position])
    return nodes

def calculate_direction(from_node, to_node):
    return atan2(to_node.position[1] - from_node.position[1], to_node.position[0] - from_node.position[0])

# Planner algorithm
def dijkstra_search(start, goal):
    open_set = set()
    closed_set = set()
    start.g_cost = 0
    open_set.add(start)

    while open_set:
        current_node = min(open_set, key=lambda n: n.g_cost)
        if current_node == goal:
            return reconstruct_path(current_node)
        open_set.remove(current_node)
        closed_set.add(current_node)

        for neighbor in current_node.neighbors:
            if neighbor in closed_set:
                continue
            temp_g_cost = current_node.g_cost + sqrt((neighbor.position[0] - current_node.position[0])**2 + (neighbor.position[1] - current_node.position[1])**2)
            if temp_g_cost < neighbor.g_cost:
                neighbor.g_cost = temp_g_cost
                neighbor.parent = current_node
                open_set.add(neighbor)
            yield current_node, open_set, closed_set, []

def a_star_search(start, goal):
    open_set = set()
    closed_set = set()
    start.g_cost = 0
    start.f_cost = start.g_cost + heuristic(start, goal)
    start.direction = None
    open_set.add(start)

    while open_set:
        current_node = min(open_set, key=lambda n: n.f_cost)
        if current_node == goal:
            return reconstruct_path(current_node)
        open_set.remove(current_node)
        closed_set.add(current_node)

        for neighbor in current_node.neighbors:
            if neighbor in closed_set:
                continue

            new_direction = calculate_direction(current_node, neighbor)
            distance = sqrt((neighbor.position[0] - current_node.position[0])**2 + (neighbor.position[1] - current_node.position[1])**2)
            temp_g_cost = current_node.g_cost + distance

            # Apply turn penalty
            if current_node.direction is not None:
                angle_difference = abs(new_direction - current_node.direction)
                if angle_difference > 0:
                    temp_g_cost += 4  # Apply turn penalty

            if temp_g_cost < neighbor.g_cost:
                neighbor.g_cost = temp_g_cost
                neighbor.f_cost = neighbor.g_cost + heuristic(neighbor, goal)
                neighbor.parent = current_node
                neighbor.direction = new_direction  # Update direction
                if neighbor not in open_set:
                    open_set.add(neighbor)

            yield current_node, open_set, closed_set, []

def heuristic(node, goal):
    #0 for euclidean, 1 for manhattan, 2 for octile
    norm = 2
    if norm == 0:
        return sqrt((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2)

    elif norm == 1:
        return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])
    
    elif norm == 2:
        dx = abs(node.position[0] - goal.position[0])
        dy = abs(node.position[1] - goal.position[1])
        return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Reverse path

# Obstacle definitions (top-left corner x, top-left corner y, width, height)
obstacles = [
    (400, 200, 200, 400), (800, 600, 200, 100)
]

nodes = create_graph(obstacles)
start_position = (NODE_DISTANCE, NODE_DISTANCE)
goal_position = (WIDTH - NODE_DISTANCE, HEIGHT - NODE_DISTANCE)
start_node = nodes[start_position]
goal_node = nodes[goal_position]
generator = a_star_search(start_node, goal_node)

running = True
final_path = []
iteration_count = 0
path_length = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not final_path:
        try:
            current_node, open_set, closed_set, _ = next(generator)
            iteration_count += 1
        except StopIteration as e:
            final_path = e.value
            path_length = len(final_path)
    draw_nodes_and_edges(nodes, final_path, open_set, closed_set, start_node, goal_node, iteration_count, path_length)
    pygame.display.update()
    clock.tick(100)

pygame.quit()