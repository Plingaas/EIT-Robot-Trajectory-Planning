import random, math
import matplotlib.pyplot as plt

def dist(a, b): 
    return math.hypot(a[0]-b[0], a[1]-b[1])

def steer(a, b, step):
    d = dist(a, b)
    if d <= step: 
        return b
    t = step / d
    return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)

def collides(p, obstacles):
    # obstacles are circles: (cx, cy, r)
    x, y = p
    for cx, cy, r in obstacles:
        if (x-cx)**2 + (y-cy)**2 <= r*r:
            return True
    return False

def segment_hits_circle(a, b, circle):
    # collision of segment a->b with circle (cx,cy,r)
    cx, cy, r = circle
    ax, ay = a; bx, by = b
    abx, aby = bx-ax, by-ay
    acx, acy = cx-ax, cy-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0:
        return (ax-cx)**2 + (ay-cy)**2 <= r*r
    t = max(0.0, min(1.0, (acx*abx + acy*aby)/ab2))
    px, py = ax + t*abx, ay + t*aby
    return (px-cx)**2 + (py-cy)**2 <= r*r

def edge_collides(a, b, obstacles):
    return any(segment_hits_circle(a, b, c) for c in obstacles)

def rrt(start, goal, obstacles, bounds, step=0.5, goal_sample=0.1, iters=5000):
    # tree: list of nodes + parent index for each node
    nodes = [start]
    parent = [-1]

    for _ in range(iters):
        # 1) sample
        if random.random() < goal_sample:
            q_rand = goal
        else:
            q_rand = (random.uniform(bounds[0], bounds[1]),
                      random.uniform(bounds[2], bounds[3]))

        # 2) nearest
        i_near = min(range(len(nodes)), key=lambda i: dist(nodes[i], q_rand))
        q_near = nodes[i_near]

        # 3) steer
        q_new = steer(q_near, q_rand, step)

        # 4) collision checks (point + edge)
        if collides(q_new, obstacles): 
            continue
        if edge_collides(q_near, q_new, obstacles):
            continue

        # 5) add node
        nodes.append(q_new)
        parent.append(i_near)

        # 6) check goal
        if dist(q_new, goal) < step and not edge_collides(q_new, goal, obstacles) and not collides(goal, obstacles):
            nodes.append(goal)
            parent.append(len(nodes)-2)
            return nodes, parent

    return nodes, parent  # maybe no path

def simplify_path(path, obstacles):
    # Greedy: from i, jump to the farthest j reachable by a straight line
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and edge_collides(path[i], path[j], obstacles):
            j -= 1
        out.append(path[j])
        i = j
    return out


def extract_path(nodes, parent):
    if parent[-1] == -1:
        return None
    path = []
    i = len(nodes) - 1
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    return path[::-1]


# --- demo ---
start = (1, 1)
goal  = (9, 9)
bounds = (0, 10, 0, 10)
obstacles = [
    (5, 5, 1.5),
    (3, 7, 1.0),
    (7, 3, 1.0),
]

nodes, parent = rrt(start, goal, obstacles, bounds, step=0.4, goal_sample=0.15, iters=8000)

path = extract_path(nodes, parent)
path = simplify_path(path, obstacles)

# plot
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])

# obstacles
for cx, cy, r in obstacles:
    ax.add_patch(plt.Circle((cx, cy), r, fill=False))

# tree edges
for i in range(1, len(nodes)):
    p = parent[i]
    if p != -1:
        ax.plot([nodes[i][0], nodes[p][0]], [nodes[i][1], nodes[p][1]], linewidth=0.5)

# start/goal
ax.plot(start[0], start[1], "o")
ax.plot(goal[0], goal[1], "o")

# path
if path:
    ax.plot([p[0] for p in path], [p[1] for p in path], linewidth=3)
    ax.set_title("RRT found a path")
else:
    ax.set_title("RRT did not reach the goal (try more iters / bigger goal_sample)")

plt.show()
