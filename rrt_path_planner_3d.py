import random
import math
import numpy as np
import matplotlib.pyplot as plt

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def steer(a, b, step):
    d = dist(a, b)
    if d <= step:
        return b
    t = step / d
    return (a[0] + (b[0]-a[0])*t,
            a[1] + (b[1]-a[1])*t,
            a[2] + (b[2]-a[2])*t)

def collides(p, obstacles):
    # obstacles are spheres: (cx, cy, cz, r)
    x, y, z = p
    for cx, cy, cz, r in obstacles:
        if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= r*r:
            return True
    return False

def segment_hits_sphere(a, b, sph):
    # segment a->b intersects sphere (cx,cy,cz,r)
    cx, cy, cz, r = sph
    ax, ay, az = a
    bx, by, bz = b

    abx, aby, abz = bx-ax, by-ay, bz-az
    acx, acy, acz = cx-ax, cy-ay, cz-az

    ab2 = abx*abx + aby*aby + abz*abz
    if ab2 == 0:
        return (ax-cx)**2 + (ay-cy)**2 + (az-cz)**2 <= r*r

    t = (acx*abx + acy*aby + acz*abz) / ab2
    t = max(0.0, min(1.0, t))
    px, py, pz = ax + t*abx, ay + t*aby, az + t*abz

    return (px-cx)**2 + (py-cy)**2 + (pz-cz)**2 <= r*r

def edge_collides(a, b, obstacles):
    return any(segment_hits_sphere(a, b, s) for s in obstacles)

def rrt_3d(start, goal, obstacles, bounds, step=0.5, goal_sample=0.1, iters=10000):
    # bounds = (xmin,xmax, ymin,ymax, zmin,zmax)
    nodes = [start]
    parent = [-1]

    for _ in range(iters):
        # sample (goal bias)
        if random.random() < goal_sample:
            q_rand = goal
        else:
            q_rand = (random.uniform(bounds[0], bounds[1]),
                      random.uniform(bounds[2], bounds[3]),
                      random.uniform(bounds[4], bounds[5]))

        # nearest
        i_near = min(range(len(nodes)), key=lambda i: dist(nodes[i], q_rand))
        q_near = nodes[i_near]

        # steer
        q_new = steer(q_near, q_rand, step)

        # collision
        if collides(q_new, obstacles):
            continue
        if edge_collides(q_near, q_new, obstacles):
            continue

        # add
        nodes.append(q_new)
        parent.append(i_near)

        # reached?
        if dist(q_new, goal) < step and not collides(goal, obstacles) and not edge_collides(q_new, goal, obstacles):
            nodes.append(goal)
            parent.append(len(nodes) - 2)
            return nodes, parent

    return nodes, parent

def extract_path(nodes, parent):
    if len(nodes) < 2 or parent[-1] == -1:
        return None
    path = []
    i = len(nodes) - 1
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    return path[::-1]

def simplify_path(path, obstacles):
    # minimal-waypoint greedy shortcutting (farthest visible jump)
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

def draw_sphere(ax, cx, cy, cz, r):
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)

    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.3)  # transparent sphere


# ---- demo ----
start = (1, 1, 1)
goal  = (9, 9, 9)
bounds = (0, 10, 0, 10, 0, 10)

obstacles = [
    (5, 5, 5, 1.8),
    (3, 7, 6, 1.2),
    (7, 3, 4, 1.2),
]

nodes, parent = rrt_3d(start, goal, obstacles, bounds, step=0.5, goal_sample=0.15, iters=20000)
path = extract_path(nodes, parent)
path_s = simplify_path(path, obstacles) if path else None

# ---- plot ----
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[2], bounds[3])
ax.set_zlim(bounds[4], bounds[5])

# draw obstacles
for cx, cy, cz, r in obstacles:
    draw_sphere(ax, cx, cy, cz, r)

# draw tree edges
for i in range(1, len(nodes)):
    p = parent[i]
    if p != -1:
        ax.plot([nodes[i][0], nodes[p][0]],
                [nodes[i][1], nodes[p][1]],
                [nodes[i][2], nodes[p][2]], linewidth=0.5)

# start / goal
ax.scatter(*start)
ax.scatter(*goal)

# path
if path_s:
    ax.plot([p[0] for p in path_s],
            [p[1] for p in path_s],
            [p[2] for p in path_s], linewidth=3)

plt.show()
