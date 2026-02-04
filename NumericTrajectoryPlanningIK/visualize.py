# visualize.py (FAST)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kinematics import forwardKinematicsT
from constraints import clampQ

RAD2DEG = 180.0 / np.pi

def limitCheck(q, jointLimits):
    q = np.asarray(q, dtype=float)
    lim = np.asarray(jointLimits, dtype=float)
    lo, hi = lim[:, 0], lim[:, 1]
    violMask = (q < lo) | (q > hi)
    viol = np.flatnonzero(violMask).tolist()
    return (not viol), viol, violMask

def animate(
    traj,
    dt,
    jointLimits=None,
    showTrace=True,
    traceLen=300,

    # PERFORMANCE CONTROLS
    renderFps=30,
    tableFps=10,
    showTable=True,
    tableRows=12,

    view=(20, 45),
):
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError("traj must be shape (N, 6)")

    if jointLimits is None:
        jointLimits = [(-np.pi, np.pi)] * 6

    renderStep = max(1, int(round((1.0 / renderFps) / dt)))
    renderIdx = np.arange(0, len(traj), renderStep, dtype=int)
    if renderIdx[-1] != len(traj) - 1:
        renderIdx = np.append(renderIdx, len(traj) - 1)

    tableStep = max(1, int(round((1.0 / tableFps) / dt)))

    fig = plt.figure(figsize=(12, 7))
    if showTable:
        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.8])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        axTbl = fig.add_subplot(gs[0, 1])
        axTbl.axis("off")
    else:
        ax = fig.add_subplot(111, projection="3d")
        axTbl = None

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.2)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    ax.view_init(elev=view[0], azim=view[1])

    robotLine, = ax.plot([], [], [], "-o", lw=2)
    jointLabels = [
        ax.text(0, 0, 0, f"{i+1}", fontsize=10, weight="bold", ha="center", va="center")
        for i in range(6)
    ]

    traceLine = None
    if showTrace:
        traceLine, = ax.plot([], [], [], "--", lw=1)

    eePts = []

    hud = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")

    headers = ["t (s)", "q1 (°)", "q2 (°)", "q3 (°)", "q4 (°)", "q5 (°)", "q6 (°)"]
    tblRef = [None]
    lastTableUpdateK = -10**9

    def makeTableRows(k):
        start = max(0, k - tableRows + 1)
        idx = np.arange(start, k + 1, dtype=int)
        rows = []
        for i in idx:
            qDeg = traj[i] * RAD2DEG
            rows.append([f"{i*dt:5.2f}"] + [f"{a: .1f}°" for a in qDeg])
        return rows, idx

    def recolorTable(rows_idx):
        tbl = tblRef[0]
        if tbl is None:
            return
        HeaderBG = (0.92, 0.92, 0.92, 1.0)
        CellBG = (1.0, 1.0, 1.0, 1.0)
        ViolMaskRow = (1.0, 0.85, 0.85, 1.0)

        cells = tbl.get_celld()
        for (r, c), cell in cells.items():
            cell.set_facecolor(HeaderBG if r == 0 else CellBG)

        for r in range(1, len(rows_idx) + 1):
            kk = rows_idx[r - 1]
            ok, _, _ = limitCheck(traj[kk], jointLimits)
            if not ok:
                for c in range(len(headers)):
                    tbl[(r, c)].set_facecolor(ViolMaskRow)

    def buildTable(k):
        rows, idx = makeTableRows(k)
        axTbl.clear()
        axTbl.axis("off")
        tblRef[0] = axTbl.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
        tblRef[0].auto_set_font_size(False)
        tblRef[0].set_fontsize(9)
        tblRef[0].scale(1.05, 1.25)
        recolorTable(idx)

    def init():
        robotLine.set_data([], [])
        robotLine.set_3d_properties([])
        if traceLine is not None:
            traceLine.set_data([], [])
            traceLine.set_3d_properties([])
        hud.set_text("")
        if showTable:
            buildTable(0)
        return (robotLine, hud) if traceLine is None else (robotLine, traceLine, hud)

    def update(frame_i):
        nonlocal lastTableUpdateK

        k = int(renderIdx[frame_i])
        qPlan = traj[k]
        qCmd, hit, viol = clampQ(qPlan, jointLimits)
        _, pts = forwardKinematicsT(qCmd, returnPoints=True)

        for j in range(6):
            jointLabels[j].set_position((pts[j+1, 0], pts[j+1, 1]))
            jointLabels[j].set_3d_properties(pts[j+1, 2])

        robotLine.set_data(pts[:, 0], pts[:, 1])
        robotLine.set_3d_properties(pts[:, 2])
        robotLine.set_color("red" if hit else "C0")

        if showTrace and traceLine is not None:
            eePts.append(pts[-1])
            if traceLen is not None and len(eePts) > traceLen:
                del eePts[:-traceLen]
            ee = np.asarray(eePts)
            traceLine.set_data(ee[:, 0], ee[:, 1])
            traceLine.set_3d_properties(ee[:, 2])

        qDeg = qCmd * RAD2DEG
        hudText = f"t = {k*dt:0.2f}s\n" + "\n".join(
            [f"q{j+1} = {qDeg[j]: .1f}°" for j in range(6)]
        )
        if hit:
            hudText += f"\nCLAMPED joints: {viol}"
        hud.set_text(hudText)

        if showTable and axTbl is not None:
            if (k - lastTableUpdateK) >= tableStep or hit:
                buildTable(k)
                lastTableUpdateK = k

        return (robotLine, hud) if traceLine is None else (robotLine, traceLine, hud)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(renderIdx),
        init_func=init,
        interval=1000.0 / renderFps,
        blit=False,
        repeat=False,
    )
    plt.show()
    return ani
