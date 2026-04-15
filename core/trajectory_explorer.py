import numpy as np

from core.trajectory_optimization import sample_shaped_joint_trajectory

POLYNOMIAL_DEGREE = 7
NUM_SAMPLES = 300
DEFAULT_DURATION = 5.0
SLIDER_MIN = -100.0
SLIDER_MAX = 100.0

DEFAULT_Q_START = np.zeros(6, dtype=float)
DEFAULT_Q_GOAL = np.array([0.0, np.pi, np.pi / 3.0, 0.0, 0.0, 0.0], dtype=float)


def _format_joint_vector(q: np.ndarray) -> str:
    return ", ".join(f"{value:.6g}" for value in q)


def _parse_joint_vector(text: str, expected_size: int = 6) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.size != expected_size:
        raise ValueError(f"Expected {expected_size} comma-separated values, got {values.size}.")
    return values


class TrajectoryExplorer:
    def __init__(self) -> None:
        self.q_start = DEFAULT_Q_START.copy()
        self.q_goal = DEFAULT_Q_GOAL.copy()
        self.duration = DEFAULT_DURATION
        self.shape_params = np.zeros(12, dtype=float)

        self._build_figure()
        self._update_plot()

    def _build_figure(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button, Slider, TextBox
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for the trajectory explorer. Install it with 'pip install matplotlib'."
            ) from exc

        self.plt = plt
        self.Slider = Slider
        self.TextBox = TextBox
        self.Button = Button

        self.fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        self.axes = axes.ravel()
        self.fig.subplots_adjust(left=0.08, right=0.73, bottom=0.22, top=0.92, hspace=0.35)

        self.baseline_lines = []
        self.shaped_lines = []
        for joint_idx, ax in enumerate(self.axes):
            baseline_line, = ax.plot([], [], "--", linewidth=1.8, label="Baseline Quintic")
            shaped_line, = ax.plot([], [], linewidth=2.0, label="7th Degree")
            ax.set_title(f"Joint {joint_idx + 1}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("q [rad]")
            ax.grid(True, alpha=0.3)
            self.baseline_lines.append(baseline_line)
            self.shaped_lines.append(shaped_line)

        self.axes[0].legend(loc="upper right")
        self.status_text = self.fig.text(0.08, 0.955, "", fontsize=10)

        self.duration_slider = Slider(
            self.fig.add_axes([0.08, 0.13, 0.58, 0.03]),
            "Duration [s]",
            0.5,
            15.0,
            valinit=self.duration,
        )
        self.duration_slider.on_changed(self._on_duration_change)

        self.shape_sliders = []
        self.shape_slider_keys = []
        slider_y_positions = np.linspace(0.80, 0.24, 6)
        slider_columns = [0.74, 0.86]
        for coeff_idx in range(2):
            for joint_idx, y in enumerate(slider_y_positions):
                slider = Slider(
                    self.fig.add_axes([slider_columns[coeff_idx], y, 0.10, 0.025]),
                    f"c{joint_idx + 1},{coeff_idx + 1}",
                    SLIDER_MIN,
                    SLIDER_MAX,
                    valinit=0.0,
                )
                slider.on_changed(self._on_shape_change)
                self.shape_sliders.append(slider)
                self.shape_slider_keys.append((joint_idx, coeff_idx))

        self.q_start_box = TextBox(
            self.fig.add_axes([0.08, 0.05, 0.26, 0.045]),
            "q_start",
            initial=_format_joint_vector(self.q_start),
        )
        self.q_goal_box = TextBox(
            self.fig.add_axes([0.40, 0.05, 0.26, 0.045]),
            "q_goal",
            initial=_format_joint_vector(self.q_goal),
        )

        self.apply_button = Button(self.fig.add_axes([0.68, 0.05, 0.10, 0.045]), "Apply q")
        self.apply_button.on_clicked(self._on_apply_q)

        self.reset_button = Button(self.fig.add_axes([0.80, 0.05, 0.10, 0.045]), "Reset shape")
        self.reset_button.on_clicked(self._on_reset_shape)

    def _compute_trajectories(self):
        baseline = sample_shaped_joint_trajectory(
            q_start=self.q_start,
            q_goal=self.q_goal,
            duration=self.duration,
            shape_params=np.zeros(12, dtype=float),
            polynomial_degree=POLYNOMIAL_DEGREE,
            num_samples=NUM_SAMPLES,
        )
        shaped = sample_shaped_joint_trajectory(
            q_start=self.q_start,
            q_goal=self.q_goal,
            duration=self.duration,
            shape_params=self.shape_params,
            polynomial_degree=POLYNOMIAL_DEGREE,
            num_samples=NUM_SAMPLES,
        )
        return baseline, shaped

    def _update_plot(self) -> None:
        baseline, shaped = self._compute_trajectories()

        for joint_idx, ax in enumerate(self.axes):
            self.baseline_lines[joint_idx].set_data(baseline.t, baseline.q[:, joint_idx])
            self.shaped_lines[joint_idx].set_data(shaped.t, shaped.q[:, joint_idx])
            ax.relim()
            ax.autoscale_view()

        self.status_text.set_text(
            "Boundary conditions enforced automatically: "
            "q(0), q(T), qdot(0)=qdot(T)=0, qddot(0)=qddot(T)=0"
        )
        self.fig.canvas.draw_idle()

    def _on_duration_change(self, value: float) -> None:
        self.duration = float(value)
        self._update_plot()

    def _on_shape_change(self, _value: float) -> None:
        coeffs = np.zeros((6, 2), dtype=float)
        for slider, (joint_idx, coeff_idx) in zip(self.shape_sliders, self.shape_slider_keys):
            coeffs[joint_idx, coeff_idx] = slider.val
        self.shape_params = coeffs.reshape(-1)
        self._update_plot()

    def _on_apply_q(self, _event) -> None:
        try:
            self.q_start = _parse_joint_vector(self.q_start_box.text)
            self.q_goal = _parse_joint_vector(self.q_goal_box.text)
        except ValueError as exc:
            self.status_text.set_text(f"Invalid q input: {exc}")
            self.fig.canvas.draw_idle()
            return

        self._update_plot()

    def _on_reset_shape(self, _event) -> None:
        for slider in self.shape_sliders:
            slider.reset()

    def show(self) -> None:
        self.plt.show()


if __name__ == "__main__":
    explorer = TrajectoryExplorer()
    explorer.show()
