from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class DroneState:
    """State vector x = [x, y, z, vx, vy, vz, roll, pitch, yaw]^T"""
    position: np.ndarray       # Shape (3,) in local metric frame
    velocity: np.ndarray       # Shape (3,) m/s
    orientation: np.ndarray    # Shape (3,) Euler angles in radians


class DroneEKF:
    """Extended Kalman Filter for fusing Visual Odometry with GPS and Altimeter

    to constrain metric scale drift on single-pass flight trajectories.
    """

    def __init__(
        self,
        process_noise_pos: float = 0.05,
        process_noise_vel: float = 0.2,
        process_noise_rot: float = 0.01,
        gps_noise: float = 0.5,
        alt_noise: float = 0.5,
    ):
        # 9-dimensional state vector: [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, yaw]
        self.state = np.zeros(9, dtype=np.float64)

        # State covariance matrix P (9x9)
        self.P = np.eye(9, dtype=np.float64) * 1.0

        # Process noise covariance Q (9x9)
        self.Q = np.diag([
            process_noise_pos, process_noise_pos, process_noise_pos,
            process_noise_vel, process_noise_vel, process_noise_vel,
            process_noise_rot, process_noise_rot, process_noise_rot,
        ]) ** 2

        # Measurement noise covariance for GPS/Alt (3x3)
        self.R_gps = np.diag([gps_noise, gps_noise, alt_noise]) ** 2

    def initialize(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        """Initializes the filter state with the first valid GPS/telemetry fix."""
        self.state[0:3] = initial_position.copy()
        if initial_velocity is not None:
            self.state[3:6] = initial_velocity.copy()
        else:
            self.state[3:6] = 0.0
        self.P = np.eye(9, dtype=np.float64) * 0.1

    def predict(self, dt: float, gyro_rates: Optional[np.ndarray] = None):
        """State propagation step based on constant-velocity motion model."""
        if dt <= 0.0:
            return

        # State transition Jacobian F
        F = np.eye(9, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Propagate position using velocity
        self.state[0:3] += self.state[3:6] * dt

        # Propagate orientation with gyro angular velocity if provided
        if gyro_rates is not None:
            self.state[6:9] += gyro_rates * dt

        # Propagate covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, measured_pos: np.ndarray):
        """Measurement update using absolute GPS/altimeter coordinates in local metric frame."""
        H = np.zeros((3, 9), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Innovation (residual)
        y = measured_pos - self.state[0:3]

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R_gps

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Joseph form covariance update for numerical symmetry & stability
        I = np.eye(9, dtype=np.float64)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T

    @staticmethod
    def estimate_relative_motion(
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        camera_k: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Recovers relative camera rotation (R) and unit translation (t)

        from 2D-2D feature correspondences using 5-point Essential Matrix algorithm.
        """
        if len(kpts0) < 8:
            return False, None, None

        E, inlier_mask = cv2.findEssentialMat(
            kpts0,
            kpts1,
            camera_k,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or E.shape != (3, 3):
            return False, None, None

        num_inliers, R, t, _ = cv2.recoverPose(E, kpts0, kpts1, camera_k, mask=inlier_mask)

        if num_inliers < 8:
            return False, None, None

        return True, R, t.flatten()

    def get_state(self) -> DroneState:
        return DroneState(
            position=self.state[0:3].copy(),
            velocity=self.state[3:6].copy(),
            orientation=self.state[6:9].copy(),
        )


if __name__ == "__main__":
    print("Testing DroneEKF sensor fusion...")
    ekf = DroneEKF()

    # Ground truth: Drone flying along X axis at 5 m/s at 50m altitude
    simulated_pos = np.array([0.0, 0.0, 50.0], dtype=np.float64)
    velocity = np.array([5.0, 0.0, 0.0], dtype=np.float64)

    # Initialize EKF with the first GPS fix
    ekf.initialize(initial_position=simulated_pos, initial_velocity=velocity)

    for step in range(50):
        dt = 0.1
        simulated_pos += velocity * dt

        # Propagate filter
        ekf.predict(dt=dt)

        # GPS measurement update every 0.5s (every 5 steps) with 0.3m sensor noise
        if step % 5 == 0:
            gps_reading = simulated_pos + np.random.normal(0.0, 0.3, size=3)
            ekf.update_gps(gps_reading)

    final_state = ekf.get_state()
    error = float(np.linalg.norm(final_state.position - simulated_pos))

    print(f"Final True Position: {np.round(simulated_pos, 2)}")
    print(f"Final EKF Position:  {np.round(final_state.position, 2)}")
    print(f"Position Error:      {error:.2f} meters")

    assert error < 1.0, f"Error {error:.2f}m exceeded 1.0m tolerance!"
    print("EKF Fusion Verification Successful!")