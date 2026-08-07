import numpy as np

def compute_POI(E, N, U, Y, P, R):
    """
    Vectorized computation of POI on ground (U=0) using ENU coordinates.

    Convention:
    - Base LOS = (0, 1, 0)  (drone forward = North)
    - Yaw: positive clockwise  → internally use yaw = -Y
    - Pitch: positive = nose up
    - Roll:  positive = right wing down
    """

    # Convert to radians
    yaw   = np.radians(-Y)      # IMPORTANT: clockwise yaw → negate here
    pitch = np.radians(P)
    roll  = np.radians(R)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    # ---------------------------------------------------------
    # Apply rotations explicitly to base LOS = (0,1,0)
    # Rotation order: Roll → Pitch → Yaw
    # ---------------------------------------------------------

    # 1) Roll around N (forward vector unaffected)
    vx = 0
    vy = 1
    vz = 0

    # 2) Pitch around E
    vx2 = vx
    vy2 = cp * vy + (-sp) * vz       # = cp
    vz2 = sp * vy +  cp   * vz       # = sp

    # 3) Yaw around U (with corrected sign)
    dE = cy * vx2 - sy * vy2
    dN = sy * vx2 + cy * vy2
    dU = vz2

    # Normalize direction
    norm = np.sqrt(dE*dE + dN*dN + dU*dU)
    dE /= norm
    dN /= norm
    dU /= norm

    # ---------------------------------------------------------
    # Intersection with ground plane U = 0
    # ---------------------------------------------------------
    t = -U / dU
    valid = (np.abs(dU) > 1e-9) & (t >= 0)

    poi = np.full((len(E), 3), np.nan)
    poi[valid, 0] = E[valid] + t[valid] * dE[valid]
    poi[valid, 1] = N[valid] + t[valid] * dN[valid]
    poi[valid, 2] = 0.0

    return poi

def beam_2D_gauss(x,y,FWHM_H,FWHM_E):
    """
    Computes a 2D gaussian beam centered in (0,0) given the FWHM on the H and E planes.
    Input params:
        -x (np.array): x grid.
        -y (np.array): y grid.
        -FWHM_H (float): H plane FWHM in deg.
        -FWHM_E (float): E plane FWHM in deg.
    Output:
        - norm_beam
    """
    # --- Convert FWHM to sigma ---
    sigma_H = FWHM_H / (2 * np.sqrt(np.log(2)))
    sigma_E = FWHM_E / (2 * np.sqrt(np.log(2)))
    
    # compute the 2D Gaussian
    X, Y = np.meshgrid(x, y)
    beam = np.exp(-(X**2 / (sigma_H**2) + Y**2 / (sigma_E**2)))
    norm_beam = beam/beam.max()
    return norm_beam

def load_measured_beams(file_Hplane,file_Eplane,freq):
    
    arr_freq_H,arr_FWHM_H,arr_err_FWHM_H = np.loadtxt(file_Hplane,skiprows=1,unpack=True)
    arr_freq_E,arr_FWHM_E,arr_err_FWHM_E = np.loadtxt(file_Eplane,skiprows=1,unpack=True)

    freq_idx_H = (arr_freq_H==freq)
    freq_idx_E = (arr_freq_E==freq)

    FWHM_H,err_FWHM_H = arr_FWHM_H[freq_idx_H], arr_err_FWHM_H[freq_idx_H]
    FWHM_E,err_FWHM_E = arr_FWHM_E[freq_idx_E], arr_err_FWHM_H[freq_idx_E]

    return FWHM_H,err_FWHM_H,FWHM_E,err_FWHM_E

# build a new reference frame aligned with the drone:
def normalize(v):
    return v / np.linalg.norm(v)

def rot_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

# Point P → new frame origin;
# Point Q → direction the new frame should “look at”
def build_frame(P, Q, roll, up=np.array([0, 0, 1])):
    P = np.array(P)
    Q = np.array(Q)
    up = normalize(up)
    roll = np.radians(roll)

    # X axis points from P to Q
    x_axis = normalize(Q - P)

    # Prevent degeneracy
    if abs(np.dot(x_axis, up)) > 0.99:
        up = np.array([0, 1, 0])

    z_axis = up - np.dot(up,x_axis) * x_axis #normalize(np.cross(x_axis, up)) 
    z_axis= normalize(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix (columns = axes)
    R = np.column_stack((x_axis, y_axis, z_axis))

    # rotate to align the roll
    R_rot = R @ rot_x(roll)

    # Homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R_rot
    T[:3, 3] = P

    return T 

# T is a 4×4 homogeneous transform

#Converts local frame coordinates → world coordinates

#Inverse of T converts world → new frame

def world_to_local(T, point):
    T_inv = np.linalg.inv(T)
    p_h = np.append(point, 1)
    return (T_inv @ p_h)[:3]

def skew(v):
    """Skew-symmetric matrix for cross product"""
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

def axis_angle_to_rotation_matrix(axis, angle):
    """Rodrigues' rotation formula"""
    K = skew(axis)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def rotation_matrix_to_euler_zyx(R):
    """
    Convert rotation matrix to ZYX Euler angles
    yaw (Z), pitch (Y), roll (X)
    """
    if abs(R[2,0]) < 1:
        pitch = -np.arcsin(R[2,0])
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock
        pitch = np.pi/2 if R[2,0] <= -1 else -np.pi/2
        roll  = 0
        yaw   = np.arctan2(-R[0,1], R[1,1])

    return yaw, pitch, roll

def relative_yaw_pitch_roll(A, B):
    A = normalize(np.array(A))
    B = normalize(np.array(B))

    # Axis-angle
    axis = np.cross(A, B)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # Vectors are parallel
        return 0.0, 0.0, 0.0

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))

    # Rotation matrix
    R = axis_angle_to_rotation_matrix(axis, angle)

    # Euler angles
    return rotation_matrix_to_euler_zyx(R)

def get_azel(point3D):

    x = np.asarray(point3D[0])
    y = np.asarray(point3D[1])
    z = np.asarray(point3D[2])
    
    az = np.arctan(y/x)
    el = np.arccos(z/(np.sqrt(x**2+y**2+z**2)))
    return az,el
