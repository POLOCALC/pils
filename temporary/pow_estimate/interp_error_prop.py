import numpy as np
from scipy.interpolate import interp1d
import numdifftools as nd


def mc_interp_sigma(x_cal, y_cal, dx_cal, dy_cal, x_new, dx_new, method='linear', n=2000):
    y_samps = np.empty(n)
    for i in range(n):
        xp = x_cal + np.random.normal(0, dx_cal)
        yp = y_cal + np.random.normal(0, dy_cal)
        x0p = x_new + np.random.normal(0, dx_new)
        order = np.argsort(xp)
        xp_s, yp_s = xp[order], yp[order]
        f = interp1d(xp_s, yp_s, kind=method, fill_value='extrapolate')
        y_samps[i] = f(x0p)
    return y_samps.mean(), y_samps.std(ddof=1)
    

def interpolate_with_errors(x_cal, y_cal, delta_x_cal, delta_y_cal, 
                           x_new, delta_x_new, method='linear'):
    """
    Interpolate y value with error propagation using numdifftools.
    
    Parameters:
    -----------
    x_cal : array-like
        Calibration x values
    y_cal : array-like
        Calibration y values
    delta_x_cal : array-like
        Uncertainties in calibration x values
    delta_y_cal : array-like
        Uncertainties in calibration y values
    x_new : float
        New x value to interpolate
    delta_x_new : float
        Uncertainty in new x value
    method : str, optional
        Interpolation method: 'linear', 'cubic', 'quadratic'
        Default is 'linear'
    
    Returns:
    --------
    y_new : float
        Interpolated y value
    delta_y_new : float
        Uncertainty in interpolated y value
    """
    
    # Convert to numpy arrays
    x_cal = np.asarray(x_cal)
    y_cal = np.asarray(y_cal)
    delta_x_cal = np.asarray(delta_x_cal)
    delta_y_cal = np.asarray(delta_y_cal)
    
    # Sort calibration data by x values
    sort_idx = np.argsort(x_cal)
    x_cal = x_cal[sort_idx]
    y_cal = y_cal[sort_idx]
    delta_x_cal = delta_x_cal[sort_idx]
    delta_y_cal = delta_y_cal[sort_idx]
    
    # Check if x_new is within calibration range
    if x_new < x_cal[0] or x_new > x_cal[-1]:
        print(f"Warning: x_new={x_new} is outside calibration range [{x_cal[0]}, {x_cal[-1]}]")
        print("Extrapolation may be unreliable!")
    
    # Interpolate y value
    f = interp1d(x_cal, y_cal, kind=method, fill_value='extrapolate')
    y_new = float(f(x_new))
    
    # Error propagation using numdifftools
    delta_y_new = _numdiff_error_propagation(
        x_cal, y_cal, delta_x_cal, delta_y_cal, x_new, delta_x_new, method
    )
    
    return y_new, delta_y_new


def _numdiff_error_propagation(x_cal, y_cal, delta_x_cal, delta_y_cal, 
                                x_new, delta_x_new, method):
    """
    Error propagation using numdifftools for accurate derivatives.
    """
    n_cal = len(x_cal)
    
    # Create a parameter vector: [x_new, x_cal[0], ..., x_cal[n-1], y_cal[0], ..., y_cal[n-1]]
    def interpolation_function(params):
        """
        Function that takes all parameters and returns interpolated y.
        params = [x_new_val, x_cal_vals..., y_cal_vals...]
        """
        x_new_val = params[0]
        x_cal_vals = params[1:n_cal+1]
        y_cal_vals = params[n_cal+1:2*n_cal+1]
        
        # Sort by x values
        sort_idx = np.argsort(x_cal_vals)
        x_cal_sorted = x_cal_vals[sort_idx]
        y_cal_sorted = y_cal_vals[sort_idx]
        
        # Interpolate
        f = interp1d(x_cal_sorted, y_cal_sorted, kind=method, fill_value='extrapolate')
        return float(f(x_new_val))
    
    # Current parameter values
    params = np.concatenate([[x_new], x_cal, y_cal])
    
    # Calculate gradient using numdifftools
    gradient = nd.Gradient(interpolation_function)(params)
    
    # Gradient components
    dy_dx_new = gradient[0]
    dy_dx_cal = gradient[1:n_cal+1]
    dy_dy_cal = gradient[n_cal+1:2*n_cal+1]
    
    # Error propagation (quadrature sum)
    var_x_new = (dy_dx_new * delta_x_new)**2
    var_x_cal = np.sum((dy_dx_cal * delta_x_cal)**2)
    var_y_cal = np.sum((dy_dy_cal * delta_y_cal)**2)
    
    delta_y = np.sqrt(var_x_new + var_x_cal + var_y_cal)
    
    return delta_y


def interpolate_with_errors_jacobian(x_cal, y_cal, delta_x_cal, delta_y_cal, 
                                     x_new_array, delta_x_new_array, method='linear'):
    """
    Vectorized version for multiple x_new values using Jacobian.
    
    Parameters:
    -----------
    x_cal : array-like
        Calibration x values
    y_cal : array-like
        Calibration y values
    delta_x_cal : array-like
        Uncertainties in calibration x values
    delta_y_cal : array-like
        Uncertainties in calibration y values
    x_new_array : array-like
        Array of new x values to interpolate
    delta_x_new_array : array-like
        Uncertainties in new x values
    method : str, optional
        Interpolation method: 'linear', 'cubic', 'quadratic'
    
    Returns:
    --------
    y_new_array : ndarray
        Interpolated y values
    delta_y_new_array : ndarray
        Uncertainties in interpolated y values
    """
    
    # Convert to numpy arrays
    x_cal = np.asarray(x_cal)
    y_cal = np.asarray(y_cal)
    delta_x_cal = np.asarray(delta_x_cal)
    delta_y_cal = np.asarray(delta_y_cal)
    x_new_array = np.asarray(x_new_array)
    delta_x_new_array = np.asarray(delta_x_new_array)
    
    # Sort calibration data
    sort_idx = np.argsort(x_cal)
    x_cal = x_cal[sort_idx]
    y_cal = y_cal[sort_idx]
    delta_x_cal = delta_x_cal[sort_idx]
    delta_y_cal = delta_y_cal[sort_idx]
    
    n_cal = len(x_cal)
    n_new = len(x_new_array)
    
    # Interpolate all y values
    f = interp1d(x_cal, y_cal, kind=method, fill_value='extrapolate')
    y_new_array = f(x_new_array)
    
    # Function for Jacobian calculation
    def interpolation_vector_function(params):
        """
        params = [x_new_vals..., x_cal_vals..., y_cal_vals...]
        returns = [y_new_vals...]
        """
        x_new_vals = params[:n_new]
        x_cal_vals = params[n_new:n_new+n_cal]
        y_cal_vals = params[n_new+n_cal:n_new+2*n_cal]
        
        # Sort by x values
        sort_idx = np.argsort(x_cal_vals)
        x_cal_sorted = x_cal_vals[sort_idx]
        y_cal_sorted = y_cal_vals[sort_idx]
        
        # Interpolate
        f = interp1d(x_cal_sorted, y_cal_sorted, kind=method, fill_value='extrapolate')
        return f(x_new_vals)
    
    # Current parameter values
    params = np.concatenate([x_new_array, x_cal, y_cal])
    
    # Calculate Jacobian using numdifftools
    jac = nd.Jacobian(interpolation_vector_function)(params)
    
    # Extract Jacobian blocks
    jac_x_new = jac[:, :n_new]  # dy/dx_new for each measurement
    jac_x_cal = jac[:, n_new:n_new+n_cal]  # dy/dx_cal
    jac_y_cal = jac[:, n_new+n_cal:n_new+2*n_cal]  # dy/dy_cal
    
    # Error propagation for each measurement
    delta_y_new_array = np.zeros(n_new)
    for i in range(n_new):
        var_x_new = (jac_x_new[i, i] * delta_x_new_array[i])**2
        var_x_cal = np.sum((jac_x_cal[i, :] * delta_x_cal)**2)
        var_y_cal = np.sum((jac_y_cal[i, :] * delta_y_cal)**2)
        delta_y_new_array[i] = np.sqrt(var_x_new + var_x_cal + var_y_cal)
    
    return y_new_array, delta_y_new_array


# Example usage
if __name__ == "__main__":
    # Example calibration data
    x_cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_cal = np.array([2.1, 4.3, 6.2, 8.1, 10.0])
    delta_x_cal = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    delta_y_cal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    print("="*60)
    print("Single measurement interpolation")
    print("="*60)
    
    # New measurement
    x_new = 2.5
    delta_x_new = 0.15
    
    # Interpolate with linear method
    y_new, delta_y_new = interpolate_with_errors(
        x_cal, y_cal, delta_x_cal, delta_y_cal,
        x_new, delta_x_new, method='linear'
    )
    
    print(f"Linear interpolation:")
    print(f"x_new = {x_new} ± {delta_x_new}")
    print(f"y_new = {y_new:.3f} ± {delta_y_new:.3f}")
    print()
    
    # Try cubic interpolation
    y_new_cubic, delta_y_new_cubic = interpolate_with_errors(
        x_cal, y_cal, delta_x_cal, delta_y_cal,
        x_new, delta_x_new, method='cubic'
    )
    
    print(f"Cubic interpolation:")
    print(f"x_new = {x_new} ± {delta_x_new}")
    print(f"y_new = {y_new_cubic:.3f} ± {delta_y_new_cubic:.3f}")
    
    print("\n" + "="*60)
    print("Multiple measurements interpolation (vectorized)")
    print("="*60)
    
    # Multiple new measurements
    x_new_array = np.array([1.5, 2.5, 3.5, 4.5])
    delta_x_new_array = np.array([0.1, 0.15, 0.12, 0.13])
    
    y_new_array, delta_y_new_array = interpolate_with_errors_jacobian(
        x_cal, y_cal, delta_x_cal, delta_y_cal,
        x_new_array, delta_x_new_array, method='cubic'
    )
    
    print(f"Cubic interpolation (vectorized):")
    for i in range(len(x_new_array)):
        print(f"x = {x_new_array[i]} ± {delta_x_new_array[i]:.3f} "
              f"→ y = {y_new_array[i]:.3f} ± {delta_y_new_array[i]:.3f}")
