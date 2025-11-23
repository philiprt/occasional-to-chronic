import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def create_sigmoid_features_hemi(x_values, center=0, scale=10):
    """
    Create a single sigmoid function for latitude feature transformation.

    Parameters:
    - x_values: latitude values in range [-90, 90]
    - center: center point of sigmoid in latitude degrees (e.g., 0 for equator)
    - scale: steepness of sigmoid curve (higher = steeper)
    """
    # No normalization needed - work directly with latitude values
    sigmoid_feature = 1 / (1 + np.exp(-scale * (x_values - center)))
    return sigmoid_feature.reshape(-1, 1)


def create_combined_features(x, center_s, scale_s, center_n, scale_n):
    """
    Create combined feature matrix for latitude dependency fitting.

    Parameters:
    - x: latitude values in degrees [-90, 90]
    - center_s: center of southern hemisphere sigmoid
    - scale_s: steepness of southern hemisphere sigmoid
    - center_n: center of northern hemisphere sigmoid
    - scale_n: steepness of northern hemisphere sigmoid

    Returns:
    - X_combined: feature matrix with shape (n_samples, n_features)
    """

    # Southern hemisphere sigmoid: 1 - sigmoid(x - center_s)
    # This makes it start at 1 for negative latitudes and approach 0 at equator
    X_sigmoid_s_raw = create_sigmoid_features_hemi(x, center=center_s, scale=scale_s)
    X_sigmoid_s = 1 - X_sigmoid_s_raw  # Flip to get desired shape

    # Northern hemisphere sigmoid: sigmoid(x - center_n)
    # This starts at 0 at equator and approaches 1 at north pole
    X_sigmoid_n = create_sigmoid_features_hemi(x, center=center_n, scale=scale_n)

    # # Sigmoids only
    # X_combined = np.hstack([X_sigmoid_s, X_sigmoid_n])

    # Quadratic + sigmoids
    X_quad = np.hstack([x.reshape(-1, 1), (x**2).reshape(-1, 1)])
    X_combined = np.hstack([X_quad, X_sigmoid_s, X_sigmoid_n])

    # # Linear + sigmoids
    # X_linr = np.hstack([x.reshape(-1, 1)])  # Alternative: linear only
    # X_combined = np.hstack([X_linr, X_sigmoid_s, X_sigmoid_n])

    return X_combined


def ols_fit_hemi_sigmoid(x, y, center_s, scale_s, center_n, scale_n):
    """
    Fit OLS with hemisphere-specific sigmoid features.

    For latitude x in [-90, 90]:
    - Southern sigmoid: starts at 1 (south pole) → 0 (equator)
    - Northern sigmoid: starts at 0 (equator) → 1 (north pole)
    """
    X_combined = create_combined_features(x, center_s, scale_s, center_n, scale_n)

    ols = LinearRegression()
    ols.fit(X_combined, y)
    y_pred = ols.predict(X_combined)
    return y_pred, ols


def predict_hemi_sigmoid(x, ols_model, center_s, scale_s, center_n, scale_n):
    """
    Predict using a fitted OLS model with hemisphere-specific sigmoid features.
    Must match the features used in ols_fit_hemi_sigmoid.
    """
    X_combined = create_combined_features(x, center_s, scale_s, center_n, scale_n)
    return ols_model.predict(X_combined)


def optimize_sigmoid_parameters_hemi(x, y):
    """
    Optimize sigmoid parameters for hemisphere-specific sigmoids.
    """

    def objective(params):
        center_s, scale_s, center_n, scale_n = params

        try:
            y_pred, _ = ols_fit_hemi_sigmoid(x, y, center_s, scale_s, center_n, scale_n)
            return np.mean((y - y_pred) ** 2)
        except:
            return 1e10

    # Initial guess: southern center at -45°, northern center at 45°
    initial_guess = [-45, 0.05, 45, 0.05]  # Use latitude degrees, not 0-1 range

    # Bounds: southern center, southern scale, northern center, northern scale
    bounds = [(-90, 0), (0.0, 0.1), (0, 90), (0.0, 0.1)]

    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 100000},
    )

    # Return separate parameters for south and north
    return result.x[0], result.x[1], result.x[2], result.x[3]


def fit_latitude_dependence(x, y):
    """
    Two-stage fitting process for full latitude range [-90, 90].

    Parameters:
    - x: latitude values in degrees [-90, 90]
    - y: target values to fit (log-transformed)

    Returns:
    - y_pred: fitted latitude dependency
    - params: tuple of (center_s, scale_s, center_n, scale_n, ols_model)
    """

    idx = x.index.copy()

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Transform y values
    y_clean_transformed = np.sign(y_clean) * np.log1p(np.abs(y_clean))

    # Convert to numpy array if it's a pandas Series
    X = x_clean.values.flatten()

    # Optimize sigmoid parameters
    opt_center_s, opt_scale_s, opt_center_n, opt_scale_n = (
        optimize_sigmoid_parameters_hemi(X, y_clean_transformed)
    )

    # Create final fit with optimized parameters
    y_fit_transformed, ols_model = ols_fit_hemi_sigmoid(
        X, y_clean_transformed, opt_center_s, opt_scale_s, opt_center_n, opt_scale_n
    )

    # Transform back
    y_fit = np.sign(y_fit_transformed) * (np.exp(np.abs(y_fit_transformed)) - 1)

    # Store residuals, keeping NaN values where they were
    # Create arrays of same length by aligning indices
    y_aligned = pd.Series(y, index=idx)

    # Ensure x_clean has index attribute
    y_fit_aligned = pd.Series(y_fit, index=x_clean.index)
    y_rmlatdep = y_aligned - y_fit_aligned.reindex(idx)

    # Store predictions
    y_fit_aligned = pd.Series(y_fit, index=x_clean.index)
    y_latpred = y_fit_aligned.reindex(idx)

    params = (opt_center_s, opt_scale_s, opt_center_n, opt_scale_n, ols_model)
    return y_latpred, y_rmlatdep, params


def plot_latitude_dependence(
    ax1,
    ax2,
    var_name,
    var_col,
    analysis,
    analysis_latpred,
    fitted_params,
    colors,
    subplot_label=None,
):
    """
    Plot latitude dependence for a single variable on provided axes.

    Parameters:
    - ax1: axis for latitude vs observed scatter plot
    - ax2: axis for predicted vs observed scatter plot
    - var_name: display name for the variable (e.g., "$\Delta h$")
    - var_col: column name in the dataframe (e.g., "dh_median_26_days")
    - analysis: dataframe with original data
    - analysis_latpred: dataframe with latitude predictions
    - fitted_params: dictionary with fitted parameters for each variable
    - colors: list of colors for plotting
    - subplot_label: optional label for subplot (e.g., "a", "b", etc.)
    """
    # Get original data and predictions
    y = analysis[var_col]
    x = analysis["lat"]
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Get predictions from fitted model
    y_fit = analysis_latpred[var_col][mask]

    # Calculate r-squared
    r_squared = 1 - np.var(y_clean - y_fit) / np.var(y_clean)

    # Create scatter plot on first axis
    ax1.scatter(x_clean, y_clean, alpha=1, s=10, c=colors[3])

    # Add best fit curve - evaluate on regular grid for smooth plotting
    x_grid = np.arange(x_clean.min(), x_clean.max(), 0.1)

    # Get fitted parameters and evaluate on grid
    center_s, scale_s, center_n, scale_n, ols_model = fitted_params[var_col]
    y_grid_transformed = predict_hemi_sigmoid(
        x_grid, ols_model, center_s, scale_s, center_n, scale_n
    )
    # Transform back from log space
    y_grid = np.sign(y_grid_transformed) * (np.exp(np.abs(y_grid_transformed)) - 1)

    ax1.plot(x_grid, y_grid, "-", color=colors[6], alpha=1, lw=2)

    # Add subplot label and r-squared annotation
    if subplot_label is not None:
        ax1.annotate(
            subplot_label,
            xy=(0.05, 0.9),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
    ax1.annotate(
        f"r² = {r_squared:0.2f}",
        xy=(0.125, 0.9) if subplot_label else (0.05, 0.9),
        xycoords="axes fraction",
        fontsize=12,
    )

    ax1.set_title(var_name)
    ax1.set_ylabel("Observed (cm)")
    ax1.set_xlabel("Latitude (°N)")

    # Create predicted vs observed scatter plot on second axis
    ax2.scatter(y_fit, y_clean, alpha=1, s=10, c=colors[3])
    md = np.median(y_fit)
    ax2.axline((md, md), slope=1, color=colors[6], linestyle="-", alpha=1, lw=2)

    ax2.set_title(var_name)
    ax2.set_ylabel("Observed (cm)")
    ax2.set_xlabel("Latitude-Based Prediction (cm)")

    return r_squared
