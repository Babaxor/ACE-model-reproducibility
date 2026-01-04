"""
Plotting utilities for the ACE model.

Generates figures from the manuscript (Figure 1 and Figure 2).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_stability_interval(model, c_range=None, n_points=500, 
                           figsize=(10, 6), save_path=None):
    """
    Plot stability dependence on c (Figure 1 in manuscript).
    
    Parameters
    ----------
    model : ACEModel
        ACE model instance (used to get parameter values)
    c_range : tuple, optional
        Range of c values to plot. If None, uses [0, U*1.5]
    n_points : int
        Number of points in c sweep
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if c_range is None:
        c_range = (0, model.U * 1.5)
    
    c_vals = np.linspace(c_range[0], c_range[1], n_points)
    
    # Compute L and U (they depend on other parameters, not c)
    L = model.L
    U = model.U
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stability regions
    ax.axvspan(L, U, alpha=0.3, color='green', label='Architect stable')
    ax.axvspan(c_range[0], L, alpha=0.3, color='red', label='Tyrant wins')
    ax.axvspan(U, c_range[1], alpha=0.3, color='blue', label='Victim wins')
    
    # Add vertical lines for bounds
    ax.axvline(L, color='black', linestyle='--', alpha=0.7, label=f'L = {L:.3f}')
    ax.axvline(U, color='black', linestyle='--', alpha=0.7, label=f'U = {U:.3f}')
    
    # Mark example value
    ax.axvline(model.c, color='purple', linewidth=2, 
               label=f'Example c = {model.c}')
    
    # Add text annotations
    ax.text(L/2, 0.5, 'Tyrant\nwins', ha='center', va='center', 
            fontsize=12, transform=ax.get_xaxis_transform())
    ax.text((L+U)/2, 0.5, 'Architect\nstable', ha='center', va='center', 
            fontsize=12, transform=ax.get_xaxis_transform())
    ax.text((U + c_range[1])/2, 0.5, 'Victim\nwins', ha='center', va='center', 
            fontsize=12, transform=ax.get_xaxis_transform())
    
    # Formatting
    ax.set_xlabel('Self-cost intensity $c$', fontsize=14)
    ax.set_ylabel('Stability region', fontsize=14)
    ax.set_title(f'Stability dependence on $c$ '
                f'($\\tilde{{a}}={model.a_tilde}$, '
                f'$b={model.b}$, $\\gamma={model.gamma}$, '
                f'$p^*={model.p_star}$)', fontsize=14)
    ax.set_xlim(c_range)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_policy_feasibility(feasibility_data, figsize=(10, 8), 
                           save_path=None):
    """
    Plot policy feasibility map (Figure 2 in manuscript).
    
    Parameters
    ----------
    feasibility_data : dict
        Output from compute_policy_feasibility
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    B = feasibility_data['b_grid']
    Gamma = feasibility_data['gamma_grid']
    R = feasibility_data['stability_margin']
    feasible = feasibility_data['feasible']
    c_policy = feasibility_data['c_policy']
    p_star = feasibility_data['p_star']
    
    # Create custom colormap: red for negative, green for positive
    colors = ['red', 'lightgray', 'green']
    cmap = LinearSegmentedColormap.from_list('feasibility', colors, N=256)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feasibility map
    im = ax.pcolormesh(B, Gamma, R, cmap=cmap, 
                      vmin=-1.0, vmax=1.0, shading='auto')
    
    # Add contour for R=0 boundary
    ax.contour(B, Gamma, R, levels=[0], colors='black', linewidths=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Stability margin $R$', fontsize=14)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Fail (R<0)', 'Marginal (R=0)', 'Work (R>0)'])
    
    # Formatting
    ax.set_xlabel('Cooperation cost $b$', fontsize=14)
    ax.set_ylabel('Responsibility benefit $\\gamma$', fontsize=14)
    ax.set_title(f'Policy feasibility for $c_{{policy}} = {c_policy}$ '
                f'($p^* = {p_star}$)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text annotation
    ax.text(0.05, 0.95, 'Green: Policy works\nRed: Policy fails', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_separatrix_curve(separatrix_data, model=None, figsize=(10, 6),
                         save_path=None):
    """
    Plot separatrix curve V_crit(xT).
    
    Parameters
    ----------
    separatrix_data : dict
        Output from compute_separatrix_curve
    model : ACEModel, optional
        Model instance for parameter info
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    xT_vals = separatrix_data['xT_values']
    V_crit_vals = separatrix_data['V_crit_values']
    
    # Remove NaN values
    valid_mask = ~np.isnan(V_crit_vals)
    xT_vals = xT_vals[valid_mask]
    V_crit_vals = V_crit_vals[valid_mask]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot separatrix curve
    ax.plot(xT_vals, V_crit_vals, 'b-', linewidth=2, label='Separatrix')
    ax.fill_between(xT_vals, 0, V_crit_vals, alpha=0.3, color='blue',
                   label='Architect basin')
    ax.fill_between(xT_vals, V_crit_vals, V_crit_vals.max()*1.1, 
                   alpha=0.3, color='red', label='Tyrant basin')
    
    # Formatting
    ax.set_xlabel('Initial Tyrant frequency $x_T(0)$', fontsize=14)
    ax.set_ylabel('Critical initial resource $V_{crit}(0)$', fontsize=14)
    
    if model:
        title = (f'Separatrix curve ($\\tilde{{a}}={model.a_tilde}$, '
                f'$b={model.b}$, $\\gamma={model.gamma}$, '
                f'$c={model.c}$, $p^*={model.p_star}$)')
    else:
        title = 'Separatrix curve (basin boundary)'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, xT_vals.max()*1.1)
    ax.set_ylim(0, V_crit_vals.max()*1.1)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_parameter_sensitivity(sensitivity_data, figsize=(12, 8), 
                              save_path=None):
    """
    Plot sensitivity analysis results.
    
    Parameters
    ----------
    sensitivity_data : dict
        Output from sensitivity_analysis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    param = sensitivity_data['parameter']
    values = sensitivity_data['values']
    L_vals = sensitivity_data['L_values']
    U_vals = sensitivity_data['U_values']
    width = sensitivity_data['width']
    stable = sensitivity_data['stable']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: L and U bounds
    ax1 = axes[0, 0]
    ax1.plot(values, L_vals, 'r-', linewidth=2, label='Lower bound L')
    ax1.plot(values, U_vals, 'b-', linewidth=2, label='Upper bound U')
    ax1.fill_between(values, L_vals, U_vals, alpha=0.3, color='green',
                    label='Stable interval')
    ax1.set_xlabel(f'${param}$', fontsize=12)
    ax1.set_ylabel('Bounds', fontsize=12)
    ax1.set_title(f'Stability bounds vs ${param}$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Interval width
    ax2 = axes[0, 1]
    ax2.plot(values, width, 'g-', linewidth=2)
    ax2.set_xlabel(f'${param}$', fontsize=12)
    ax2.set_ylabel('Interval width $U - L$', fontsize=12)
    ax2.set_title(f'Stability interval width vs ${param}$', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Binary stability
    ax3 = axes[1, 0]
    ax3.plot(values, stable.astype(float), 'k-', linewidth=2)
    ax3.set_xlabel(f'${param}$', fontsize=12)
    ax3.set_ylabel('Stable (1) / Unstable (0)', fontsize=12)
    ax3.set_title(f'Stability vs ${param}$', fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Unstable', 'Stable'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative sensitivity
    ax4 = axes[1, 1]
    # Compute normalized sensitivity
    if len(values) > 1:
        d_width = np.diff(width)
        d_param = np.diff(values)
        sensitivity = d_width / d_param
        param_mid = (values[:-1] + values[1:]) / 2
        ax4.plot(param_mid, sensitivity, 'm-', linewidth=2)
        ax4.set_xlabel(f'${param}$', fontsize=12)
        ax4.set_ylabel('Sensitivity $\\frac{d(U-L)}{d' + param + '}$', fontsize=12)
        ax4.set_title(f'Sensitivity of interval width to ${param}$', fontsize=12)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor sensitivity', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_axis_off()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes