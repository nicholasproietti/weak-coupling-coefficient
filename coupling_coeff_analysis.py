# Author: Nicholas Proietti
# Code that calculates the coupling coefficient for a given model number through fitting the mixed modes

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mesa_reader as mr

# User-defined inputs and filenames
profile_number = 70
profiles_index_file = 'profiles.index'
pms_data_file = 'pms_history.data'
ms_data_file = 'ms_history.data'
rgb_data_file = 'rgb_history.data'
summary_file = 'summary.txt'
output_dir = './plots/'
plot_file_hr = output_dir + 'HR_diagram.png'
plot_file_l1 = output_dir + 'l1_modes.png'
plot_file_l0 = output_dir + 'l0_modes_fit.png'
plot_file_fitted_l1 = output_dir + 'fitted_l1_modes.png'

# Constants
nu_max_solar = 3100  # micro Hz
nu_ac_solar = 5300  # micro Hz
Teff_solar = 5777  # K

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# FUNCTIONS

def get_model_number(profiles_index_file, target_profile_number):
    with open(profiles_index_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 3:
                model_number, priority, profile_number = parts
                if int(profile_number) == target_profile_number:
                    return int(model_number)
    return None

# Combined function to get stellar parameters (Teff, radius, mass, luminosity)
def get_stellar_params(history_data, target_model_number):
    for i, model_number in enumerate(history_data.model_number):
        if model_number == target_model_number:
            Teff = 10**history_data.log_Teff[i]  # Effective temperature in K
            radius_solar = 10**history_data.log_R[i]  # Radius in solar radii
            mass_solar = history_data.star_mass[i]  # Mass in solar masses
            luminosity_solar = 10**history_data.log_L[i]  # Luminosity in solar units
            return Teff, radius_solar, mass_solar, luminosity_solar
    return None, None, None, None

# Read the frequencies and radial orders from the summary file
def read_frequencies_and_orders(summary_file):
    data = np.genfromtxt(summary_file, skip_header=5)
    
    # Extract relevant columns
    l_values = data[:, 3]         # l values
    n_pg_values = data[:, 6]      # n_pg values
    n_p_values = data[:, 5]       # n_p values
    freq_values = data[:, 1]      # Re(freq) values (in microhertz)
    
    return l_values, n_pg_values, n_p_values, freq_values

def find_delta_pg(history_data_file, target_model_number):
    with open(history_data_file, 'r') as file:
        lines = file.readlines()
        
        # Find the header line for the data columns
        for idx, line in enumerate(lines):
            if "model_number" in line:
                header_idx = idx
                break
        
        header = lines[header_idx].split()
        model_number_index = header.index('model_number')
        delta_pg_index = header.index('delta_Pg')
        
        # Iterate over the data lines starting from the header index + 1
        for line in lines[header_idx + 1:]:
            parts = line.split()
            if len(parts) == len(header):
                model_number = int(parts[model_number_index])
                if model_number == target_model_number:
                    return float(parts[delta_pg_index])
    return None

# Load MESA data
pms_data = mr.MesaData(pms_data_file)
ms_data = mr.MesaData(ms_data_file)
rgb_data = mr.MesaData(rgb_data_file)

# Find the model number corresponding to the profile number
model_number = get_model_number(profiles_index_file, profile_number)

if model_number is not None:
    # Find the period spacing (delta_Pg) for the found model number
    delta_pg = find_delta_pg(rgb_data_file, model_number)
    if delta_pg is not None:
        print(f"Model Number: {model_number}")
        print(f"Period Spacing (delta_Pg): {delta_pg}")
    else:
        print(f"Period spacing (delta_Pg) not found for model number {model_number}.")
else:
    print(f"Profile number {target_profile_number} not found.")

# Find the corresponding Teff, radius, mass, and luminosity for the model number
Teff, radius_solar, mass_solar, luminosity_solar = get_stellar_params(rgb_data, model_number)

# Create a plot for the HR diagram
plt.figure(figsize=(10, 8))
plt.plot(pms_data.Teff, pms_data.log_L, color='black', marker='none', linestyle='-')
plt.plot(ms_data.Teff, ms_data.log_L, color='black', marker='none', linestyle='-')
plt.plot(rgb_data.Teff, rgb_data.log_L, color='black', marker='none', linestyle='-')

# Mark the specific point on the plot
if Teff is not None and luminosity_solar is not None:
    plt.scatter(Teff, np.log10(luminosity_solar), color='red', s=100, 
                label=(f'Mass: {mass_solar:.2f} $M_{{\odot}}$\n'
                       f'Radius: {radius_solar:.2f} $R_{{\odot}}$\n'
                       f'Teff: {Teff:.2f} K\n'
                       f'Luminosity: {luminosity_solar:.2f} $L_{{\odot}}$'))

plt.xlabel('Effective Temperature [K]')
plt.ylabel(r'$\log \, L/L_{\odot}$')
plt.gca().invert_xaxis()
plt.legend()
plt.title('Hertzsprung-Russell Diagram')
plt.grid(False)

# Save the HR diagram plot
plt.savefig(plot_file_hr)
plt.close()

# Calculate nu_max and nu_ac 
if Teff is not None and radius_solar is not None:
    # Calculate nu_max
    nu_max = nu_max_solar * (mass_solar / (radius_solar**2 * np.sqrt(Teff / Teff_solar)))
    
    # Calculate the acoustic cut-off frequency (nu_ac, from Eq. 2 of Jimenez et al. 2011)
    nu_ac_1 = nu_ac_solar * (mass_solar / luminosity_solar) * (Teff / Teff_solar)**3.5
    
    # Load the summary.txt file
    data = np.genfromtxt(summary_file, skip_header=5)
    
    # Extract the columns
    E_norm = data[:, 0]
    Re_freq = data[:, 1]
    Im_freq = data[:, 2]
    l = data[:, 3]
    n_g = data[:, 4]
    n_p = data[:, 5]
    n_pg = data[:, 6]

    # Filter the data for l=1
    l_equals_1 = l == 1
    Re_freq_l_1 = Re_freq[l_equals_1]
    n_p_l_1 = n_p[l_equals_1]

    # Plot the Re(freq) modes for l=1 as a function of n_p
    plt.figure(figsize=(10, 6))
    plt.plot(n_p_l_1, Re_freq_l_1, marker='o', linestyle='-', color='b', label='Mode frequencies for l=1')
    
    # Plot nu_max and nu_ac as dashed lines
    plt.axhline(y=nu_max, color='r', linestyle='--', label=r'$\nu_{\mathrm{max}}$')
    plt.axhline(y=nu_ac_1, color='g', linestyle='--', label=r'$\nu_{\mathrm{ac}}$')

    plt.xlabel('acoustic-wave winding number n_p')
    plt.ylabel(r'$\nu$ (µHz)')
    plt.title('Mode Frequency for l=1 as a Function of n_p')
    plt.legend()
    plt.grid(True)

    # Save the l=1 mode plot
    plt.savefig(plot_file_l1)
    plt.close()

    # Filter for l=0 modes
    l_equals_0 = l == 0
    n_p_l_0 = n_p[l_equals_0]
    Re_freq_l_0 = Re_freq[l_equals_0]

    # Function for linear fitting
    def linear_fit(x, a, b):
        return a * x + b

    # Perform linear fitting
    popt, _ = curve_fit(linear_fit, n_p_l_0, Re_freq_l_0)

    # Plot the l=0 data and the fitted line
    plt.figure(figsize=(10, 6))
    plt.scatter(n_p_l_0, Re_freq_l_0, color='blue', label='Data')
    plt.plot(n_p_l_0, linear_fit(n_p_l_0, *popt), color='red', label=f'Fitted line: $\\Delta\\nu = {popt[0]:.2f}$ µHz')
    plt.xlabel('Radial Order (n_p)')
    plt.ylabel('Frequency (µHz)')
    plt.title('Frequency vs Radial Order for l=0 Modes')
    plt.legend()
    plt.grid(True)

    # Save the l=0 mode fitting plot
    plt.savefig(plot_file_l0)
    plt.close()

    delta_nu = popt[0]

    # Filter to only include l=1 modes
    l_values, n_pg_values, n_p_values, nu_values = read_frequencies_and_orders(summary_file)
    l1_indices = np.where(l_values == 1)[0]
    n_p_l1 = n_p_values[l1_indices]
    nu_values_l1 = nu_values[l1_indices]

    # Filter to include l=0 modes for calculating the uncoupled p-mode frequencies
    l0_indices = np.where(l_values == 0)[0]
    n_p_l0 = n_p_values[l0_indices]
    nu_values_l0 = nu_values[l0_indices]

    # Calculate uncoupled p-mode frequencies
    nu_np_l1 = []
    for n_p in n_p_l1:
        idx_current = np.where((l_values == 0) & (n_p_values == n_p))[0]
        idx_next = np.where((l_values == 0) & (n_p_values == n_p + 1))[0]
        
        if len(idx_current) > 0 and len(idx_next) > 0:
            nominal_p_mode_freq = (nu_values_l0[idx_current[0]-1] + nu_values_l0[idx_next[0]-1]) / 2
        elif len(idx_current) > 0:
            nominal_p_mode_freq = nu_values_l0[idx_current[0]-1]
        elif len(idx_next) > 0:
            nominal_p_mode_freq = nu_values_l0[idx_next[0]-1]
        else:
            raise ValueError(f"No valid l=0, n_p={n_p} or n_p={n_p+1} frequency found.")
        
        nu_np_l1.append(nominal_p_mode_freq)

    nu_np_l1 = np.array(nu_np_l1)

    # Asymptotic relation for mixed modes (Mosser et al. 2012)
    def mixed_mode_relation(nu, epsilon_g, q, delta_pg, delta_nu):
        term1 = np.pi * (1 / (delta_pg * nu) - epsilon_g)
        return nu_np_l1 + (delta_nu / np.pi) * np.arctan(q * np.tan(term1))

    # Perform a grid search over possible values of q and epsilon
    q_grid = np.linspace(0.000001, 1, 1000)  # grid for q
    epsilon_grid = np.linspace(0.00001, 1, 1000)  # grid for epsilon

    # Define a function to calculate residuals
    def calculate_residuals(nu_values, epsilon, q, delta_pg, delta_nu):
        predicted_nu = mixed_mode_relation(nu_values, epsilon, q, delta_pg, delta_nu)
        return np.sum((nu_values - predicted_nu) ** 2)

    # Perform grid search
    best_q = 0
    best_epsilon = 0
    min_residuals = float('inf')

    for epsilon in epsilon_grid:
        for q in q_grid:
            residuals = calculate_residuals(nu_values_l1, epsilon, q, delta_pg, delta_nu)
            if residuals < min_residuals:
                min_residuals = residuals
                best_q = q
                best_epsilon = epsilon

    print(f'Grid Search Results: Best q = {best_q}, Best epsilon = {best_epsilon}')

    # Bounds for delta_pg and delta_nu to allow small flexibility
    delta_pg_bounds = [delta_pg * 0.95, delta_pg * 1.05]
    delta_nu_bounds = [delta_nu * 0.95, delta_nu * 1.05]

    # Use least-squares fitting to refine the estimates with flexible delta_pg and delta_nu
    popt, pcov = curve_fit(
        lambda nu, epsilon, q, delta_pg_flex, delta_nu_flex: mixed_mode_relation(nu, epsilon, q, delta_pg_flex, delta_nu_flex), 
        nu_values_l1, 
        nu_values_l1, 
        p0=[best_epsilon, best_q, delta_pg, delta_nu], 
        bounds=([0, 0, delta_pg_bounds[0], delta_nu_bounds[0]], [np.inf, 1, delta_pg_bounds[1], delta_nu_bounds[1]])
    )

    epsilon_fit, q_fit, delta_pg_fit, delta_nu_fit = popt

    # Calculate percentage differences
    delta_pg_percent_diff = ((delta_pg_fit - delta_pg) / delta_pg) * 100
    delta_nu_percent_diff = ((delta_nu_fit - delta_nu) / delta_nu) * 100

    # Print the fitted parameters with units
    print(f"Fitted epsilon: {epsilon_fit}")
    print(f"Fitted q: {q_fit}")
    print(f"Fitted delta_pg: {delta_pg_fit} s")  # delta_pg in seconds
    print(f"Fitted delta_nu: {delta_nu_fit} µHz")  # delta_nu in microhertz
    print(f"Percentage difference in delta_pg: {delta_pg_percent_diff:.2f}%")
    print(f"Percentage difference in delta_nu: {delta_nu_percent_diff:.2f}%")

    # Calculate the fitted frequencies
    nu_fitted = mixed_mode_relation(nu_values_l1, epsilon_fit, q_fit, delta_pg_fit, delta_nu_fit)

    # Plot the observed and fitted frequencies
    plt.figure(figsize=(10, 6))
    plt.plot(nu_values_l1[:len(nu_values_l1)], nu_values_l1[:len(nu_values_l1)], 'bo', label='Observed Frequencies')
    plt.plot(nu_values_l1[:len(nu_values_l1)], nu_fitted[:len(nu_fitted)], 'ro', label=f'Fitted Frequencies, q = {q_fit:.4f}')
    plt.xlabel(r'$\nu$ (µHz)')
    plt.ylabel(r'$\nu$ (µHz)')
    plt.title('Observed vs Fitted Frequencies for l=1 Modes')
    plt.legend()
    plt.grid(True)
    
    # Save the fitted l=1 mode plot
    plt.savefig(plot_file_fitted_l1)
    plt.close()
else:
    print(f"Could not find stellar parameters for model number {model_number}")
