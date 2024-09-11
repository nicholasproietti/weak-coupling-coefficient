# Author: Nicholas Proietti
# Plots a propagration diagram and calculates the weak coupling coefficient q for specified modes

import pygyre as pg
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.integrate import quad, IntegrationWarning
import warnings
import mesa_reader as mr

# Controls
profile_number = 70 # MESA Profile Number
profiles_index_file = 'profiles.index' # Path to profiles.index file
rgb_data = mr.MesaData('rgb_history.data') # Load the MESA history data
plot_directory = 'plots' # Directory to save the plots
n_min = 1 # Set the desired range of radial orders (n) to illustrate (does not calculate q)
n_max = 1
specified_modes = [
    # Specify which modes to illustrate AND calculate q for
    ('detail.l1.n+56.h5', 'Specified mode (l=1, n=56)', 'red', '--'),
    ('detail.l1.n+90.h5', 'Specified mode (l=1, n=90)', 'green', '--'),
    ('detail.l1.n+130.h5', 'Specified mode (l=1, n=130)', 'purple', '--'),
    ('detail.l1.n+167.h5', 'Specified mode (l=1, n=167)', 'orange', '--')
]
spike_start = 0.3 # Define region of the buoyancy frequency spike (in fractional radius) to mask
spike_end = 0.312

# Constants
nu_max_solar = 3100  # micro Hz
nu_ac_solar = 5300  # micro Hz
Teff_solar = 5777  # K
solar_radius = 6.957e10  # cm

# Function to find the model number corresponding to a profile number
def find_model_number(profiles_index_file, target_profile_number):
    with open(profiles_index_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 3:
                model_number, priority, profile_number = parts
                if int(profile_number) == target_profile_number:
                    return int(model_number)
    return None

# Function to find stellar parameters from the history data
def find_stellar_parameters(history_data, target_model_number):
    for i, model_number in enumerate(history_data.model_number):
        if model_number == target_model_number:
            Teff = 10**history_data.log_Teff[i]  # Effective temperature in K
            radius = 10**history_data.log_R[i]  # Radius in solar radii
            luminosity = 10**history_data.log_L[i]  # Luminosity in solar units
            return Teff, radius, luminosity
    return None, None, None

# Find the model number corresponding to the profile number
model_number = find_model_number(profiles_index_file, profile_number)

# Get the stellar parameters for the corresponding model number
Teff, radius_solar, luminosity_solar = find_stellar_parameters(rgb_data, model_number)

# Convert radius to cm
star_radius = radius_solar * solar_radius

# Calculate nu_max
nu_max = nu_max_solar * (1 / (radius_solar**2 * np.sqrt(Teff / Teff_solar)))

# Calculate the acoustic cut-off frequency (eq. 2 of Jimenez et al. 2011)
nu_ac = nu_ac_solar * (1 / luminosity_solar) * (Teff / Teff_solar)**3.5

# Convert nu_max and nu_ac to omega_max and omega_ac in cyc/day
omega_max = 2 * np.pi * nu_max * 1e-6 * 86400
omega_ac = 2 * np.pi * nu_ac * 1e-6 * 86400

print(f"nu_max (micro Hz): {nu_max}")
print(f"nu_ac (micro Hz): {nu_ac}")

# Function to read the omega value from a GYRE detail file
def read_omega(file_path):
    d = pg.read_output(file_path)
    omega = d.meta['omega']
    return omega.real**2

# Function to read the data needed for plotting the propagation diagram
def read_propagation_data(file_path):
    d = pg.read_output(file_path)
    l = d.meta['l']
    omega = d.meta['omega']
    
    x = d['x']
    V = d['V_2'] * d['x']**2
    As = d['As']
    c_1 = d['c_1']
    Gamma_1 = d['Gamma_1']
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        N2 = d['As'] / d['c_1']
        Sl2 = np.where(V * c_1 != 0, l * (l + 1) * Gamma_1 / (V * c_1), np.nan)  # Avoid division by zero
    
    return x, N2, Sl2, omega.real**2

# Function to find crossing points
def find_crossing_points(x, y1, y2):
    interp_y1 = interp1d(x, y1, kind='linear', fill_value='extrapolate')
    interp_y2 = interp1d(x, y2, kind='linear', fill_value='extrapolate')
    crossings = []
    for i in range(len(x) - 1):
        if np.sign(interp_y1(x[i]) - interp_y2(x[i])) != np.sign(interp_y1(x[i + 1]) - interp_y2(x[i + 1])):
            try:
                root = brentq(lambda z: interp_y1(z) - interp_y2(z), x[i], x[i + 1])
                crossings.append(root)
            except ValueError:
                pass  # Ignore cases where brentq fails
    return crossings

# Function to extract the radial order (n) from the filename
def extract_n(file_path):
    match = re.search(r'n\+(\d+)', file_path)
    return int(match.group(1)) if match else None

# Function to compute the sound speed c_s from Sl and r
def compute_sound_speed(Sl2, l, x):
    r = x * star_radius  # Convert x to actual radius
    return np.sqrt(Sl2) * r / np.sqrt(l * (l + 1))

# Function to compute the radial wavenumber k_r
def wavenumber_kr(r, N2, Sl2, omega2, c_s):
    return np.sqrt(np.abs((omega2 / c_s**2) * ((Sl2 / omega2) - 1) * ((N2 / omega2) - 1)))

# Integrand for the weak coupling coefficient
def integrand(r, N2_interp, Sl2_interp, omega2, c_s_interp):
    kr_value = wavenumber_kr(r, N2_interp(r), Sl2_interp(r), omega2, c_s_interp(r))
    return kr_value

# Calculate the weak coupling coefficient q
def calculate_weak_coupling(x, N2, Sl2, omega2, c_s, x_buoyancy, x_lamb):
    r = x * star_radius  # Convert x to actual radius
    r_buoyancy = x_buoyancy * star_radius if x_buoyancy is not None else None
    r_lamb = x_lamb * star_radius if x_lamb is not None else None

    # Interpolate N2, Sl2, and c_s over the range of r
    N2_interp = interp1d(r, N2, kind='linear', fill_value='extrapolate')
    Sl2_interp = interp1d(r, Sl2, kind='linear', fill_value='extrapolate')
    c_s_interp = interp1d(r, c_s, kind='linear', fill_value='extrapolate')
    
    # Calculate the integral for the weak coupling coefficient
    integral = 0
    if r_buoyancy is not None and r_lamb is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            integral, _ = quad(integrand, r_buoyancy, r_lamb, args=(N2_interp, Sl2_interp, omega2, c_s_interp))
    
    # Calculate q
    q = 0.25 * np.exp(-2 * integral)
    
    return q

# Function to find the relevant crossings for the specified mode
def find_relevant_crossings(buoyancy_crossings, lamb_crossings):
    relevant_crossings = []
    
    # Check all combinations of buoyancy and lamb crossings
    for b_cross in buoyancy_crossings:
        for l_cross in lamb_crossings:
            if b_cross < l_cross:
                relevant_crossings.append((b_cross, l_cross))
            elif l_cross < b_cross:
                relevant_crossings.append((l_cross, b_cross))
                
    # If we have relevant crossings, choose the pair with the smallest separation
    if relevant_crossings:
        return min(relevant_crossings, key=lambda pair: abs(pair[0] - pair[1]))
    else:
        return None, None


plt.figure(figsize=(12, 8)) 

# Read and plot data for each specified mode
for file_path, label, color, linestyle in specified_modes:
    x, N2, Sl2, omega = read_propagation_data(file_path)
    
    # Interpolate to ignore the buoyancy frequency spike
    mask = (x < spike_start) | (x > spike_end)
    x_interp = x[mask]
    N2_interp = N2[mask]
    interp_func = interp1d(x_interp, N2_interp, kind='linear', fill_value='extrapolate')
    N2_smooth = N2.copy()
    N2_smooth[~mask] = interp_func(x[~mask])
    
    # Find crossing points for the specified mode
    buoyancy_crossings = find_crossing_points(x, N2_smooth, np.full_like(x, omega))
    lamb_crossings = find_crossing_points(x, Sl2, np.full_like(x, omega))
    
    # Filter to find the relevant crossings for the specified mode
    if buoyancy_crossings and lamb_crossings:
        buoyancy_crossing, lamb_crossing = find_relevant_crossings(buoyancy_crossings, lamb_crossings)
    else:
        buoyancy_crossing, lamb_crossing = None, None
    
    # Output the crossing points for the specified mode
    print(f"Mode frequency {label} (omega^2): {omega}")
    print(f"Buoyancy frequency crossing point (x): {buoyancy_crossing}")
    print(f"Lamb frequency crossing point (x): {lamb_crossing}")
    
    # Calculate sound speed c_s using the lamb frequency Sl
    c_s = compute_sound_speed(Sl2, 1, x)  # Given l=1 for all specified modes
    
    # Calculate q for the specified mode
    if buoyancy_crossing is not None and lamb_crossing is not None:
        q = calculate_weak_coupling(x, N2_smooth, Sl2, omega, c_s, buoyancy_crossing, lamb_crossing)
        q_value = round(q, 4)  # Round q to four decimal places
        print(f"Weak coupling coefficient (q): {q}\n")

    # Update the label to include the q value
    if q_value is not None:
        label += f' (q = {q_value})'
    
    # Plot the specified mode frequency with a unique color and linestyle
    plt.axhline(omega, linestyle=linestyle, color=color, label=label)
    
    # Plot the crossing points for the specified mode
    if buoyancy_crossing is not None:
        plt.plot(buoyancy_crossing, omega, 'o', color=color)
    if lamb_crossing is not None:
        plt.plot(lamb_crossing, omega, 'o', color=color)


detail_files = glob.glob('detail.l1.n+*.h5')

# Filter the detail files based on the desired range of radial orders (n)
filtered_files = [file for file in detail_files if n_min <= extract_n(file) <= n_max]

# Extract all the omega values for l=1 modes within the desired range of radial orders (n)
omega_values = [read_omega(file) for file in filtered_files]

# Plot the frequencies of the other modes within the specified range
for omega in omega_values:
    if all(omega != read_omega(file) for file, _, _, _ in specified_modes):  # Avoid plotting the specified modes again
        plt.axhline(omega, linestyle='dashed', color='blue')

# Plot the propagation diagram
plt.plot(x, N2_smooth, label='N^2')
plt.plot(x, Sl2, label='S_l^2')

# Plot omega_max and omega_ac as horizontal lines
plt.axhline(omega_max**2, color='black', linestyle='dashdot', label=f'omega_max^2 = {omega_max**2:.2f} (cyc/day)^2 (nu =  {nu_max:.2f} micro Hz)')
plt.axhline(omega_ac**2, color='brown', linestyle='dashdot', label=f'omega_ac^2 = {omega_ac**2:.2f} (cyc/day)^2 (nu =  {nu_ac:.2f} micro Hz)')

plt.xlabel('x')
plt.ylabel('omega^2')

plt.xlim(0, 0.2)
plt.ylim(5e1, 5e6)
plt.yscale('log')
plt.legend()

# Check if the directory exists; if not, create it
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

plot_file = os.path.join(plot_directory, 'propagation_diagram.png')

# Save the plot to the specified directory
plt.savefig(plot_file, dpi=300, bbox_inches='tight')

print(f"Plot saved to {plot_file}")