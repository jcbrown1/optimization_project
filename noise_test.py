import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
n = 100000
sigma_g = 1.0  # Gaussian std (baseline)

# Tail parameters to test
nu_values = [100]        # Student-t (smaller = heavier tails)
k_values = [100]       # Gamma (smaller = heavier/skewed)

# -----------------------------
# Base Gaussian
# -----------------------------
gaussian = np.random.normal(0, sigma_g, n)

# -----------------------------
# Student-t generator (matched variance)
# -----------------------------
def student_t_samples(n, sigma_g, nu):
    sigma = sigma_g * np.sqrt((nu - 2) / nu)
    z = np.random.normal(0, 1, n)
    u = np.random.chisquare(nu, n)
    return sigma * z / np.sqrt(u / nu)

# -----------------------------
# Gamma generator (centered + matched variance)
# -----------------------------
def gamma_samples(n, sigma_g, k):
    # theta = sigma_g / np.sqrt(k)
    theta = sigma_g / k
    g = np.random.gamma(shape=k, scale=theta, size=n)
    return g - k * theta  # center to zero mean
    # return g

# -----------------------------
# Plotting helper
# -----------------------------
def plot_distributions(data_dict, title):
    plt.figure()
    
    bins = 200
    for label, data in data_dict.items():
        plt.hist(data, bins=bins, density=True, histtype='step', label=label)
    
    plt.title(title)
    plt.legend()
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.show()

# # -----------------------------
# # Plot Student-t vs Gaussian
# # -----------------------------
# student_data = {"Gaussian": gaussian}
# for nu in nu_values:
#     student_data[f"Student-t (nu={nu})"] = student_t_samples(n, sigma_g, nu)

# plot_distributions(student_data, "Gaussian vs Student-t (matched variance)")

# # -----------------------------
# # Plot Gamma vs Gaussian
# # -----------------------------
# gamma_data = {"Gaussian": gaussian}
# for k in k_values:
#     gamma_data[f"Gamma (k={k})"] = gamma_samples(n, sigma_g, k)

# plot_distributions(gamma_data, "Gaussian vs Centered Gamma (matched variance)")

# -----------------------------
# Optional: All together (moderate tails)
# -----------------------------
combined = {
    "Gaussian": gaussian,
    "Student-t (nu=100)": student_t_samples(n, sigma_g, 3),
    # "Gamma (k=)": gamma_samples(n, sigma_g, 0.1),
}

plot_distributions(combined, "All Distributions (Comparable Settings)")