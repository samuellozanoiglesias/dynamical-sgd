import numpy as np
from scipy.optimize import least_squares

# ---- Data extracted from the heatmap (test_accuracy) ----
T_list = [2000,1800,1500,1200,1000,800,400,200,100,50,20,10,5]
w_list = [1,5,10,20,50,70,100,150,250]

grid = [
 [0.78,0.78,0.78,0.79,0.77,0.77,0.75,0.75,0.70],  # T=2000
 [0.78,0.77,0.78,0.78,0.77,0.76,0.76,0.73,0.67],  # T=1800
 [0.78,0.79,0.79,0.78,0.77,0.77,0.76,0.73,0.70],  # T=1500
 [0.78,0.78,0.78,0.79,0.77,0.77,0.76,0.74,0.69],  # T=1200
 [0.78,0.79,0.79,0.79,0.78,0.78,0.78,0.76,0.67],  # T=1000
 [0.78,0.79,0.79,0.79,0.79,0.78,0.77,0.75,0.68],  # T=800
 [0.78,0.78,0.79,0.79,0.79,0.79,0.77,0.73,0.62],  # T=400
 [0.78,0.79,0.79,0.80,0.78,0.78,0.75,0.66,0.24],  # T=200
 [0.78,0.79,0.79,0.80,0.78,0.72,0.64,0.30,0.17],  # T=100
 [0.78,0.79,0.80,0.80,0.62,0.28,0.22,0.13,0.11],  # T=50
 [0.78,0.80,0.78,0.58,0.19,0.29,0.20,0.18,0.11],  # T=20
 [0.78,0.79,0.78,0.63,0.45,0.33,0.33,0.26,0.26],  # T=10
 [0.78,0.80,0.79,0.75,0.52,0.50,0.42,0.38,0.36],  # T=5
]
grid = np.array(grid)

Ts = np.array(T_list, dtype=float)
Ws = np.array(w_list, dtype=float)

TT, WW = np.meshgrid(Ts, Ws, indexing='ij')
TT = TT.ravel(); WW = WW.ravel(); ACC = grid.ravel()

# drop the w_max=1 control column from the fit (it's the "no bumps" baseline,
# epsilon_max(1)=0 identically, nothing to explain there)
mask = WW > 1
TTm, WWm, ACCm = TT[mask], WW[mask], ACC[mask]

def eps_max(w, c):
    return (w - 1.0) / (w + c - 1.0)

def model(params, T, w):
    A0, beta, c, delta0, p, T0base, sigma = params
    e = eps_max(w, c)
    A_ceiling = A0 - beta * e
    Delta = delta0 * e
    T0 = T0base * np.power(w, p)          # power-law resonance frontier
    L = np.exp(-(np.log(T) - np.log(T0))**2 / (2*sigma**2))
    return A_ceiling - Delta * L

def resid(params):
    return model(params, TTm, WWm) - ACCm

# initial guesses: A0, beta, c, delta0, p, T0base, sigma
x0 = [0.80, 0.08, 10.0, 0.65, 0.5, 5.0, 1.0]
lb = [0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.2]
ub = [0.85, 0.3, 300.0, 1.0, 1.5, 50.0, 3.0]

res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
pred = model(res.x, TTm, WWm)
rmse = np.sqrt(np.mean((pred-ACCm)**2))
ss_res = np.sum((pred-ACCm)**2)
ss_tot = np.sum((ACCm-np.mean(ACCm))**2)
r2 = 1 - ss_res/ss_tot

names = ["A0","beta","c(~K-1)","delta0","p (exponent)","T0base","sigma"]
print("Fitted parameters:")
for n,v in zip(names,res.x):
    print(f"  {n:15s} = {v:.4f}")
print(f"\nRMSE = {rmse:.4f}   R^2 = {r2:.4f}   (accuracy units, 0-1 scale)")
print(f"\n==> Danger-frontier exponent p = {res.x[4]:.3f}  (T0(w_max) ~ w_max^p)")
print("    p=1 -> linear/diagonal,  p=0.5 -> sqrt,  p->0 -> log-like")

# report resonance period T0(w) for the w values in the grid, and compare to
# rough "trough" T read visually from the grid for sanity
A0,beta,c,delta0,p,T0base,sigma = res.x
print("\nw_max -> fitted resonance period T0(w) [steps]  (this is the predicted worst-T)")
for w in w_list[1:]:
    T0 = T0base*(w**p)
    print(f"  w_max={w:5d}  ->  T0 = {T0:8.1f}")

# also print predicted vs actual grid for inspection
print("\nPredicted grid (rows=T desc, cols=w_max, w=1 excluded):")
pred_grid = model(res.x, TT.reshape(grid.shape), WW.reshape(grid.shape))
np.set_printoptions(precision=2, suppress=True)
print("T\\w   " + "  ".join(f"{w:5d}" for w in w_list))
for i,T in enumerate(T_list):
    row = pred_grid[i]
    print(f"{T:5d} " + " ".join(f"{v:5.2f}" for v in row))