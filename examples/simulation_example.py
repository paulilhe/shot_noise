from shot_noise import simulation as sim
import numpy as np
import matplotlib.pyplot as plt

# definition des elements d'un shot-noise
intensity =1

def h(x):
    return 25 * x ** 2 * np.exp(2 - 10 * x) * (x >= 0)

def mixed_gaussian(n):
    U = np.random.rand(n) >= 0.5
    Y = [2 + .1 * np.random.randn() if u else np.pi + .2 * np.random.randn() for u in U]
    return Y

sn = sim.ShotNoise(impulse_response=h, intensity=intensity)
sn.impulse_response = h
sn.intensity = .2
sn.marks_density = mixed_gaussian

sn.simulate(500, 0.01);

print("Slicing a ShotNoise")

sub_sn = sn[150:2000]

plt.clf()
plt.figure(figsize=(15,6))
plt.plot(sub_sn.signal)
plt.stem(sub_sn.times, sub_sn.marks, 'r')
plt.savefig("sub_sn.png")

plt.clf()
plt.figure(figsize=(15, 6))
plt.hist(sn.marks, bins=50);
plt.savefig("sn_marks.png")

plt.clf()
plt.figure(figsize=(15, 6))
plt.hist(np.diff(sn.times), bins=50);
plt.savefig("sn_times.png")

plt.clf()
plt.plot(sn.eval_kernel(0.01, 300))
plt.savefig("sn_eval_kernel.png")

print("Done!")

