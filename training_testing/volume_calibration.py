from scipy.ndimage import center_of_mass, rotate

angle = 1.9

rot1 = rotate(cavities[1], angle*180/np.pi, reshape=False, prefilter=True)
rot2 = rotate(cavities[1], angle*180/np.pi, reshape=False, prefilter=False)

rot1 = np.where(rot1 > 0.2, rot1, 0)
rot2 = np.where(rot2 > 0.2, rot2, 0)

rot1 = rotate(rot1, -angle*180/np.pi, reshape=False, prefilter=True)
rot2 = rotate(rot2, -angle*180/np.pi, reshape=False, prefilter=False)

rot1 = np.where(rot1 > 0.2, rot1, 0)
rot2 = np.where(rot2 > 0.2, rot2, 0)

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(cavities[1])

plt.subplot(132)
plt.imshow(rot1)

plt.subplot(133)
plt.imshow(rot2)

np.sum(cavities[0]), np.sum(rot1), np.sum(rot2)
