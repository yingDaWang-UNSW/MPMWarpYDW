# import numpy as np
# import h5py

# # Cantilever beam dimensions
# beam_length = 1
# beam_height = 0.2
# beam_width = 0.2

# # Origin of the beam (bottom-left-front corner)
# origin_x, origin_y, origin_z = 0.1, 0.5, 0.5

# # Number of particles along each dimension
# num_particles_x = 100
# num_particles_y = 20
# num_particles_z = 20

# # Generate particle coordinates
# dx = beam_length / num_particles_x
# dy = beam_height / num_particles_y
# dz = beam_width / num_particles_z

# x_coords = np.linspace(origin_x + dx / 2, origin_x + beam_length - dx / 2, num_particles_x)
# y_coords = np.linspace(origin_y + dy / 2, origin_y + beam_height - dy / 2, num_particles_y)
# z_coords = np.linspace(origin_z + dz / 2, origin_z + beam_width - dz / 2, num_particles_z)


# # Create grid of particle positions
# X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
# particle_positions = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
# # particle_positions=particle_positions+(np.random.rand(particle_positions.shape[0],particle_positions.shape[1])*2-1)*0.01
# # Assign initial particle volumes (uniform, from grid spacing)
# particle_volume = np.full((particle_positions.shape[0],), dx * dy * dz)

# # Save to HDF5 file
# output_filename = "./warp_mpm/cantilever_beam_particles.h5"

# with h5py.File(output_filename, "w") as h5file:
#     h5file.create_dataset("x", data=particle_positions.T)
#     h5file.create_dataset("particle_volume", data=particle_volume)

# print(f"HDF5 file '{output_filename}' created successfully with {particle_positions.shape[0]} particles.")
import numpy as np
import scipy.ndimage as ndi
import h5py
import pyvista as pv

# --- Settings ---
img_size = 100
center = (50, 50)
outer_radius = 500
inner_radius = 15
layer_count = 10
spacing = 1  # Real-world spacing between pixels/layers

# --- Generate annulus mask using EDT ---
mask = np.zeros((img_size, img_size), dtype=np.uint8)
mask[center] = 1
edt = ndi.distance_transform_edt(1 - mask)  # Distance from center

# Binary annulus
annulus = (edt <= outer_radius) & (edt >= inner_radius)

# Keep top half only
yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing="ij")
top_half_mask = yy >= center[1]
arch_mask = annulus# & top_half_mask

# Get 2D coordinates of centroids (cell centers)
arch_indices = np.argwhere(arch_mask)
arch_coords_2d = (arch_indices + 0.5) * spacing  # cell-centered

# Extrude into 3D by repeating in Z
z_layers = np.arange(layer_count) * spacing
arch_coords_3d = np.vstack([
    np.hstack([arch_coords_2d, np.full((arch_coords_2d.shape[0], 1), z)]) for z in z_layers
])

# add an origin
arch_coords_3d=arch_coords_3d+25
arch_coords_3d=arch_coords_3d[:,[2,1,0]]
# Particle volumes (each cell)
particle_volume = np.full((arch_coords_3d.shape[0],), spacing**3)

# --- Save to HDF5 ---
h5_filename = "./annular_arch_particles.h5"
with h5py.File(h5_filename, "w") as h5file:
    h5file.create_dataset("x", data=arch_coords_3d.T)
    h5file.create_dataset("particle_volume", data=particle_volume)
print(f"Saved HDF5: {h5_filename} with {arch_coords_3d.shape[0]} particles")

# --- Save to VTP for ParaView ---
cloud = pv.PolyData(arch_coords_3d)
cloud["volume"] = particle_volume
vtp_filename = "./annular_arch_particles.vtp"
cloud.save(vtp_filename)
print(f"Saved VTP: {vtp_filename}")
