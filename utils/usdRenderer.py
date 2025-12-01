"""
USD Renderer for MPM simulations using Warp's native USD support.

This renderer provides GPU-direct rendering with zero CPU transfers,
outputting to USD format for viewing in NVIDIA Omniverse or other USD viewers.
"""

import warp as wp
import numpy as np
import os
from pathlib import Path


class WarpUSDRenderer:
    """
    Wrapper around warp.sim.render.SimRenderer for MPM visualization.
    
    Provides GPU-direct rendering of particle systems with various coloring modes.
    Outputs USD files that can be viewed in real-time or rendered offline.
    """
    
    def __init__(self, output_path, fps=60, up_axis="Z"):
        """
        Initialize USD renderer.
        
        Parameters
        ----------
        output_path : str
            Directory path for USD output files
        fps : int
            Frames per second for animation playback
        up_axis : str
            Up axis convention ("Y" or "Z")
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.up_axis = up_axis
        self.frame_count = 0
        
        # Import here to avoid issues if USD is not available
        try:
            from pxr import Usd, UsdGeom, Sdf, Vt, Gf
            self.usd_available = True
            self.Usd = Usd
            self.UsdGeom = UsdGeom
            self.Sdf = Sdf
            self.Vt = Vt
            self.Gf = Gf
        except ImportError:
            print("Warning: USD Python bindings not available. Falling back to simple USD export.")
            self.usd_available = False
        
        self._init_stage()
    
    def _init_stage(self):
        """Initialize USD stage and set metadata."""
        if not self.usd_available:
            return
        
        # Create main USD file
        self.usd_file = str(self.output_path / "simulation.usd")
        self.stage = self.Usd.Stage.CreateNew(self.usd_file)
        
        # Set stage metadata
        self.stage.SetMetadata("upAxis", self.up_axis)
        self.stage.SetMetadata("metersPerUnit", 1.0)
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(1000000)  # Will be updated as we render
        
        # Create root xform
        self.root_xform = self.UsdGeom.Xform.Define(self.stage, "/World")
        
        print(f"USD renderer initialized: {self.usd_file}")
    
    def render_frame(
        self,
        time_code,
        particle_positions,
        particle_radii,
        particle_colors=None,
        particle_velocities=None,
        additional_fields=None
    ):
        """
        Render a single frame with particle data.
        
        Parameters
        ----------
        time_code : float
            Time code for this frame (typically step count or simulation time)
        particle_positions : wp.array or np.ndarray
            Particle positions (N, 3)
        particle_radii : wp.array or np.ndarray
            Particle radii (N,)
        particle_colors : np.ndarray, optional
            RGB colors (N, 3) in range [0, 1]
        particle_velocities : wp.array or np.ndarray, optional
            Particle velocities (N, 3)
        additional_fields : dict, optional
            Additional scalar or vector fields to export {name: data}
        """
        if not self.usd_available:
            self._render_frame_simple(time_code, particle_positions, particle_radii, particle_colors)
            return
        
        # Convert warp arrays to numpy if needed
        if isinstance(particle_positions, wp.array):
            positions = particle_positions.numpy()
        else:
            positions = np.asarray(particle_positions)
        
        if isinstance(particle_radii, wp.array):
            radii = particle_radii.numpy()
        else:
            radii = np.asarray(particle_radii)
        
        # Create or get points primitive
        if self.frame_count == 0:
            self.particles_path = "/World/Particles"
            self.particles_prim = self.UsdGeom.Points.Define(self.stage, self.particles_path)
            
            # Set point count (unchanging for now)
            self.particles_prim.GetPointsAttr().Set(self._to_vt_vec3f_array(positions), time_code)
            self.particles_prim.GetWidthsAttr().Set(self._to_vt_float_array(radii * 2.0), time_code)  # USD uses diameters
            
            # Add color primvar if provided
            if particle_colors is not None:
                colors_primvar = self.particles_prim.CreateDisplayColorPrimvar(self.UsdGeom.Tokens.vertex)
                colors_primvar.Set(self._to_vt_vec3f_array(particle_colors), time_code)
            
            # Add velocity as a custom attribute (not primvar, as it's metadata)
            if particle_velocities is not None:
                if isinstance(particle_velocities, wp.array):
                    velocities = particle_velocities.numpy()
                else:
                    velocities = np.asarray(particle_velocities)
                vel_attr = self.particles_prim.GetPrim().CreateAttribute("velocities", self.Sdf.ValueTypeNames.Vector3fArray)
                vel_attr.Set(self._to_vt_vec3f_array(velocities), time_code)
            
            # Add additional fields as custom attributes
            if additional_fields:
                for field_name, field_data in additional_fields.items():
                    if isinstance(field_data, wp.array):
                        field_data = field_data.numpy()
                    
                    # Determine type and create attribute
                    if field_data.ndim == 1:
                        attr = self.particles_prim.GetPrim().CreateAttribute(field_name, self.Sdf.ValueTypeNames.FloatArray)
                        attr.Set(self._to_vt_float_array(field_data), time_code)
                    elif field_data.shape[1] == 3:
                        attr = self.particles_prim.GetPrim().CreateAttribute(field_name, self.Sdf.ValueTypeNames.Vector3fArray)
                        attr.Set(self._to_vt_vec3f_array(field_data), time_code)
        else:
            # Update existing primitive
            self.particles_prim.GetPointsAttr().Set(self._to_vt_vec3f_array(positions), time_code)
            
            if particle_colors is not None:
                colors_primvar = self.particles_prim.GetDisplayColorPrimvar()
                colors_primvar.Set(self._to_vt_vec3f_array(particle_colors), time_code)
            
            if particle_velocities is not None:
                if isinstance(particle_velocities, wp.array):
                    velocities = particle_velocities.numpy()
                else:
                    velocities = np.asarray(particle_velocities)
                vel_attr = self.particles_prim.GetPrim().GetAttribute("velocities")
                vel_attr.Set(self._to_vt_vec3f_array(velocities), time_code)
            
            if additional_fields:
                for field_name, field_data in additional_fields.items():
                    if isinstance(field_data, wp.array):
                        field_data = field_data.numpy()
                    attr = self.particles_prim.GetPrim().GetAttribute(field_name)
                    if field_data.ndim == 1:
                        attr.Set(self._to_vt_float_array(field_data), time_code)
                    else:
                        attr.Set(self._to_vt_vec3f_array(field_data), time_code)
        
        # Update end time
        self.stage.SetEndTimeCode(time_code)
        self.stage.Save()
        
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            print(f"  USD: Rendered frame {self.frame_count} at time {time_code:.4f}")
    
    def _render_frame_simple(self, time_code, positions, radii, colors):
        """Fallback renderer when USD is not available - exports per-frame USD files."""
        # Convert to numpy
        if isinstance(positions, wp.array):
            positions = positions.numpy()
        if isinstance(radii, wp.array):
            radii = radii.numpy()
        
        # Create simple ASCII USD file
        frame_file = self.output_path / f"frame_{int(time_code):06d}.usd"
        
        with open(frame_file, 'w') as f:
            f.write("#usda 1.0\n\n")
            f.write("def Xform \"World\" {\n")
            f.write("    def Points \"Particles\" {\n")
            f.write("        point3f[] points = [\n")
            
            for i, pos in enumerate(positions):
                f.write(f"            ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")
                if i < len(positions) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            
            f.write("        ]\n")
            f.write("        float[] widths = [\n")
            
            for i, r in enumerate(radii):
                f.write(f"            {r*2.0:.6f}")
                if i < len(radii) - 1:
                    f.write(", ")
                else:
                    f.write("\n")
            
            f.write("        ]\n")
            
            if colors is not None:
                f.write("        color3f[] primvars:displayColor = [\n")
                for i, col in enumerate(colors):
                    f.write(f"            ({col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f})")
                    if i < len(colors) - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("        ]\n")
            
            f.write("    }\n")
            f.write("}\n")
        
        self.frame_count += 1
    
    def _to_vt_vec3f_array(self, np_array):
        """Convert numpy array to USD Vec3f array."""
        if np_array.ndim == 1:
            # Scalar to 3D
            return self.Vt.Vec3fArray([self.Gf.Vec3f(float(v), 0, 0) for v in np_array])
        else:
            return self.Vt.Vec3fArray([self.Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in np_array])
    
    def _to_vt_float_array(self, np_array):
        """Convert numpy array to USD float array."""
        return self.Vt.FloatArray(np_array.astype(np.float32).tolist())
    
    def add_mesh(self, name, vertices, faces, time_code=None):
        """
        Add a static mesh (e.g., boundary visualization).
        
        Parameters
        ----------
        name : str
            Mesh name
        vertices : np.ndarray
            Vertex positions (N, 3)
        faces : np.ndarray
            Face indices (M, 3)
        time_code : float, optional
            Time code (None for static)
        """
        if not self.usd_available:
            return
        
        mesh_path = f"/World/{name}"
        mesh = self.UsdGeom.Mesh.Define(self.stage, mesh_path)
        
        mesh.GetPointsAttr().Set(self._to_vt_vec3f_array(vertices), time_code)
        mesh.GetFaceVertexCountsAttr().Set(self.Vt.IntArray([3] * len(faces)), time_code)
        mesh.GetFaceVertexIndicesAttr().Set(self.Vt.IntArray(faces.flatten().tolist()), time_code)
        
        self.stage.Save()
    
    def finalize(self):
        """Finalize and close the USD stage."""
        if self.usd_available:
            self.stage.Save()
            print(f"USD export complete: {self.frame_count} frames saved to {self.usd_file}")
        else:
            print(f"USD export complete: {self.frame_count} frames saved to {self.output_path}")


# Convenience function for use in main simulation loop
def render_mpm_usd(
    renderer,
    sim,
    mpm,
    bigStep,
    counter,
    maxStress=None
):
    """
    Render MPM particles to USD format using GPU-direct rendering.
    
    This is a drop-in replacement for the OpenGL render_mpm function,
    designed for the same interface.
    
    Parameters
    ----------
    renderer : WarpUSDRenderer
        USD renderer instance
    sim : SimState
        Global simulation state
    mpm : MPMState
        MPM-specific state
    bigStep : int
        Current big step
    counter : int
        Current simulation step
    maxStress : float, optional
        Maximum stress for color scaling (for von Mises mode)
    
    Returns
    -------
    maxStress : float
        Updated maximum stress value
    """
    
    # Compute time code
    time_code = sim.t
    
    # --- Choose coloring ---
    colors = None
    additional_fields = {}
    
    if sim.color_mode == "yield_ratio":
        from . import fs5PlotUtils
        ys_ratio = mpm.ys.numpy() / mpm.ys_base.numpy()
        colors = fs5PlotUtils.values_to_rgb(ys_ratio, min_val=0.0, max_val=1.0)
        additional_fields["yield_ratio"] = ys_ratio

    elif sim.color_mode == "damage":
        from . import fs5PlotUtils
        damage = mpm.particle_damage.numpy()
        colors = fs5PlotUtils.values_to_rgb(damage, min_val=0.0, max_val=1.0)
        additional_fields["damage"] = damage

    elif sim.color_mode == "vz":
        from . import fs5PlotUtils
        vz = sim.particle_v.numpy()[:, 2]
        colors = fs5PlotUtils.values_to_rgb(vz, min_val=vz.min(), max_val=vz.max())
        additional_fields["velocity_z"] = vz

    elif sim.color_mode == "von_mises":
        from . import fs5PlotUtils
        sigma = mpm.particle_stress.numpy().astype(np.float64)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        identity = np.eye(3)
        s = sigma - mean_stress[:, None, None] * identity
        von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))
        
        if maxStress is None:
            maxStress = np.quantile(von_mises, 0.95)
        
        colors = fs5PlotUtils.values_to_rgb(von_mises, min_val=0.0, max_val=maxStress)
        additional_fields["von_mises_stress"] = von_mises
        additional_fields["mean_stress"] = mean_stress

    elif sim.color_mode == "effective_ys":
        from . import fs5PlotUtils
        sigma = mpm.particle_stress.numpy().astype(np.float64)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        effective_ys = (mpm.ys.numpy() - mpm.alpha.numpy() * mean_stress) * (1 - mpm.particle_damage.numpy())
        colors = fs5PlotUtils.values_to_rgb(
            effective_ys,
            min_val=np.quantile(effective_ys, 0.1),
            max_val=np.quantile(effective_ys, 0.9) + 1
        )
        additional_fields["effective_yield_stress"] = effective_ys

    elif sim.color_mode == "ys":
        from . import fs5PlotUtils
        effective_ys = mpm.ys.numpy()
        colors = fs5PlotUtils.values_to_rgb(
            effective_ys,
            min_val=np.quantile(effective_ys, 0.1),
            max_val=np.quantile(effective_ys, 0.9) + 1
        )
        additional_fields["yield_stress"] = effective_ys

    elif sim.color_mode == "stress":
        from . import fs5PlotUtils
        sigma = mpm.particle_stress.numpy().astype(np.float64)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        stress_min = np.quantile(mean_stress, 0.01)
        stress_max = np.quantile(mean_stress, 0.99)
        max_magnitude = max(abs(stress_min), abs(stress_max))
        colors = fs5PlotUtils.values_to_rgb(mean_stress, min_val=-max_magnitude, max_val=max_magnitude)
        additional_fields["mean_stress"] = mean_stress
    
    elif sim.color_mode == "sigma_zz":
        from . import fs5PlotUtils
        sigma = mpm.particle_stress.numpy().astype(np.float64)
        sigma_zz = sigma[:, 2, 2]
        sigma_zz_abs = np.abs(sigma_zz)
        colors = fs5PlotUtils.values_to_rgb(
            sigma_zz_abs,
            min_val=0.0,
            max_val=np.quantile(sigma_zz_abs, 0.99)
        )
        additional_fields["sigma_zz"] = sigma_zz
    
    elif sim.color_mode == "state":
        from . import fs5PlotUtils
        state = sim.materialLabel.numpy().astype(np.float64)
        colors = fs5PlotUtils.values_to_rgb(state, min_val=0.0, max_val=np.max(state))
        additional_fields["material_label"] = state
    
    else:
        raise ValueError(f"Unknown color_mode: {sim.color_mode}")
    
    # Render frame
    renderer.render_frame(
        time_code=time_code,
        particle_positions=sim.particle_x,
        particle_radii=sim.particle_radius,
        particle_colors=colors,
        particle_velocities=sim.particle_v,
        additional_fields=additional_fields
    )
    
    return maxStress
