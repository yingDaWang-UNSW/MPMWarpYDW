# Domain and Config Files - Fixes Applied

## Summary of Issues Fixed

### 1. HDF5 Field Names ❌→✅

**Problem**: Domain generation script used incorrect field names that don't match `runMPMYDW.py` expectations.

**Fixed in `createRandomRockDomain.py`**:
- ❌ `'position'` → ✅ `'x'` (transposed to shape `(3, n_particles)`)
- ❌ `'volume'` → ✅ `'particle_volume'`

**Why**: `runMPMYDW.py` line 46-47 expects:
```python
x = np.array(h5file["x"]).T
particle_volume = np.array(h5file["particle_volume"])
```

### 2. JSON Config Parameters ❌→✅

**Problem**: `config_random_rock.json` contained invalid parameters not defined in `getArgs.py`.

**Removed invalid parameters**:
- ❌ `"xpbd_substeps"` - not defined
- ❌ `"xpbd_k_contact"` - not defined
- ❌ `"xpbd_k_volume"` - not defined
- ❌ `"xpbd_contactDistance"` - not defined
- ❌ `"xpbd_enable_volume"` - not defined

**Added missing required parameters**:
- ✅ `"outputFolder"` - output directory path
- ✅ `"color_mode"` - rendering color mode
- ✅ `"K0"` - lateral earth pressure coefficient
- ✅ `"xpbd_relaxation"` - XPBD relaxation factor
- ✅ `"dynamicParticleFriction"` - dynamic friction
- ✅ `"staticVelocityThreshold"` - static velocity threshold
- ✅ `"staticParticleFriction"` - static friction
- ✅ `"particle_cohesion"` - particle cohesion
- ✅ `"sleepThreshold"` - sleep threshold
- ✅ `"swellingRatio"` - swelling ratio
- ✅ `"swellingActivationFactor"` - swelling activation
- ✅ `"swellingMaxFactor"` - swelling max
- ✅ `"particle_v_max"` - max particle velocity

**Fixed type issues**:
- ❌ `"render": true` → ✅ `"render": 1`
- ❌ `"saveFlag": true` → ✅ `"saveFlag": 1`

**Fixed path format**:
- ❌ `"domainFile": "benchmarks/..."` → ✅ `"./benchmarks/..."`

### 3. Visualization Script ❌→✅

**Fixed in `visualizeRandomRockDomain.py`**:
- ❌ `f['position'][:]` → ✅ `f['x'][:].T` (transpose back to `(n_particles, 3)`)

## Verification

Run `verify_hdf5_fields.py` to check that all required fields are present:

```bash
python benchmarks/cavingBenchmark/verify_hdf5_fields.py
```

Expected output:
```
✓ REQUIRED FIELDS (from runMPMYDW.py):
  ✓ x                    shape=(3, 522801)
  ✓ particle_volume      shape=(522801,)
✓ OPTIONAL SPATIAL PROPERTY FIELDS:
  ✓ density, E, nu, ys, alpha, hardening, softening, eta_shear, eta_bulk, strainCriteria
```

## Files Updated

1. **`createRandomRockDomain.py`** - Fixed HDF5 field names
2. **`config_random_rock.json`** - Fixed parameter names and added missing fields
3. **`visualizeRandomRockDomain.py`** - Fixed position field reading
4. **`random_rock_domain_50x50x200.h5`** - Regenerated with correct field names

## Ready to Run

The benchmark is now properly configured and can be run with:

```bash
cd d:\sourceCodes\MPMWarpYDW
python runMPMYDW.py --config benchmarks/cavingBenchmark/config_random_rock.json
```

All field names now match the expectations in:
- `utils/getArgs.py` (JSON parameter definitions)
- `runMPMYDW.py` (HDF5 field loading)
