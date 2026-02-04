# PLANNING AGENT: GPS-Based Correlation Synchronizer Redesign

## Mission Brief
Redesign `pils/synchronizer.py` to implement correlation-based **TIME synchronization** with GPS payload as the single source of truth. Replace current interpolation-based approach with cross-correlation TIME offset detection using:
- **GPS sources**: NED position signals (3D) for time offset detection
- **Inclinometer**: Pitch angle signal (1D) for time offset detection (vs litchi gimbal pitch)

All synchronization outputs are TIME OFFSETS to align data to GPS payload timebase.

---

## Codebase Analysis

### Current Implementation (`pils/synchronizer.py`)
**Architecture**: Generic time-series synchronizer
- **Pattern**: Add sources → interpolate to common timebase → merge DataFrames
- **Method**: Linear interpolation at target sample rate
- **Weaknesses**: 
  - Assumes timestamps are already aligned
  - No offset detection capability
  - No hierarchical reference (all sources equal)
  - Generic approach not optimized for GPS/IMU sensor fusion

### User's Working Example (`loader.ipynb`)
**Key Implementation**: Complete GPS offset finder with:
1. `lla_to_enu()` - LLA to local ENU coordinate conversion
2. `find_subsample_peak()` - Parabolic interpolation for sub-sample precision
3. `find_gps_offset()` - Full GPS cross-correlation pipeline:
   - Convert both GPS sources to ENU relative to reference point
   - Interpolate to common high-rate timebase
   - Cross-correlate E, N, U axes independently
   - Weighted average of offsets by correlation strength
   - Compute spatial offset after time alignment
4. Angular correlation concept shown for inclinometer vs litchi gimbal pitch

### Integration Points (`pils/flight.py`)
- `Flight` class stores `raw_data` with `drone_data` (drone/litchi) and `payload_data` (gps, imu, inclinometer, adc)
- No current synchronization integration in Flight class
- Need to add method to trigger new synchronizer

---

## Architecture Design

### Core Concept: Hierarchical Reference-Based TIME Synchronization

```
GPS Payload (Reference Time t_ref)
     ↓
     ├─→ Drone GPS (TIME offset via NED position correlation)
     ├─→ Litchi GPS (TIME offset via NED position correlation)
     └─→ Inclinometer (TIME offset via pitch angle correlation with Litchi gimbal)
```

**All outputs are TIME OFFSETS in seconds to align data to GPS payload timebase.**

### Class Structure

```python
class CorrelationSynchronizer:
    """GPS-based correlation synchronizer with hierarchical reference."""
    
    # Core Data
    gps_payload: Optional[pl.DataFrame]  # Single source of truth
    drone_gps: Optional[pl.DataFrame]    # Optional drone GPS
    litchi_gps: Optional[pl.DataFrame]   # Optional litchi GPS  
    inclinometer: Optional[pl.DataFrame] # Optional inclinometer
    other_payload: Dict[str, pl.DataFrame]  # Other payload sensors (adc, etc.)
    
    # Metadata
    offsets: Dict[str, Dict[str, Any]]  # Detected offsets per source
    
    # Methods
    add_gps_reference()      # Set GPS payload as reference
    add_drone_gps()          # Add drone GPS for correlation
    add_litchi_gps()         # Add litchi GPS for correlation
    add_inclinometer()       # Add inclinometer for angular correlation
    add_payload_sensor()     # Add other payload sensors (simple alignment)
    
    # Correlation functions
    _lla_to_enu()            # Coordinate conversion
    _find_subsample_peak()   # Sub-sample peak detection
    _find_gps_offset()       # GPS TIME offset via NED correlation (3D signal)
    _find_pitch_offset()     # Inclinometer TIME offset via pitch correlation (1D signal)
    
    # Main sync
    synchronize()            # Execute correlation-based sync
    get_offset_summary()     # Return offset metadata
    save_synchronized()      # Export results
```

---

## Implementation Phases

### Phase 1: Core Correlation Functions (30 min)
**Goal**: Implement coordinate conversion and correlation utilities

**Tasks**:
1. Implement `_lla_to_enu()` for LLA → ENU conversion
2. Implement `_find_subsample_peak()` for parabolic interpolation
3. Add comprehensive docstrings and type hints

**Tests**:
- Test ENU conversion against known values
- Test sub-sample peak with synthetic data
- Verify numerical stability

**Deliverables**:
- Working utility functions with tests
- Ready for GPS offset detection

---

### Phase 2: GPS Offset Detection (45 min)
**Goal**: Implement complete GPS cross-correlation pipeline

**Tasks**:
1. Implement `_find_gps_offset()` following user's working example:
   - Convert both GPS sources to ENU
   - Interpolate to common timebase
   - Cross-correlate E, N, U axes
   - Weighted average by correlation strength
   - Compute spatial offsets
2. Handle edge cases (no overlap, insufficient data)
3. Return rich metadata (offsets per axis, correlations, etc.)

**Tests**:
- Test with real flight data from user's example
- Test with misaligned GPS (known offset)
- Test edge cases (short overlap, noisy data)

**Deliverables**:
- Robust GPS offset finder
- Comprehensive offset metadata

---

### Phase 3: Inclinometer TIME Synchronization (30 min)
**Goal**: Implement inclinometer TIME offset detection via pitch angle correlation

**IMPORTANT**: This is TIME synchronization, not angular offset detection.

**Tasks**:
1. Implement `_find_pitch_offset()` following GPS pattern:
   - Extract litchi gimbal pitch as reference signal (1D)
   - Extract inclinometer pitch as target signal (1D)
   - Interpolate both to common high-rate timebase
   - Cross-correlate pitch signals
   - Sub-sample peak detection
   - **Return TIME offset in seconds** (not angular offset)
2. Handle missing litchi data gracefully
3. Optional: Support roll/yaw as fallback signals if pitch unavailable

**Method Signature**:
```python
def _find_pitch_offset(
    time1: np.ndarray,  # Litchi timestamps
    pitch1: np.ndarray, # Litchi gimbal pitch angles
    time2: np.ndarray,  # Inclinometer timestamps
    pitch2: np.ndarray, # Inclinometer pitch angles
) -> Dict[str, Any]:
    """Find TIME offset between pitch signals using cross-correlation.
    
    Returns:
    --------
    dict with 'time_offset' (in seconds), 'correlation', etc.
    """
```

**Tests**:
- Test with real inclinometer + litchi data
- Test with synthetic pitch data (known time offset)
- Test missing litchi scenario
- Verify returns TIME offset, not angle

**Deliverables**:
- Pitch-based TIME offset detection
- Fallback strategies

---

### Phase 4: Synchronizer Class Redesign (60 min)
**Goal**: Complete class implementation with hierarchical sync

**Tasks**:
1. Redesign `CorrelationSynchronizer` class:
   - GPS payload as mandatory reference
   - Optional drone/litchi GPS sources
   - Optional inclinometer source
   - Generic payload sensors (adc, etc.)
2. Implement `add_*` methods for each source type
3. Implement main `synchronize()` method:
   - Detect offsets for all sources
   - Apply corrections
   - Interpolate to GPS payload timebase
   - Merge into single DataFrame
4. Store offset metadata in `self.offsets`

**Tests**:
- Integration test with full flight data
- Test partial data scenarios (e.g., no drone GPS)
- Verify offset metadata storage
- Compare results with user's working example

**Deliverables**:
- Complete synchronizer class
- Offset metadata structure

---

### Phase 5: Flight Class Integration (30 min)
**Goal**: Integrate new synchronizer into Flight API

**Tasks**:
1. Update `Flight.synchronize()` or add new method
2. Auto-detect available data sources from `raw_data`
3. Create user-friendly API:
   ```python
   flight.synchronize_with_gps(target_rate_hz=10.0)
   ```
4. Preserve backward compatibility if possible

**Tests**:
- Test Flight integration with real data
- Test API convenience methods
- Test error messages for missing GPS payload

**Deliverables**:
- Flight class integration
- User-friendly API

---

### Phase 6: Documentation & Examples (20 min)
**Goal**: Document new synchronizer with usage examples

**Tasks**:
1. Update synchronizer docstrings
2. Create usage example in docstring
3. Update README.md with new sync approach
4. Document offset metadata structure

**Tests**:
- Verify docstrings render correctly
- Test example code runs

**Deliverables**:
- Complete documentation
- Working examples

---

## Risk Assessment

### High Risk
- **Coordinate conversion accuracy** - ENU conversion must be numerically stable
  - Mitigation: Test against known transformations, use established formulas

- **Correlation robustness** - Noisy data may yield poor correlation peaks
  - Mitigation: Implement quality checks, require minimum correlation threshold
  
- **Signal type confusion** - All sync is TIME-based (not angular offsets)
  - Mitigation: Clear function naming, comprehensive docstrings, explicit return types

### Medium Risk
- **Missing data sources** - Not all flights have litchi or inclinometer
  - Mitigation: Graceful fallbacks, clear error messages

- **API breaking changes** - Existing code may depend on old synchronizer
  - Mitigation: Consider backward compatibility, version bump if needed

### Low Risk
- **Performance** - Correlation operations are computationally expensive
  - Mitigation: NumPy/SciPy are highly optimized, acceptable for batch processing

---

## Dependencies

### Required Packages (already in pyproject.toml)
- `numpy>=1.23.0` ✓
- `polars>=0.20.0` ✓
- `scipy` - **MISSING** - need for `signal.correlate`, `interpolate.interp1d`

### New Dependency
Must add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing
    "scipy>=1.9.0",
]
```

---

## Testing Strategy

### Unit Tests
- Test individual correlation functions
- Test coordinate conversions
- Test sub-sample peak detection

### Integration Tests
- Test with real flight data from user's example
- Test partial data scenarios
- Test edge cases

### Validation
- Compare results with user's working notebook
- Verify offset detection accuracy
- Check offset metadata completeness

---

## Success Criteria

✅ GPS payload is mandatory reference timebase  
✅ NED correlation finds TIME offsets for drone/litchi GPS  
✅ Pitch correlation finds TIME offset for inclinometer  
✅ **All offsets are TIME-based** (seconds, not angles or positions)  
✅ Graceful handling of missing sources  
✅ Offset metadata stored and accessible  
✅ Flight class integration works  
✅ Tests pass with real flight data  
✅ Documentation complete  

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Core functions | 30 min | 30 min |
| Phase 2: GPS offset | 45 min | 75 min |
| Phase 3: Angular correlation | 30 min | 105 min |
| Phase 4: Synchronizer class | 60 min | 165 min |
| Phase 5: Flight integration | 30 min | 195 min |
| Phase 6: Documentation | 20 min | 215 min |

**Total: ~3.5 hours**

---

## Next Steps

1. **User Review**: Approve this plan or request changes
2. **Dependency Check**: Add scipy to pyproject.toml
3. **Begin Phase 1**: Implement core correlation functions with TDD
4. **Iterate**: Each phase → implement → review → commit

---

**Planning Agent Status**: PLAN COMPLETE ✓
**Awaiting User Approval**: Please review and approve to proceed with implementation.
