# ==========================================
# 1. Imports
# ==========================================
import math
import matplotlib.pyplot as plt
import orekit
vm = orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.propagation import Propagator

setup_orekit_curdir("/Users/beaveracosta/orekit-data")

from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.frames import FramesFactory, LOFType
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import ApsideDetector, DateDetector
from org.orekit.propagation.events.handlers import StopOnEvent
from org.orekit.forces.maneuvers import ImpulseManeuver
from org.orekit.attitudes import LofOffset
from org.hipparchus.geometry.euclidean.threed import Vector3D

from org.orekit.propagation.sampling import PythonOrekitFixedStepHandler

# ==========================================
# 2. Step Handler
# ==========================================
class TrajectoryCollector(PythonOrekitFixedStepHandler):
    """Samples and stores spacecraft 2D position (in km) at fixed intervals."""
    def __init__(self):
        super().__init__()
        self.x = []
        self.y = []

    def init(self, s0, t, step):
        pass

    def handleStep(self, currentState):
        # Extract inertial position in meters and convert to kilometers
        pos = currentState.getPVCoordinates().getPosition()
        self.x.append(pos.getX() / 1000.0)
        self.y.append(pos.getY() / 1000.0)

    def finish(self, *args):
        pass


def demonstrate_event_driven_hohmann():
    # ==========================================
    # 3. Create the Initial Orbit
    # ==========================================
    # --- System Constants ---
    mu = Constants.EIGEN5C_EARTH_MU
    earth_radius = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    
    # --- Define Initial Elliptical Orbit ---
    r_perigee = earth_radius + 400e3
    r_apogee = earth_radius + 2000e4
    a_init = (r_perigee + r_apogee) / 2.0
    e_init = (r_apogee - r_perigee) / (r_apogee + r_perigee)
    
    inertial_frame = FramesFactory.getEME2000()
    utc = TimeScalesFactory.getUTC()
    initial_date = AbsoluteDate(2026, 7, 23, 23, 54, 30.0, utc)
    
    initial_orbit = KeplerianOrbit(a_init, e_init, 0.0, 0.0, 0.0, 0.0,
                                   PositionAngleType.TRUE, 
                                   inertial_frame, initial_date, mu)
    
    print("--- Initial Orbit ---")
    print(f"Semi-major axis: {initial_orbit.getA() / 1000:.2f} km")
    print(f"Eccentricity:    {initial_orbit.getE():.4f}")

    # ==========================================
    # 4. Calculate Maneuvers
    # ==========================================
    r_target = earth_radius + 6000e3
    a_transfer = (r_apogee + r_target) / 2.0

    # Calculate when initial apogee is reached
    initial_period = initial_orbit.getKeplerianPeriod()
    time_to_apogee = initial_period / 2
    t1 = initial_date.shiftedBy(time_to_apogee)

    # Burn 1 Delta-V (at initial apogee)
    v_i1 = math.sqrt(mu * (2.0 / r_apogee - 1.0 / a_init))
    v_t1 = math.sqrt(mu * (2.0 / r_apogee - 1.0 / a_transfer))
    delta_v1 = v_t1 - v_i1

    # Burn 2 Delta-V (at transfer apogee)
    v_t2 = math.sqrt(mu * (2.0 / r_target - 1.0 / a_transfer))
    v_circ = math.sqrt(mu / r_target)
    delta_v2 = v_circ - v_t2

    # Time of flight for the transfer orbit (half a period)
    transfer_period = 2.0 * math.pi * math.sqrt((a_transfer ** 3) / mu)
    t_tof = transfer_period / 2.0
    t2 = t1.shiftedBy(t_tof)

    print(f'Perigee-raising burn will occur at: {t1}')
    print(f"\nCircularizing burn will occur at: {t2}")

    # ==========================================
    # 5. Create the Propagator
    # ==========================================
    main_prop = KeplerianPropagator(initial_orbit)

    collector = TrajectoryCollector()
    Propagator.cast_(main_prop).setStepHandler(30.0, collector)  
    
    # Align attitude X-axis with velocity (TNW frame: Tangent, Normal, out-of-plane W)
    attitude_provider = LofOffset(inertial_frame, LOFType.TNW)
    main_prop.setAttitudeProvider(attitude_provider)
    
    # Define Maneuver 1 (Transfer Injection)
    # Vector3D(x, y, z) -> Applying strictly along the Tangent (velocity) axis
    dv1_vec = Vector3D(delta_v1, 0.0, 0.0) 
    maneuver1 = ImpulseManeuver(DateDetector(t1), attitude_provider, dv1_vec, 300.0)
    main_prop.addEventDetector(maneuver1)
    
    # Define Maneuver 2 (Circularization)
    dv2_vec = Vector3D(delta_v2, 0.0, 0.0)
    maneuver2 = ImpulseManeuver(DateDetector(t2), attitude_provider, dv2_vec, 300.0)
    main_prop.addEventDetector(maneuver2)

    # ==========================================
    # 6. Propagate the Orbits
    # ==========================================
    # Get position at Burn 1 (Initial Apogee)
    pv_burn1 = main_prop.propagate(t1).getPVCoordinates().getPosition()
    x_burn1, y_burn1 = pv_burn1.getX() / 1000.0, pv_burn1.getY() / 1000.0

    # Get position at Burn 2 (Transfer Apogee)
    final_state = main_prop.propagate(t2)
    pv_burn2 = final_state.getPVCoordinates().getPosition()
    x_burn2, y_burn2 = pv_burn2.getX() / 1000.0, pv_burn2.getY() / 1000.0

    final_orbit = final_state.getOrbit()
    final_period = final_orbit.getKeplerianPeriod()
    more_len = main_prop.propagate(t2.shiftedBy(final_period)).getPVCoordinates().getPosition()

    # ==========================================
    # 7. Plotting the Trajectory
    # ==========================================
    plt.figure(figsize=(9, 9))

    # Draw Earth
    earth_radius_km = Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1000.0
    earth_circle = plt.Circle((0, 0), earth_radius_km, color='royalblue', alpha=0.3, label='Earth')
    plt.gca().add_patch(earth_circle)

    # Plot Trajectory Path
    plt.plot(collector.x, collector.y, color='darkorange', linewidth=1.5, label='Spacecraft Path')

    # Mark Burn 1
    plt.scatter(x_burn1, y_burn1, color='red', marker='^', s=120, zorder=5, label='Burn 1 (Transfer Injection)')
    plt.annotate(f'Burn 1\nΔV1 = {delta_v1:.1f} m/s', 
                xy=(x_burn1, y_burn1), 
                xytext=(x_burn1 + 500, y_burn1 + 500),
                # arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight='bold', color='crimson')

    # Mark Burn 2
    plt.scatter(x_burn2, y_burn2, color='green', marker='^', s=120, zorder=5, label='Burn 2 (Circularization)')
    plt.annotate(f'Burn 2\nΔV2 = {delta_v2:.1f} m/s', 
                xy=(x_burn2, y_burn2), 
                xytext=(x_burn2 - 2500, y_burn2 - 1500),
                # arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight='bold', color='darkgreen')

    # Formatting
    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)
    plt.title('Hohmann Transfer Trajectory with Burn Markers', fontsize=12, fontweight='bold')
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demonstrate_event_driven_hohmann()
