inkling "2.0"

using Goal
using Math

# UNIT CONVERSIONS
const MetersPerInch = 0.0254  # 1 inch = 0.0254 meters
const RadiansPerRevolution = 2 * Math.Pi

# CONTROL ACTIONS
const ScrewSpeedMax = (200 / 60) * RadiansPerRevolution  # radians / second
const CutterFrequencyMax = 10  # hertz

const ScrewAccelerationMax = 0.001 * ScrewSpeedMax  # radians / second^2
const CutterAccelerationMax = 0.001 * CutterFrequencyMax  # 1 / second^2

# PRODUCT PARAMETERS
const LengthMultiplier = 12 * MetersPerInch
const LengthTarget = 1 * LengthMultiplier
const LengthTolerance = 0.1 * MetersPerInch  # meters

const InitialTemperature = 190 + 273.15  # Kelvin


type SimState {

    # Angular speed of the screw (radians / second)
    screw_angular_speed: number,

    # Frequency of the cutter (hertz)
    cutter_frequency: number,

    # Length of finished product (meters)
    product_length: number,

    # Volumetric flow rate (i.e. throughput) of extruded material (meters^3 / second)
    flow_rate: number,

    # Temperature (Kelvin)
    temperature: number,

    # Manufacturing yield (dimensionless)
    yield: number,

    # Target length (meters)
    target_length: number

}


type SimAction {

    # Angular acceleration of the screw (radians / second^2)
    screw_angular_acceleration: number <-ScrewAccelerationMax .. ScrewAccelerationMax>,

    # Change in frequency of the cutter (1 / seconds^2)
    cutter_acceleration: number <-CutterAccelerationMax .. CutterAccelerationMax>,

}


type SimConfig {

    initial_screw_angular_speed: number <(5 * RadiansPerRevolution / 60) .. ScrewSpeedMax>,
    initial_cutter_frequency: number <0 .. CutterFrequencyMax>,

    initial_screw_angular_acceleration: number <-ScrewAccelerationMax .. ScrewAccelerationMax>,
    initial_cutter_acceleration: number <-CutterAccelerationMax .. CutterAccelerationMax>,

    initial_temperature: InitialTemperature,

    target_length: number <0.5 * LengthMultiplier, 1 * LengthMultiplier, 2 * LengthMultiplier>,

}


simulator ExtrusionSim (Action: SimAction, Config: SimConfig): SimState {
    # Automatically launch the simulator with this registered package name.
    package "PVC_Extruder_Variable"
}


graph (input: SimState): SimAction {

    concept MaxYieldDriveLength(input): SimAction {
        curriculum {

            source ExtrusionSim

            goal (State: SimState) {

                maximize Yield:
                    State.yield in Goal.RangeAbove(0)
                
                drive Length:
                    State.product_length
                    in Goal.Range(State.target_length - LengthTolerance, State.target_length + LengthTolerance)
                    within 120

            }

            training {
                EpisodeIterationLimit: 200,  # default is 1,000
                NoProgressIterationLimit: 1000000
            }

            lesson RandomizeStartWide {

                scenario {
                    initial_screw_angular_speed: number <(20 * RadiansPerRevolution / 60) .. (80 * RadiansPerRevolution / 60)>,
                    initial_cutter_frequency: number <0.06 .. 0.40>,

                    initial_screw_angular_acceleration: number <-ScrewAccelerationMax .. ScrewAccelerationMax>,
                    initial_cutter_acceleration: number <-CutterAccelerationMax .. CutterAccelerationMax>,                
                    initial_temperature: InitialTemperature,
                    target_length: LengthTarget
                }
            }
        }
    }
}
