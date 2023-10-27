import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

input_intensity_A = int(input("Traffic intensity A: "))
input_intensity_B = int(input("Traffic intensity B: "))

traffic_intensity_range = np.arange(0, 100, 1)
lights_duration = np.arange(0, 60, 1)

traffic_low = mf.trapmf(traffic_intensity_range, [0, 0, 20, 60])
traffic_medium = mf.trapmf(traffic_intensity_range, [10, 50, 50, 90])
traffic_high = mf.trapmf(traffic_intensity_range, [40, 80, 100, 100])

lights_short = mf.trapmf(lights_duration, [0, 0, 0, 20])
lights_medium = mf.trapmf(lights_duration, [15, 30, 30, 45])
lights_long = mf.trapmf(lights_duration, [40, 60, 60, 60])

plt.plot(traffic_intensity_range, traffic_low)
plt.plot(traffic_intensity_range, traffic_medium)
plt.plot(traffic_intensity_range, traffic_high)
plt.title("Traffic intensity")
# plt.show()

plt.plot(lights_duration, lights_short)
plt.plot(lights_duration, lights_medium)
plt.plot(lights_duration, lights_long)
plt.title("Lights duration")
# plt.show()

# fuzzification

traffic_A_low_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_low,
    input_intensity_A
)

traffic_A_medium_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_medium,
    input_intensity_A
)

traffic_A_high_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_high,
    input_intensity_A
)

traffic_B_low_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_low,
    input_intensity_B
)

traffic_B_medium_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_medium,
    input_intensity_B
)

traffic_B_high_given_input = fuzz.interp_membership(
    traffic_intensity_range,
    traffic_high,
    input_intensity_B
)

# rules

# rule 1 - if traffic A low and traffic B high then lights A long and lights B short
rule1 = np.fmin(traffic_A_low_given_input, traffic_B_high_given_input)

rule1_implication_A = np.fmin(rule1, lights_long)
rule1_implication_B = np.fmin(rule1, lights_short)

plt.plot(lights_duration, rule1_implication_A)
plt.title("Rule 1 A implication")
# plt.show()

plt.plot(lights_duration, rule1_implication_B)
plt.title("Rule 1 B implication")
# plt.show()

# rule 2 - if traffic B low and traffic A high then lights B long and lights A short
rule2 = np.fmin(traffic_B_low_given_input, traffic_A_high_given_input)

rule2_implication_A = np.fmin(rule2, lights_short)
rule2_implication_B = np.fmin(rule2, lights_long)

plt.plot(lights_duration, rule2_implication_A)
plt.title("Rule 2 A implication")
# plt.show()

plt.plot(lights_duration, rule2_implication_B)
plt.title("Rule 2 B implication")
# plt.show()

# rule 3 - if traffic A medium and traffic B medium then lights A medium and lights B medium
rule3 = np.fmin(traffic_A_medium_given_input, traffic_B_medium_given_input)

rule3_implication_A = np.fmin(rule3, lights_medium)
rule3_implication_B = np.fmin(rule3, lights_medium)

plt.plot(lights_duration, rule3_implication_A)
plt.title("Rule 3 A implication")
# plt.show()

plt.plot(lights_duration, rule3_implication_B)
plt.title("Rule 3 B implication")
# plt.show()

# rule 4 - if traffic A low and traffic B low then lights A medium and lights B medium
rule4 = np.fmin(traffic_A_low_given_input, traffic_B_low_given_input)

rule4_implication_A = np.fmin(rule4, lights_medium)
rule4_implication_B = np.fmin(rule4, lights_medium)

plt.plot(lights_duration, rule4_implication_A)
plt.title("Rule 4 A implication")
# plt.show()

plt.plot(lights_duration, rule4_implication_B)
plt.title("Rule 4 B implication")
# plt.show()

# rule 5 - if traffic A high and traffic B high then lights A medium and lights B medium
rule5 = np.fmin(traffic_A_high_given_input, traffic_B_high_given_input)

rule5_implication_A = np.fmin(rule5, lights_medium)
rule5_implication_B = np.fmin(rule5, lights_medium)

plt.plot(lights_duration, rule5_implication_A)
plt.title("Rule 5 A implication")
# plt.show()

plt.plot(lights_duration, rule5_implication_B)
plt.title("Rule 5 B implication")
# plt.show()

# composition for lights A

composition_A = np.fmax(np.fmax(np.fmax(np.fmax(rule1_implication_A, rule2_implication_A), rule3_implication_A), rule4_implication_A), rule5_implication_A)

plt.plot(lights_duration, composition_A)
plt.title("Composition A")
# plt.show()

# composition for lights B

composition_B = np.fmax(np.fmax(np.fmax(np.fmax(rule1_implication_B, rule2_implication_B), rule3_implication_B), rule4_implication_B), rule5_implication_B)

plt.plot(lights_duration, composition_B)
plt.title("Composition B")
# plt.show()

# defuzzifying result for lights A

defuzzified_A = fuzz.defuzz(lights_duration, composition_A, 'centroid')

# defuzzifying result for lights B

defuzzified_B = fuzz.defuzz(lights_duration, composition_B, 'centroid')

print("Lights A duration: ", defuzzified_A)
print("Lights B duration: ", defuzzified_B)
