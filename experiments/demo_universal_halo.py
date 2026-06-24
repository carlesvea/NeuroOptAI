from neurooptai import UniversalHALO

halo = UniversalHALO()

print(halo.decision(
    intervention_cost=3,
    avoided_cost=10,
    probability_of_failure=0.8,
))
