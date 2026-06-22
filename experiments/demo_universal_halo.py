from neurooptai import UniversalHALO

halo = UniversalHALO()

print("simple:", halo.should_intervene(3, 10))
print("score:", halo.score(3, 10))
print("class:", halo.classify(3, 10))
print("expected:", halo.expected_avoided_cost(0.8, 100))
print("probabilistic:", halo.should_intervene_probabilistic(20, 0.8, 100))
