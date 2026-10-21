"""Add the error-rate estimate per trained network (bias_formula_v4.json: [measured, predicted, cost]) to viz/model.json."""
import json

m = json.load(open("viz/model.json"))
e = json.load(open("bias_formula_v4.json"))
m["estimator"] = [[k, round(v[0], 5), round(v[1], 5)] for k, v in e.items()]
json.dump(m, open("viz/model.json", "w"))
print(len(m["estimator"]), "networks;", m["estimator"][0])
