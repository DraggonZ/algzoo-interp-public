"""Density-weighted centroids of the two tie planes (third-largest u > 0): the 2nd-3rd tie (m2 = m3 = u, m1 = u + b) with density phi(m1) phi(u)^2 Phi(u)^7, and the 1st-2nd tie (m1 = m2 = u + a, m3 = u) with phi(m1)^2 phi(u) Phi(u)^7. Used to place the nodes (bandlib.py, cheap_est.py)."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad
import common as c
log = c.Log("centroids.log")
bg = lambda m: -norm.pdf(m)/norm.cdf(m)
# a-tie density f(u+b,u,u) over u>0,b>0 ; b-tie density f(u+a,u+a,u) over u>0? 
for nm, f, m1, m2 in [("a-tie", lambda b,u: norm.pdf(u+b)*norm.pdf(u)**2*norm.cdf(u)**7, lambda u,g:u+g, lambda u,g:u),
                      ("b-tie", lambda a,u: norm.pdf(u+a)**2*norm.pdf(u)*norm.cdf(u)**7, lambda u,g:u+g, lambda u,g:u+g)]:
    Z = dblquad(f, 0, 5, 0, 6)[0]
    E = lambda h: dblquad(lambda g,u: f(g,u)*h(u,g), 0, 5, 0, 6)[0]/Z
    log(nm, "mass", Z, "E m1", E(m1), "E m2", E(m2), "E m3", E(lambda u,g:u), "E bg", E(lambda u,g: bg(u)), "bg(E u)", bg(E(lambda u,g:u)))
