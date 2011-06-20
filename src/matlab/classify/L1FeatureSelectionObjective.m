function [J diff] = L1FeatureSelectionObjective(v, zn, lambda)
%for fixed zn, line search along eta * v.

J = sum(log(1 + exp(-(v.^2).' * zn))) + lambda * norm(v)^2; %eqn. 9 from paper

diff = (lambda - zn * (exp(v.^2.' * zn) ./ (1 + exp(v.^2.' * zn))).') .* v; %eqn. 10 from paper

if(isnan(J))
    error('Garbage values in objective function.');
end