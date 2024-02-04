[X1,X2] = meshgrid(-5:0.5:5);
xs = arrayfun(@(x,y) {odeFun([],[x,y])}, X1, X2);
x1s = cellfun(@(x) x(1), xs);
x2s = cellfun(@(x) x(2), xs);
streopppphppsf(x1s, x2s)
xlabel('x_1')
ylabel('x_2')
axis tight equal;
c = 10;
a = 10;
function dxdt = odeFun(t,x)
    dxdt(1) = x(2);
    dxdt(2) = -sin(x(1)) - 1 * x(2) * (x(2)^2 / 2 - cos(x(1)) + 0.5);
end