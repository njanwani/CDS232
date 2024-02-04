T = 3;

f = @(t, x) -x + atan(x);
t0 = 0;
x0 = 1;
solver = @ode78;
fsol = solver(f, [t0, T], x0);

g = @(t, x) -x + pi / 2;
t0 = 0;
x0 = 1;
solver = @ode45;
gsol = solver(g, [t0, T], x0);
hold on
plot(fsol.x, fsol.y, color = 'red');
plot(gsol.x, gsol.y, color = 'blue');
hold off
xlabel('x');
ylabel('y');
legend();
title('Numerical Solutions');