%% ============================================================
%  PINN visualization - deformed shape
%  ============================================================

% Load inference results
load('pinn_inference_results.mat');

% Coordinates
x = nodes(:,1);
y = nodes(:,2);
z = nodes(:,3);

% Displacement components
u1 = u1(:);
u2 = u2(:);
u3 = u3(:);

% Deformation scaling (for visualization)
scale = 1000;

% Deformed coordinates
xd = x + scale * u1;
yd = y + scale * u2;
zd = z + scale * u3;

% Displacement magnitude
umag = sqrt(u1.^2 + u2.^2 + u3.^2);

%% ============================================================
%  Figure - Deformed shape (outer surface)
%  ============================================================

figure;
scatter3(xd*1e3, yd*1e3, zd*1e3, 10, umag*1e6, 'filled');

xlabel('x1 (mm)');
ylabel('x2 (mm)');
zlabel('z (mm)');

title('Deformed shape (scaled)');

colorbar;
colormap jet;
cb = colorbar;
ylabel(cb, '|u| (\mum)');

axis equal;
grid on;
view(45, 25);