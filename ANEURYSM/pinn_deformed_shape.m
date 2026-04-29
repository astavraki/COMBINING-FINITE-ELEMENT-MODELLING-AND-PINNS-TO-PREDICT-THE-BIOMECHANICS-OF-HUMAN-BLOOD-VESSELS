%% ============================================================
%  PINN visualization — outer surface deformation
%  ============================================================

clear; clc; close all;

%% Load data
data = load('pinn_inference_results.mat');

Xo = data.X_outer;
Yo = data.Y_outer;
Zo = data.Z_outer;

Uxo = data.u1_outer;
Uyo = data.u2_outer;
Uzo = data.u3_outer;

Umag = data.umag_outer;

%% Deformation scaling
scale = 5;

Xd = Xo + scale * Uxo;
Yd = Yo + scale * Uyo;
Zd = Zo + scale * Uzo;

%% Figure 1 — Outer surface
figure('Color','w');

surf( ...
    Xd*1e3, ...
    Yd*1e3, ...
    Zd*1e3, ...
    Umag, ...
    'EdgeColor','none' ...
);

colormap('jet');
colorbar;

title('Outer surface deformation');
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('z (mm)');

axis equal;
view(3);
camlight;
lighting gouraud;
grid on;