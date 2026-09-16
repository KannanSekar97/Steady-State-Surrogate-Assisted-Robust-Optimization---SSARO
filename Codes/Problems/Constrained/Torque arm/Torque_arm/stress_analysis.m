function [smax,cr_a, umax] = stress_analysis(s,forceX,forceY,draw,k)
if nargin==3
    draw = 0;
end
[Edof, Ex, Ey, numfenod, numfele] = rodDesign_k(s,0);

%Set up global matrix and vector
K = zeros(numfenod*2);
F = zeros(numfenod*2,1);

%Material properties and problem definition
E = 206.8E5;  %Young's modulus
% E=300000;
nu = 0.29;  %Poisson's ratio
thickness = 0.3; %Thickness
type = 1;  %1: plane stress, 2: plane strain
NG = 2;   %No. of Gauss integration points

ep = [type thickness NG];
if(type==1)
    cf = E/(1-nu^2);
    D = [cf cf*nu 0; cf*nu cf 0; 0 0 cf*(1-nu)/2];
elseif (type==2)
    cf = E/((1+nu)*(1-2*nu));
    D = [(1-nu)*cf cf*nu 0; cf*nu (1-nu)*cf 0; 0 0 cf*(1-2*nu)/2];
end

%Assembly
detJ = zeros(numfele,1);
for i=1:numfele
    [Ke,~,detJ(i,1),count] = Kplani4e(Ex(i,:),Ey(i,:),ep,D);
    K = assem(Edof(i,:),K,Ke);
end

% using Jacobian to compute the area of each element
% if any(detJ)<0
%     disp('Jacobideterminant less than zero!')
%     cr_a = 0;
% else
%     cr_a = 4*sum(detJ);
% end
cr_a = 4*sum(detJ);

%Boundary condition: Fixed left hole
bc = [  1 0;  2 0;  3 0;  4 0;  5 0;  6 0;  7 0;  8 0;  9 0; 10 0; 11 0; 12 0;
    13 0; 14 0; 15 0; 16 0; 17 0; 18 0; 19 0; 20 0; 21 0; 22 0;133 0;134 0;
    135 0;136 0;137 0;138 0;139 0;140 0;141 0;142 0;313 0;314 0;315 0;316 0;
    317 0;318 0;319 0;320 0];

%Applied loads: Fx = -2789N and Fy = 5066N at the right hole
fx = forceX/16;
fy = forceY/16;

%New model
F(611)=fx;
F(661:2:669) = fx;
F(721:2:729) = fx;
F(829:2:845) = fx;
F(612)=fy;
F(662:2:670) = fy;
F(722:2:730) = fy;
F(830:2:846) = fy;

%Solve the finite element matrix equation
U = solveq(K,F,bc);
Ux = U(1:2:end);
Uy = U(2:2:end);

Umag = sqrt(Ux.^2 + Uy.^2);

umax_cm = max(Umag);
umax = umax_cm * 10;  % In mm

%Extract element displacements
Ed = extract(Edof,U);
es = zeros(numfele,4,3);

%Calculate element stress at each nodes
for i=1:numfele
    es(i,:,:)=plani4s(Ex(i,:),Ey(i,:),ep,D,Ed(i,:));
end

vonMises = zeros(numfele,4);
for j=1:4
    vonMises(:,j)=sqrt(es(:,j,1).^2+es(:,j,2).^2-es(:,j,1).*es(:,j,2)+3*es(:,j,3).^2);
end

% max. stress in MPa (divided by 100)
smax = max(max(vonMises))/100;

if draw == 1
    figure()
    eldraw2(Ex,Ey,[1 2 0])
    hold on;
    cmap = max(vonMises/100,[],2);
    colormap(jet)
    for i=1:numfele
        fill(Ex(i,:),Ey(i,:),cmap(i))
    end
    % colorbar('off')
    colorbar('southoutside')
    % fpath = 'C:\Users\z5653370\OneDrive - UNSW\Documents\PhD\EMO\Codes\Torque arm\Pareto_result';
    % saveas(gca, fullfile(fpath,sprintf('%d.png',k)));
end
end
