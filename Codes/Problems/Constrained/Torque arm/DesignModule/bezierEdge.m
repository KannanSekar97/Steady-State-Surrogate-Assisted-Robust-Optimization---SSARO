function[xp]=bezierEdge(dc,topo,rs);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu

x1=dc(topo(2),:);
x2=dc(topo(3),:);
x3=dc(topo(4),:);
x4=dc(topo(5),:);

xp = (1-rs)^3*x1 + (3*rs*(1-rs)^2)*x2 + (3*rs^2*(1-rs))*x3 + rs^3*x4;