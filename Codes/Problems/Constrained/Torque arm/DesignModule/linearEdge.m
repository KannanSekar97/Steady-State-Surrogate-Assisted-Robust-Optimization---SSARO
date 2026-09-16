function[xp]=linearEdge(dc,topo,rs);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu

x1=dc(topo(2),:);
x2=dc(topo(3),:);

xp = (1-rs)*x1 + rs*x2;