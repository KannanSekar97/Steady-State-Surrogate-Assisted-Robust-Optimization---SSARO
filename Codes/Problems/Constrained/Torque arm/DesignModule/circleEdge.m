function[xp]=circleEdge(dc,topo,rs);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu

x1=dc(topo(2),:);
x2=dc(topo(3),:);
x3=dc(topo(4),:);
x4=dc(topo(5),:);

R1=norm(x1-x2);
R2=norm(x4-x2);

a=sign(x3(1));

v1=(x1-x2)/R1;
v2(1)=-a*v1(2);
v2(2)=a*v1(1);

ca=v1*(x4-x2)'/R2;
sa=v2*(x4-x2)'/R2;

if sa > 0
  alpha=acos(ca);
else
 alpha=2*pi-acos(ca);
end

Ru=R1+(R2-R1)*rs;
phi=alpha*rs;

xp=x2+Ru*(cos(phi)*v1+sin(phi)*v2);
