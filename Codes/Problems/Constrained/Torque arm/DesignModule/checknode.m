function[fecoord,lcn,numfenod,d_fecoord]=checknode(xp,fecoord,numfenod,d_fecoord,d_xp);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu

eps=1.0e-6;

for i=1:numfenod
  distance=norm(xp-fecoord(i,:));
  if (distance < eps)
    lcn=i;
    return;
  end
end

numfenod=numfenod+1;
fecoord(numfenod,:)=xp;
lcn=numfenod;

if (nargin == 5)
 d_fecoord(numfenod,:)=d_xp;
end
