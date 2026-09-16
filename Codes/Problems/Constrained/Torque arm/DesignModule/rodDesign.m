function[fecoord,fetopo,numfenod,numfele]=design;
%
% function[fecoord,fetopo,numfenod,numfele]=design;
%
% design element module for generating fe meshes
%
% output:   fecoord  - coordinates of fe-nodes
%                      numfenod x 3  
%           fetopo   - type and topology of fe-elements
%                      numfele  x maxnodes+1
%           numfenod - number of fe nodes
%           numfele  - number of fe elements
%
% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu


% coordinates of control nodes
%
% for design optimization purposes the coordinates
% can be defined as functions of a design variables s_i

s = [0 2 5 1 -.5 -.5 -.5];

dc=[0.0            0.0;
    0.0            4.0;
    0.0           -4.0;
   12.0-s(1)       0.0; 
   12.0-s(1)       1.0+s(2);
   12.0-s(1)      -1.0-s(2);
   27.0+s(3)       0.0;
   27.0+s(3)       1.0+s(4);
   27.0+s(3)      -1.0-s(4);
   42.0            0.0;
   42.0            2.5;
   42.0           -2.5;
    0.0            5.42+s(5);
    0.0           -5.42-s(5);
   42.0            4.2+s(7);
   42.0           -4.2-s(7);
   40.0            4.2+s(7);
   38.0            3.39+1.056*s(6)-0.056*s(5);
   36.0            3.5+s(6);
   40.0           -4.20-s(7);
   38.0           -3.39-1.056*s(6)+0.056*s(5);
   36.0           -3.50-s(6);   
   -1.0            0.0;   
  -12.0            0.0;  
  -27.0            0.0;  
  -42.0            0.0;  
    4.0            0.0;
   11.0-s(1)-s(2)  0.0;
    6.0            5.0+2*s(5)/3+s(6)/3;
    6.0           -5.0-2*s(5)/3-s(6)/3;
   28.0+s(3)+s(4)  0.0;
   39.5            0.0;
    1.0            0.0;
   42.0            0.0];


% number of design elements

ndele=12;

% topology of design elements

% design elements definition - Part 1
%
% type of design element, list of node numbers in order   
%
% type: 1 - 4 node Lagrange Element
%       2 - Coons Element
%

tp=[2  3   1   23  2   13  1   33  14  3;
    2  2   1   23  27  29  13  2   0   0;
    2  28  4   24  5   29  27  28  0   0;
    2  6   4   24  28  27  30  6   0   0;
    2  27  1   23  3   14  30  27  0   0;
    2  8   19  29  5   8   0   0   0   0;
    2  22  9   6   30  22  0   0   0   0;
    2  31  15  17  18  19  8   7   25  31;  
    2  32  10  26  11  15  31  32  0   0; 
    2  12  10  26  32  31  16  12  0   0;
    2  22  21  20  16  31  7   25  9   22;  
    2  11  10  26  12  16  10  34  15  11 ];


% design elements definition - Part 2
%
% intervals nx.ny and edge types for Coons patches
%
% edge types: 1 - linear edge     (2 nodes)
% edge types: 2 - quadratic edge  (3 nodes)
% edge types: 3 - Bezier edge     (4 nodes)
% edge types: 4 - circular edge   (4 nodes)

iv=zeros(ndele,6);

iv=[ 10  5  4 1 4 1;
      5  5  4 1 1 1;
      5  5  4 1 1 1;
      5  5  4 1 1 1;
      5  5  4 1 1 1;
      5 10  1 1 1 1;
      5 10  1 1 1 1;
      5  5  1 3 1 4;
      5  5  4 1 1 1;
      5  5  4 1 1 1;
      5  5  3 1 4 1;
     10  5  4 1 4 1]; 

% compute maximum number of FE nodes and initialize coordinates 

numfenod=0;

for i=1:ndele
  numfenod=numfenod+(iv(i,1)+1)*(iv(i,2)+1);   
end

fecoord=zeros(numfenod,2);

% initializing arrays

numfenod=0;

% generating fe nodes

for i=1:ndele
  irm=iv(i,1)+1;
  ism=iv(i,2)+1;
  switch tp(i,1);
    case 1
      [fecoord,locnum(i,1:ism,1:irm),numfenod]=dsgLagrange(dc,tp(i,:),iv(i,:),fecoord,numfenod);
    case 2
      [fecoord,locnum(i,1:ism,1:irm),numfenod]=dsgCoons(dc,tp(i,:),iv(i,:),fecoord,numfenod);
  end
end

% generating finite elements

numfele=0;

for i=1:ndele
  irm=iv(i,1);
  ism=iv(i,2);
  for is=1:ism
    for ir=1:irm
      numfele=numfele+1;
      fetopo(numfele,1)=3;
      fetopo(numfele,2)=locnum(i,is,ir);
      fetopo(numfele,3)=locnum(i,is,ir+1);
      fetopo(numfele,4)=locnum(i,is+1,ir+1);
      fetopo(numfele,5)=locnum(i,is+1,ir);
    end
  end
end

% only for plane stress 4-node elements

for i=1:numfele
   switch fetopo(i,1)
   case 3
      for j=1:4  
         Ex(i,j)=fecoord(fetopo(i,j+1),1);
         Ey(i,j)=fecoord(fetopo(i,j+1),2);
      end
   end
end

figure(1)
eldraw2(Ex,Ey,[1 2 0]);

