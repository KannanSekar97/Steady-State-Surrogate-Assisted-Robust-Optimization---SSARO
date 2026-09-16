function[fecoord,locnum,numfenod,d_fecoord]=dsgCoons(dc,tp,iv,fecoord,numfenod,d_fecoord,d_dc);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu

% build edge topology

edgeTopo=zeros(4,5);
cornerTopo=zeros(4,1);

topPtr=2;

for i=1:4
   cornerTopo(i)=tp(topPtr);
   switch iv(i+2)
   case 1
     edgeTopo(i,1)=1;
     edgeTopo(i,2)=tp(topPtr);
     edgeTopo(i,3)=tp(topPtr+1);
     topPtr=topPtr+1;
   case 2
     edgeTopo(i,1)=2;
     edgeTopo(i,2)=tp(topPtr);
     edgeTopo(i,3)=tp(topPtr+1);
     edgeTopo(i,4)=tp(topPtr+2);
     topPtr=topPtr+2;
   case 3
     edgeTopo(i,1)=3;
     edgeTopo(i,2)=tp(topPtr);
     edgeTopo(i,3)=tp(topPtr+1);
     edgeTopo(i,4)=tp(topPtr+2);
     edgeTopo(i,5)=tp(topPtr+3);
     topPtr=topPtr+3;
   case 4
     edgeTopo(i,1)=4;
     edgeTopo(i,2)=tp(topPtr);
     edgeTopo(i,3)=tp(topPtr+1);
     edgeTopo(i,4)=tp(topPtr+2);
     edgeTopo(i,5)=tp(topPtr+3);
     topPtr=topPtr+3;
   end
end

rsedge=zeros(4,1);
xedge=zeros(4,2);

irm=iv(1);
ism=iv(2);

for is=1:ism+1
   for ir=1:irm+1
  	 xr=(ir-1)/irm;
    xs=(is-1)/ism;
      
    % linear blend functions
      
    B0r=(1-xr);
    B1r=xr;
    B0s=(1-xs);
    B1s=xs;
    
    % local edge coordinates
    
    rsedge(1)=xr;
    rsedge(2)=xs;
    rsedge(3)=1-xr;
    rsedge(4)=1-xs;
    
    % evaluate edge contributions
    
    for iedg=1:4
      switch edgeTopo(iedg,1)
      case 1
         [xedge(iedg,:)]=linearEdge(dc,edgeTopo(iedg,:),rsedge(iedg));
      case 2
         [xedge(iedg,:)]=quadEdge(dc,edgeTopo(iedg,:),rsedge(iedg));
      case 3
         [xedge(iedg,:)]=bezierEdge(dc,edgeTopo(iedg,:),rsedge(iedg));
      case 4
         [xedge(iedg,:)]=circleEdge(dc,edgeTopo(iedg,:),rsedge(iedg));
      end
   end
   
   if (nargin == 7)       
      for iedg=1:4
        switch edgeTopo(iedg,1)
        case 1
           [d_xedge(iedg,:)]=linearEdge(d_dc,edgeTopo(iedg,:),rsedge(iedg));
        case 2
           [d_xedge(iedg,:)]=quadEdge(d_dc,edgeTopo(iedg,:),rsedge(iedg));
        case 3
           [d_xedge(iedg,:)]=bezierEdge(d_dc,edgeTopo(iedg,:),rsedge(iedg));
        case 4
           [d_xedge(iedg,:)]=dcircleEdge(dc,d_dc,edgeTopo(iedg,:),rsedge(iedg));
        end
     end
   end

   % Coons interpolation    
   
   mxp1 = (B0r*B0s)*dc(cornerTopo(1),:);
   mxp2 = (B1r*B0s)*dc(cornerTopo(2),:);
   mxp3 = (B1r*B1s)*dc(cornerTopo(3),:);
   mxp4 = (B0r*B1s)*dc(cornerTopo(4),:);

   if (nargin == 7)       
     d_mxp1 = (B0r*B0s)*d_dc(cornerTopo(1),:);
     d_mxp2 = (B1r*B0s)*d_dc(cornerTopo(2),:);
     d_mxp3 = (B1r*B1s)*d_dc(cornerTopo(3),:);
     d_mxp4 = (B0r*B1s)*d_dc(cornerTopo(4),:);
   end

   % Add new coordintate
   
   xp = B0r*xedge(4,:)+B1r*xedge(2,:)+B0s*xedge(1,:)+B1s*xedge(3,:) ...   
      - (mxp1+mxp2+mxp3+mxp4);

   if (nargin == 5)       
     [fecoord,lcn,numfenod]=checknode(xp,fecoord,numfenod);
   else
     d_xp = B0r*d_xedge(4,:)+B1r*d_xedge(2,:)+B0s*d_xedge(1,:)+B1s*d_xedge(3,:) ...   
          - (d_mxp1+d_mxp2+d_mxp3+d_mxp4);
     [fecoord,lcn,numfenod,d_fecoord]=checknode(xp,fecoord,numfenod,d_fecoord,d_xp);
   end
   
  locnum(is,ir)=lcn;
   
  end
end
