function[fecoord,locnum,numfenod,d_fecoord]=dsgLagrange(dc,tp,iv,fecoord,numfenod,d_fecoord,d_dc);

% (c) Kurt Maute
%     Department of Aerospace Engineering Sciences
%     University of Colorado, Boulder, USA
%     maute@colorado.edu


irm=iv(1);
ism=iv(2);

switch iv(3)
  case 4
    for is=1:ism+1
      for ir=1:irm+1
    	xr=(ir-1)/irm;
    	xs=(is-1)/ism;
    	n1=(1-xr)*(1-xs);
    	n2=xr*(1-xs);	  
    	n3=xr*xs;
    	n4=(1-xr)*xs;
    	xp=n1*dc(tp(2),:)+n2*dc(tp(3),:) ...
    	  +n3*dc(tp(4),:)+n4*dc(tp(5),:); 
        if (nargin == 5)       
    	  [fecoord,lcn,numfenod]=checknode(xp,fecoord,numfenod);
        else
      	  d_xp=n1*d_dc(tp(2),:)+n2*d_dc(tp(3),:) ...
    	      +n3*d_dc(tp(4),:)+n4*d_dc(tp(5),:); 
    	  [fecoord,lcn,numfenod,d_fecoord]=checknode(xp,fecoord,numfenod,d_fecoord,d_xp);
        end
    	locnum(is,ir)=lcn;
      end
    end
  case 8
    for is=1:ism+1
      for ir=1:irm+1
    	xr=2*(ir-1)/irm-1;
    	xs=2*(is-1)/ism-1;
    	n1=-(1-xr)*(1-xs)*(1+xr+xs)/4;
    	n2=-(1+xr)*(1-xs)*(1-xr+xs)/4;
        n3=-(1+xr)*(1+xs)*(1-xr-xs)/4;
    	n4=-(1-xr)*(1+xs)*(1+xr-xs)/4;
        n5= (1-xr^2)*(1-xs)/2;
        n6= (1+xr)*(1-xs^2)/2;
        n7= (1-xr^2)*(1+xs)/2;
        n8= (1-xr)*(1-xs^2)/2;
    	xp=n1*dc(tp(2),:)+n2*dc(tp(3),:) ...
    	  +n3*dc(tp(4),:)+n4*dc(tp(5),:) ...
          +n5*dc(tp(6),:)+n6*dc(tp(7),:) ...
          +n7*dc(tp(8),:)+n8*dc(tp(9),:);
        if (nargin == 5)       
    	  [fecoord,lcn,numfenod]=checknode(xp,fecoord,numfenod);
        else
    	  xp=n1*d_dc(tp(2),:)+n2*d_dc(tp(3),:) ...
    	    +n3*d_dc(tp(4),:)+n4*d_dc(tp(5),:) ...
            +n5*d_dc(tp(6),:)+n6*d_dc(tp(7),:) ...
            +n7*d_dc(tp(8),:)+n8*d_dc(tp(9),:);
    	  [fecoord,lcn,numfenod,d_fecoord]=checknode(xp,fecoord,numfenod,d_fecoord,d_xp);
        end
     	locnum(is,ir)=lcn;
      end
    end
end
